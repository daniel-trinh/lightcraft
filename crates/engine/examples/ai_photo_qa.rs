//! Developer-only real-photo QA for the production native generative command.
//! Run with `cargo run -p lightcraft-engine --example ai_photo_qa --features inpaint -- case.json`.
//! A CUDA build uses `--features inpaint,cuda` and requires a working GPU.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::process::ExitCode;

fn main() -> ExitCode {
    #[cfg(feature = "inpaint")]
    {
        match qa::run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("AI photo QA failed: {e}");
                ExitCode::FAILURE
            }
        }
    }
    #[cfg(not(feature = "inpaint"))]
    {
        eprintln!("AI photo QA requires the engine's inpaint feature (--features inpaint, or --features inpaint,cuda for GPU QA)");
        ExitCode::FAILURE
    }
}

#[cfg(feature = "inpaint")]
mod qa {
    use std::fs::{self, File, OpenOptions};
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    use lightcraft_codecs::{DecodeOptions, EncodeImage, EncodeMeta};
    use lightcraft_engine::catalog::PhotoId;
    use lightcraft_engine::media::SourceRef;
    use lightcraft_engine::pipeline::{MaskView, Overlay};
    use lightcraft_engine::{Selection, Session, SourceLevel};
    use lightcraft_raster::Rgba8;
    use serde::Deserialize;
    use serde_json::{Value, json};

    const MAX_CASE_BYTES: u64 = 1_048_576;
    const MAX_IMAGE_BYTES: u64 = 512 * 1024 * 1024;
    const MAX_PIXELS: usize = 64_000_000;

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct Case {
        source: PathBuf,
        #[serde(default = "default_backend")]
        backend: String,
        model_dir: Option<PathBuf>,
        removal_model: Option<PathBuf>,
        output_dir: PathBuf,
        command: String,
        mask: Value,
        #[serde(default)]
        brush_strokes: Vec<Value>,
        #[serde(default)]
        preview_only: bool,
        #[serde(default)]
        params: Params,
    }

    fn default_backend() -> String {
        "native".into()
    }

    #[derive(Default, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Params {
        prompt: Option<String>,
        negative: Option<String>,
        steps: Option<u32>,
        guidance: Option<f64>,
        seed: Option<u64>,
        edge: Option<usize>,
        denoise: Option<f64>,
        #[serde(rename = "maskMargin")]
        mask_margin: Option<usize>,
    }

    pub fn run() -> Result<(), String> {
        let mut args = std::env::args_os();
        let _program = args.next();
        let case_path = PathBuf::from(args.next().ok_or("provide exactly one JSON case path")?);
        if args.next().is_some() {
            return Err("provide exactly one JSON case path".into());
        }
        let case_bytes = read_limited(&case_path, MAX_CASE_BYTES)?;
        let case: Case = serde_json::from_slice(&case_bytes).map_err(|e| format!("{}: invalid QA case: {e}", case_path.display()))?;
        if !matches!(case.command.as_str(), "photo.generativeFill" | "photo.generativeRemove") {
            return Err("command must be photo.generativeFill or photo.generativeRemove".into());
        }
        if !matches!(case.backend.as_str(), "native" | "klein") {
            return Err("backend must be native or klein".into());
        }
        let klein = case.backend == "klein";
        let mask = case.mask.as_object().ok_or("mask must be an object of mask.add parameters")?;
        let kind = mask.get("kind").and_then(Value::as_str).ok_or("mask.kind is required")?;
        if !matches!(kind, "radial" | "linear" | "brush") || mask.contains_key("seg") {
            return Err("QA accepts radial, linear, or brush masks only; AI segmentation is outside this test".into());
        }
        if case.brush_strokes.len() > 32 || (kind == "brush" && case.brush_strokes.is_empty()) {
            return Err("brush masks need 1..32 brushStrokes".into());
        }
        if kind != "brush" && !case.brush_strokes.is_empty() {
            return Err("brushStrokes require mask.kind brush".into());
        }
        let mut point_count = 0usize;
        for stroke in &case.brush_strokes {
            let points = stroke.get("points").and_then(Value::as_array).ok_or("each brush stroke needs points")?;
            point_count = point_count.saturating_add(points.len());
            if points.is_empty() || point_count > 4096 {
                return Err("brush strokes need nonempty points, with at most 4096 points in total".into());
            }
            for point in points {
                let pair = point.as_array().ok_or("brush points must be [x,y] pairs")?;
                if pair.len() != 2 || pair.iter().any(|v| v.as_f64().is_none_or(|n| !n.is_finite() || !(0.0..=1.0).contains(&n))) {
                    return Err("brush points must be finite normalized [x,y] pairs in 0..1".into());
                }
            }
            for (name, lo, hi) in [("size", 0.001, 1.0), ("feather", 0.0, 100.0), ("flow", 0.0, 100.0), ("density", 0.0, 100.0)] {
                if let Some(value) = stroke.get(name)
                    && value.as_f64().is_none_or(|n| !n.is_finite() || !(lo..=hi).contains(&n))
                {
                    return Err(format!("brush {name} must be finite and in {lo}..{hi}"));
                }
            }
        }
        if cfg!(feature = "cuda") && std::env::var("LIGHTCRAFT_AI_DEVICE").as_deref() == Ok("cpu") {
            return Err("CUDA QA refuses LIGHTCRAFT_AI_DEVICE=cpu; remove the override to verify GPU inference".into());
        }
        let source = fs::canonicalize(&case.source).map_err(|e| format!("source {}: {e}", case.source.display()))?;
        if !source.is_file() {
            return Err(format!("source is not a regular file: {}", source.display()));
        }
        let mut session = Session::new().with_fs();
        if klein {
            let path = case.model_dir.as_deref().ok_or("Klein QA requires modelDir")?;
            session.generator.klein_dir = Some(fs::canonicalize(path).map_err(|e| format!("modelDir {}: {e}", path.display()))?);
            session.generator.klein_model_dir().map_err(|e| format!("Klein model is unavailable: {e}"))?;
        } else if case.command == "photo.generativeRemove" {
            if let Some(path) = &case.removal_model {
                session.generator.removal = Some(fs::canonicalize(path).map_err(|e| format!("removalModel {}: {e}", path.display()))?);
            }
            session.generator.removal_model().map_err(|e| format!("removal model is unavailable: {e}"))?;
        } else {
            let path = case.model_dir.as_deref().ok_or("Fill QA requires modelDir")?;
            session.generator.dir = Some(fs::canonicalize(path).map_err(|e| format!("modelDir {}: {e}", path.display()))?);
            session.generator.model_dir().map_err(|e| format!("local model is unavailable: {e}"))?;
        }
        fs::create_dir_all(&case.output_dir).map_err(|e| format!("outputDir {}: {e}", case.output_dir.display()))?;
        let output_dir = fs::canonicalize(&case.output_dir).map_err(|e| format!("outputDir {}: {e}", case.output_dir.display()))?;
        let copied = copy_new(&source, &output_dir)?;
        let imported = session.execute("library.import", &json!({"paths": [copied.to_string_lossy()]})).map_err(|e| e.to_string())?;
        let id =
            imported["imported"].get(0).and_then(Value::as_u64).map(PhotoId).ok_or_else(|| format!("copied photo was not imported: {imported}"))?;
        session.selection = Selection::single(id);
        session.execute("mask.add", &case.mask).map_err(|e| format!("mask.add failed: {e}"))?;
        let mask_id = session.active_mask.ok_or("mask.add did not select a mask")?;
        for mut stroke in case.brush_strokes {
            stroke.as_object_mut().ok_or("brush stroke must be an object")?.insert("id".into(), json!(mask_id));
            session.execute("mask.brushStroke", &stroke).map_err(|e| format!("mask.brushStroke failed: {e}"))?;
        }
        let source_photo = session.catalog.photo(id).ok_or("imported photo vanished")?.clone();
        let (expected, expected_mask) = render_expected(&mut session, id, mask_id)?;
        let prefix = copied.file_stem().and_then(|s| s.to_str()).ok_or("copied photo has no usable file stem")?;
        let expected_path = write_png_new(&output_dir, &format!("{prefix}-expected-source.png"), &expected)?;
        let mask_path = write_png_new(&output_dir, &format!("{prefix}-expected-mask.png"), &expected_mask)?;
        if case.preview_only {
            let preview = json!({"status":"preview", "expectedSource":expected_path, "expectedMask":mask_path});
            let path = output_dir.join(format!("{prefix}-preview.json"));
            write_new(&path, &serde_json::to_vec_pretty(&preview).map_err(|e| e.to_string())?)?;
            println!("{}", path.display());
            return Ok(());
        }

        let mut params = json!({
            "maskId": mask_id,
            "backend": case.backend,
            "prompt": case.params.prompt.unwrap_or_default(),
            "negative": case.params.negative.unwrap_or_default(),
            "steps": case.params.steps.unwrap_or(if klein { 4 } else { 24 }),
            "guidance": case.params.guidance.unwrap_or(if klein { 1.0 } else { 7.5 }),
            "seed": case.params.seed.unwrap_or(42),
            "edge": case.params.edge.unwrap_or(if klein { 1024 } else { 512 }),
            "denoise": case.params.denoise.unwrap_or(if klein { 1.0 } else { 0.99 }),
        });
        if let Some(margin) = case.params.mask_margin {
            params["maskMargin"] = json!(margin);
        }
        let undo_before = session.undo.len();
        let started = Instant::now();
        let result = session.execute(&case.command, &params).map_err(|e| format!("{} failed: {e}", case.command))?;
        let elapsed = started.elapsed().as_secs_f64();
        let result_id = result["id"].as_u64().map(PhotoId).ok_or_else(|| format!("generation returned no photo id: {result}"))?;
        let output_path = PathBuf::from(result["path"].as_str().ok_or_else(|| format!("generation returned no output path: {result}"))?);
        let generated = decode_output(&output_path)?;
        if (generated.width, generated.height) != (expected.width, expected.height) {
            return Err(format!(
                "output is {}x{}, expected full resolution {}x{}",
                generated.width, generated.height, expected.width, expected.height
            ));
        }
        if (expected_mask.width, expected_mask.height) != (expected.width, expected.height) {
            return Err("expected source and mask dimensions differ".into());
        }
        let mut outside = 0usize;
        let mut selected = 0usize;
        let mut changed = 0usize;
        for ((before, after), mask_px) in expected.data.iter().zip(&generated.data).zip(&expected_mask.data) {
            if mask_px[0] == 0 {
                if before != after {
                    return Err(format!("pixel outside mask changed at image offset {}", outside + selected));
                }
                outside += 1;
            } else {
                selected += 1;
                if before != after {
                    changed += 1;
                }
            }
        }
        if outside == 0 || selected == 0 || changed == 0 {
            return Err(format!(
                "QA needs unselected and visibly changed selected pixels (outside={outside}, selected={selected}, changed={changed})"
            ));
        }
        if session.catalog.photo(id).is_none_or(|p| p.develop != source_photo.develop || p.source != source_photo.source) {
            return Err("generation changed the source photo or its develop settings".into());
        }
        if session.undo.len() != undo_before + 1 {
            return Err(format!("generation should add one undo entry; before={undo_before}, after={}", session.undo.len()));
        }
        session.execute("edit.undo", &json!({})).map_err(|e| format!("undo failed: {e}"))?;
        if session.catalog.photo(result_id).is_some()
            || session.catalog.photo(id).is_none_or(|p| p.develop != source_photo.develop || p.source != source_photo.source)
            || !output_path.is_file()
        {
            return Err("undo did not remove only the generated catalog photo while retaining its PNG and source".into());
        }
        let backend = if cfg!(feature = "cuda") {
            "native NVIDIA CUDA required (CPU override refused)"
        } else {
            "native platform backend (CPU on Windows/Linux; Metal preferred on macOS)"
        };
        let report = json!({
            "status": "passed",
            "command": case.command,
            "backend": backend,
            "source": source,
            "copiedSource": copied,
            "expectedSource": expected_path,
            "expectedMask": mask_path,
            "resultPath": output_path,
            "fullWidth": expected.width,
            "fullHeight": expected.height,
            "outsidePixelsIdentical": outside,
            "selectedPixels": selected,
            "changedSelectedPixels": changed,
            "elapsedSeconds": elapsed,
            "undoRemovedCatalogResult": true,
            "resultFileRetained": true,
            "params": params,
        });
        let report_path = output_dir.join(format!("{prefix}-qa.json"));
        write_new(&report_path, &serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)?;
        println!("{}", report_path.display());
        Ok(())
    }

    fn read_limited(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if file.metadata().map_err(|e| format!("{}: {e}", path.display()))?.len() > limit {
            return Err(format!("{} exceeds the {} MiB input limit", path.display(), limit / 1_048_576));
        }
        let mut bytes = Vec::new();
        file.take(limit + 1).read_to_end(&mut bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        if bytes.len() as u64 > limit {
            return Err(format!("{} exceeds the {} MiB input limit", path.display(), limit / 1_048_576));
        }
        Ok(bytes)
    }

    fn copy_new(source: &Path, dir: &Path) -> Result<PathBuf, String> {
        if source.metadata().map_err(|e| format!("{}: {e}", source.display()))?.len() > MAX_IMAGE_BYTES {
            return Err("source exceeds the 512 MiB QA input limit".into());
        }
        let ext = source.extension().and_then(|s| s.to_str()).ok_or("source needs a filename extension")?;
        if !matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "tif" | "tiff" | "webp") {
            return Err("QA source must be a supported rendered photo (PNG, JPEG, TIFF, WebP)".into());
        }
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        for n in 0..1024 {
            let path = dir.join(format!("ai-photo-qa-{}-{nonce}-{n}.{ext}", std::process::id()));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut output) => {
                    let copied = (|| -> Result<(), String> {
                        let input = File::open(source).map_err(|e| format!("{}: {e}", source.display()))?;
                        let bytes = std::io::copy(&mut input.take(MAX_IMAGE_BYTES + 1), &mut output)
                            .map_err(|e| format!("copying {}: {e}", source.display()))?;
                        if bytes > MAX_IMAGE_BYTES {
                            return Err("source grew beyond the 512 MiB QA input limit while copying".into());
                        }
                        output.sync_all().map_err(|e| format!("{}: {e}", path.display()))?;
                        Ok(())
                    })();
                    if copied.is_err() {
                        let _ = fs::remove_file(&path);
                    }
                    copied?;
                    return Ok(path);
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(format!("{}: {e}", path.display())),
            }
        }
        Err("could not reserve a unique QA photo name".into())
    }

    fn render_expected(session: &mut Session, id: PhotoId, mask_id: u32) -> Result<(Rgba8, Rgba8), String> {
        let photo = session.catalog.photo(id).ok_or("source photo vanished")?.clone();
        if u64::from(photo.width).saturating_mul(u64::from(photo.height)) > MAX_PIXELS as u64 {
            return Err("source exceeds the 64 MP QA limit".into());
        }
        let (w, h) = lightcraft_engine::pipeline::native_output_size(photo.width as usize, photo.height as usize, &photo.develop);
        if !w.is_finite() || !h.is_finite() || w < 1.0 || h < 1.0 {
            return Err("source has invalid native dimensions".into());
        }
        let edge = (w.round() as usize).max(h.round() as usize);
        if (w.round() as usize).saturating_mul(h.round() as usize) > MAX_PIXELS {
            return Err("source exceeds the 64 MP QA limit".into());
        }
        let mut source_job = session.render_job(id, edge, edge, false, true).ok_or("could not render expected source")?;
        source_job.level = SourceLevel::Full;
        source_job.source = session.media.source_ref(&photo, SourceLevel::Full);
        let rendered = source_job.run();
        let source = rendered.rendered?.image;
        let mask_id = u16::try_from(mask_id).map_err(|_| "mask id is too large for rendering")?;
        let mut mask_job = session.render_job(id, edge, edge, false, true).ok_or("could not render expected mask")?.with_overlay(Overlay::Mask {
            id: mask_id,
            view: MaskView::WhiteOnBlack,
            color: [255; 3],
            opacity: 100,
        });
        mask_job.level = SourceLevel::Full;
        mask_job.source = match rendered.loaded {
            Some(decoded) => SourceRef::Loaded(Box::new(decoded)),
            None => session.media.source_ref(&photo, SourceLevel::Full),
        };
        let mask = mask_job.run().rendered?.image;
        Ok((source, mask))
    }

    fn write_png_new(dir: &Path, file_name: &str, image: &Rgba8) -> Result<PathBuf, String> {
        let bytes = lightcraft_codecs::encode_png(&EncodeImage::rgba8(image), &EncodeMeta::default()).map_err(|e| e.to_string())?;
        let path = dir.join(file_name);
        write_new(&path, &bytes)?;
        Ok(path)
    }

    fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        file.write_all(bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        file.sync_all().map_err(|e| format!("{}: {e}", path.display()))
    }

    fn decode_output(path: &Path) -> Result<Rgba8, String> {
        let bytes = read_limited(path, MAX_IMAGE_BYTES)?;
        lightcraft_codecs::decode(&bytes, DecodeOptions { max_pixels: MAX_PIXELS as u64, ..Default::default() })
            .map(|image| image.to_srgb8())
            .map_err(|e| format!("{}: invalid generated PNG: {e}", path.display()))
    }
}
