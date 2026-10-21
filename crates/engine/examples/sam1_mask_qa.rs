//! Developer-only real-photo QA for production SAM 1 Object-mask commands.
//! `cargo run -p lightcraft-engine --example sam1_mask_qa --features sam -- case.json`
//! Add `,cuda` for NVIDIA QA. No model, photo, or output is bundled with this example.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::process::ExitCode;

fn main() -> ExitCode {
    #[cfg(feature = "sam")]
    {
        match qa::run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("SAM 1 mask QA failed: {e}");
                ExitCode::FAILURE
            }
        }
    }
    #[cfg(not(feature = "sam"))]
    {
        eprintln!("SAM 1 mask QA requires --features sam (or --features sam,cuda for NVIDIA GPU QA)");
        ExitCode::FAILURE
    }
}

#[cfg(feature = "sam")]
mod qa {
    use std::fs::{self, File, OpenOptions};
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    use lightcraft_codecs::{EncodeImage, EncodeMeta};
    use lightcraft_develop::{MaskShape, ObjectModel, SegMask};
    use lightcraft_engine::catalog::PhotoId;
    use lightcraft_engine::media::SourceRef;
    use lightcraft_engine::pipeline::{MaskView, Overlay};
    use lightcraft_engine::{Selection, Session, SourceLevel};
    use lightcraft_raster::Rgba8;
    use serde::Deserialize;
    use serde_json::json;

    const MAX_CASE_BYTES: u64 = 1_048_576;
    const MAX_IMAGE_BYTES: u64 = 512 * 1024 * 1024;
    const MAX_PIXELS: usize = 64_000_000;

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct Case {
        source: PathBuf,
        sam1_model: PathBuf,
        output_dir: PathBuf,
        positive: Vec<[f64; 2]>,
        #[serde(default)]
        negative: Vec<[f64; 2]>,
        #[serde(default = "yes")]
        refine_detail: bool,
    }

    fn yes() -> bool {
        true
    }

    pub fn run() -> Result<(), String> {
        let mut args = std::env::args_os();
        let _program = args.next();
        let case_path = PathBuf::from(args.next().ok_or("provide exactly one JSON case path")?);
        if args.next().is_some() {
            return Err("provide exactly one JSON case path".into());
        }
        let case: Case = serde_json::from_slice(&read_limited(&case_path, MAX_CASE_BYTES)?)
            .map_err(|e| format!("{}: invalid QA case: {e}", case_path.display()))?;
        validate_points(&case)?;
        let source = fs::canonicalize(&case.source).map_err(|e| format!("source {}: {e}", case.source.display()))?;
        if !source.is_file() {
            return Err(format!("source is not a regular file: {}", source.display()));
        }
        let original_hash = hash_file(&source)?;
        let model = fs::canonicalize(&case.sam1_model).map_err(|e| format!("sam1Model {}: {e}", case.sam1_model.display()))?;
        let device = lightcraft_segment::best_device().map_err(|e| format!("cannot select AI device: {e}"))?;
        if cfg!(feature = "cuda") && !device.is_cuda() {
            return Err("CUDA QA requires an NVIDIA device; remove LIGHTCRAFT_AI_DEVICE=cpu or repair the CUDA runtime".into());
        }
        let device_name = if device.is_cuda() {
            "NVIDIA CUDA"
        } else if device.is_metal() {
            "Metal"
        } else {
            "CPU"
        };
        fs::create_dir_all(&case.output_dir).map_err(|e| format!("outputDir {}: {e}", case.output_dir.display()))?;
        let output_dir = fs::canonicalize(&case.output_dir).map_err(|e| format!("outputDir {}: {e}", case.output_dir.display()))?;
        let copied = copy_new(&source, &output_dir)?;
        if hash_file(&copied)? != original_hash {
            return Err("copied source differs from original".into());
        }

        let mut session = Session::new().with_fs();
        session.segmenter.sam1_model = Some(model.clone());
        session.segmenter.sam1_model_file().map_err(|e| format!("local model unavailable: {e}"))?;
        let imported = session.execute("library.import", &json!({"paths":[copied.to_string_lossy()]})).map_err(|e| e.to_string())?;
        let id = imported["imported"]
            .get(0)
            .and_then(serde_json::Value::as_u64)
            .map(PhotoId)
            .ok_or_else(|| format!("copied photo was not imported: {imported}"))?;
        session.selection = Selection::single(id);
        let photo_before = session.catalog.photo(id).ok_or("imported photo vanished")?.clone();
        let undo_before = session.undo.len();
        let started = Instant::now();
        session.execute("mask.add", &json!({"kind":"object","model":"sam1"})).map_err(|e| format!("mask.add failed: {e}"))?;
        let mask_id = session.active_mask.ok_or("mask.add did not select a mask")?;
        let prepare_seconds = started.elapsed().as_secs_f64();
        let mut click_seconds = Vec::new();
        // Synchronous mask creation prepares the embedding; clicks then reuse it.
        for (point, exclude) in case.positive.iter().map(|p| (p, false)).chain(case.negative.iter().map(|p| (p, true))) {
            let t = Instant::now();
            session
                .execute("mask.objectPoint", &json!({"id":mask_id,"x":point[0],"y":point[1],"exclude":exclude}))
                .map_err(|e| format!("mask.objectPoint failed: {e}"))?;
            click_seconds.push(t.elapsed().as_secs_f64());
        }
        // A one-point case still exercises a second decoder pass on the cached image embedding.
        let cache_probe_repeated_first_point = case.positive.len() + case.negative.len() == 1;
        if cache_probe_repeated_first_point {
            let p = case.positive.first().ok_or("positive click missing")?;
            let t = Instant::now();
            session
                .execute("mask.objectPoint", &json!({"id":mask_id,"x":p[0],"y":p[1]}))
                .map_err(|e| format!("cache-probe mask.objectPoint failed: {e}"))?;
            click_seconds.push(t.elapsed().as_secs_f64());
        }
        let refine_started = if case.refine_detail {
            session.execute("mask.refineDetail", &json!({"id":mask_id,"component":0}))
                .map_err(|e| format!("mask.refineDetail failed: {e}"))?["started"].as_bool().unwrap_or(false)
        } else {
            false
        };
        let inference_seconds = started.elapsed().as_secs_f64();

        let (seg, details) = stored_seg(&session, id, mask_id)?;
        let logits = seg.logits().ok_or("stored segmentation is damaged")?;
        if logits.len() != (seg.side as usize).saturating_mul(seg.side as usize) || logits.iter().any(|v| !v.is_finite()) {
            return Err("stored segmentation has invalid dimensions or non-finite logits".into());
        }
        let positive_logits = logits.iter().filter(|v| **v > 0.0).count();
        if positive_logits == 0 || positive_logits == logits.len() {
            return Err(format!("segmentation is empty or covers everything ({positive_logits}/{})", logits.len()));
        }
        let (rendered_source, rendered_mask, overlay) = render_outputs(&mut session, id, mask_id)?;
        if (rendered_source.width, rendered_source.height) != (rendered_mask.width, rendered_mask.height)
            || (rendered_source.width, rendered_source.height) != (overlay.width, overlay.height)
        {
            return Err("source, mask, and overlay dimensions differ".into());
        }
        let selected_pixels = rendered_mask.data.iter().filter(|p| p[0] >= 128).count();
        if selected_pixels == 0 || selected_pixels == rendered_mask.data.len() {
            return Err(format!("rendered mask is empty or full ({selected_pixels}/{})", rendered_mask.data.len()));
        }
        let prefix = copied.file_stem().and_then(|s| s.to_str()).ok_or("copied source has no usable file stem")?;
        let source_png = write_png_new(&output_dir, &format!("{prefix}-source.png"), &rendered_source)?;
        let mask_png = write_png_new(&output_dir, &format!("{prefix}-mask.png"), &rendered_mask)?;
        let overlay_png = write_png_new(&output_dir, &format!("{prefix}-overlay.png"), &overlay)?;
        if session.catalog.photo(id).is_none_or(|p| p.source != photo_before.source) {
            return Err("masking changed the imported source reference".into());
        }
        let after = session.develop_of(id).ok_or("develop settings vanished")?;
        let mut expected = (*photo_before.develop).clone();
        expected.masks = after.masks.clone();
        if *after != expected {
            return Err("masking changed develop settings outside masks".into());
        }
        if hash_file(&source)? != original_hash || hash_file(&copied)? != original_hash {
            return Err("masking changed original or copied source bytes".into());
        }
        let undo_entries = session.undo.len().saturating_sub(undo_before);
        if undo_entries == 0 || undo_entries > 68 {
            return Err(format!("unexpected mask undo count: {undo_entries}"));
        }
        for _ in 0..undo_entries {
            session.execute("edit.undo", &json!({})).map_err(|e| format!("mask undo failed: {e}"))?;
        }
        if session.develop_of(id).is_none_or(|d| *d != *photo_before.develop)
            || session.catalog.photo(id).is_none_or(|p| p.source != photo_before.source)
            || hash_file(&source)? != original_hash
            || hash_file(&copied)? != original_hash
        {
            return Err("undo did not restore the imported source and develop settings".into());
        }

        let report = json!({
            "status":"passed", "source":source, "copiedSource":copied, "sam1Model":model,
            "device":device_name, "sourceHash":"fnv1a64", "sourceHashValue":format!("{original_hash:016x}"),
            "maskId":mask_id, "modelRecorded":"sam1", "segSide":seg.side, "segRect":seg.rect,
            "segPositiveLogits":positive_logits, "segCells":logits.len(), "detailPatches":details,
            "width":rendered_source.width, "height":rendered_source.height,
            "selectedPixels":selected_pixels, "pixelCount":rendered_mask.data.len(),
            "prepareSeconds":prepare_seconds, "clickSeconds":click_seconds,
            "cacheProbeRepeatedFirstPoint":cache_probe_repeated_first_point,
            "detailRequested":case.refine_detail, "detailStarted":refine_started,
            "inferenceSeconds":inference_seconds, "undoEntriesReverted":undo_entries,
            "sourcePreserved":true, "developOutsideMasksPreserved":true,
            "sourcePng":source_png, "maskPng":mask_png, "overlayPng":overlay_png,
        });
        let path = output_dir.join(format!("{prefix}-sam1-qa.json"));
        write_new(&path, &serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)?;
        println!("{}", path.display());
        Ok(())
    }

    fn validate_points(case: &Case) -> Result<(), String> {
        let total = case.positive.len().saturating_add(case.negative.len());
        if case.positive.is_empty() || total > 64 {
            return Err("provide 1..64 clicks including a positive click".into());
        }
        for p in case.positive.iter().chain(&case.negative) {
            if p.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)) {
                return Err("clicks must be finite normalized [x,y] pairs inside 0..1".into());
            }
        }
        Ok(())
    }

    fn stored_seg(session: &Session, id: PhotoId, mask_id: u32) -> Result<(SegMask, usize), String> {
        let d = session.develop_of(id).ok_or("develop settings vanished")?;
        let mask = d.masks.iter().find(|m| m.id == mask_id).ok_or("Object mask vanished")?;
        let component = mask.components.iter().find(|c| matches!(c.shape, MaskShape::Object { .. })).ok_or("Object component vanished")?;
        let MaskShape::Object { model: ObjectModel::Sam1, seg: Some(seg), detail, .. } = &component.shape else {
            return Err("Object mask did not persist SAM 1 with a segmentation".into());
        };
        Ok((seg.clone(), detail.len()))
    }

    fn render_outputs(session: &mut Session, id: PhotoId, mask_id: u32) -> Result<(Rgba8, Rgba8, Rgba8), String> {
        let photo = session.catalog.photo(id).ok_or("source photo vanished")?.clone();
        if u64::from(photo.width).saturating_mul(u64::from(photo.height)) > MAX_PIXELS as u64 {
            return Err("source exceeds the 64 MP QA limit".into());
        }
        let (w, h) = lightcraft_engine::pipeline::native_output_size(photo.width as usize, photo.height as usize, &photo.develop);
        if !w.is_finite() || !h.is_finite() || w < 1.0 || h < 1.0 {
            return Err("invalid native output dimensions".into());
        }
        let (w, h) = (w.round() as usize, h.round() as usize);
        if w.saturating_mul(h) > MAX_PIXELS {
            return Err("output exceeds the 64 MP QA limit".into());
        }
        let edge = w.max(h);
        let mask_id = u16::try_from(mask_id).map_err(|_| "mask id is too large for overlay rendering")?;
        let mut source_job = session.render_job(id, edge, edge, false, true).ok_or("could not render source")?;
        source_job.level = SourceLevel::Full;
        source_job.source = session.media.source_ref(&photo, SourceLevel::Full);
        let source_result = source_job.run();
        let source = source_result.rendered?.image;
        let loaded = source_result.loaded;
        let mut mask_job = session.render_job(id, edge, edge, false, true).ok_or("could not render mask")?.with_overlay(Overlay::Mask {
            id: mask_id,
            view: MaskView::WhiteOnBlack,
            color: [255; 3],
            opacity: 100,
        });
        mask_job.level = SourceLevel::Full;
        mask_job.source = match loaded {
            Some(decoded) => SourceRef::Loaded(Box::new(decoded)),
            None => session.media.source_ref(&photo, SourceLevel::Full),
        };
        let mask = mask_job.run().rendered?.image;
        let mut overlay_job = session.render_job(id, edge, edge, false, true).ok_or("could not render mask overlay")?.with_overlay(Overlay::Mask {
            id: mask_id,
            view: MaskView::Color,
            color: [255, 55, 45],
            opacity: 55,
        });
        overlay_job.level = SourceLevel::Full;
        overlay_job.source = session.media.source_ref(&photo, SourceLevel::Full);
        let overlay = overlay_job.run().rendered?.image;
        Ok((source, mask, overlay))
    }

    fn read_limited(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if file.metadata().map_err(|e| e.to_string())?.len() > limit {
            return Err(format!("{} exceeds the input limit", path.display()));
        }
        let mut bytes = Vec::new();
        file.take(limit + 1).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        if bytes.len() as u64 > limit {
            return Err(format!("{} exceeds the input limit", path.display()));
        }
        Ok(bytes)
    }

    fn copy_new(source: &Path, dir: &Path) -> Result<PathBuf, String> {
        if source.metadata().map_err(|e| e.to_string())?.len() > MAX_IMAGE_BYTES {
            return Err("source exceeds the 512 MiB QA limit".into());
        }
        let ext = source.extension().and_then(|s| s.to_str()).ok_or("source needs a filename extension")?;
        if !matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "tif" | "tiff" | "webp") {
            return Err("QA source must be PNG, JPEG, TIFF, or WebP".into());
        }
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        for n in 0..1024 {
            let path = dir.join(format!("sam1-mask-qa-{}-{nonce}-{n}.{ext}", std::process::id()));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut output) => {
                    let copied = (|| -> Result<(), String> {
                        let input = File::open(source).map_err(|e| e.to_string())?;
                        let bytes = std::io::copy(&mut input.take(MAX_IMAGE_BYTES + 1), &mut output).map_err(|e| e.to_string())?;
                        if bytes > MAX_IMAGE_BYTES {
                            return Err("source grew beyond the 512 MiB QA limit".into());
                        }
                        output.sync_all().map_err(|e| e.to_string())
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
        Err("could not reserve a unique QA source name".into())
    }

    fn hash_file(path: &Path) -> Result<u64, String> {
        let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if file.metadata().map_err(|e| e.to_string())?.len() > MAX_IMAGE_BYTES {
            return Err("source exceeds the 512 MiB QA limit".into());
        }
        let mut h = 0xcbf29ce484222325_u64;
        let mut buf = [0u8; 64 * 1024];
        let mut total = 0u64;
        loop {
            let n = file.read(&mut buf).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            total = total.saturating_add(n as u64);
            if total > MAX_IMAGE_BYTES {
                return Err("source grew beyond the 512 MiB QA limit".into());
            }
            for b in &buf[..n] {
                h = (h ^ u64::from(*b)).wrapping_mul(0x100000001b3);
            }
        }
        Ok(h)
    }

    fn write_png_new(dir: &Path, name: &str, image: &Rgba8) -> Result<PathBuf, String> {
        let bytes = lightcraft_codecs::encode_png(&EncodeImage::rgba8(image), &EncodeMeta::default()).map_err(|e| e.to_string())?;
        let path = dir.join(name);
        write_new(&path, &bytes)?;
        Ok(path)
    }

    fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        file.write_all(bytes).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())
    }
}
