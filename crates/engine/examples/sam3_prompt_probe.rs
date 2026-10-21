//! Developer-only probe of SAM 3's concept detector on one developed, uncropped photo.
//! `cargo run -p lightcraft-engine --example sam3_prompt_probe --features sam -- case.json`
//! Add `,cuda` for NVIDIA QA. This never changes the source photo or production settings.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::process::ExitCode;

fn main() -> ExitCode {
    #[cfg(feature = "sam")]
    {
        match probe::run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("SAM 3 prompt probe failed: {e}");
                ExitCode::FAILURE
            }
        }
    }
    #[cfg(not(feature = "sam"))]
    {
        eprintln!("SAM 3 prompt probe requires --features sam (or --features sam,cuda for NVIDIA QA)");
        ExitCode::FAILURE
    }
}

#[cfg(feature = "sam")]
mod probe {
    use std::fs::{self, File, OpenOptions};
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use lightcraft_codecs::{EncodeImage, EncodeMeta};
    use lightcraft_engine::catalog::PhotoId;
    use lightcraft_engine::{Selection, Session};
    use lightcraft_raster::Rgba8;
    use lightcraft_segment::{MASK_SIDE, Sam3};
    use serde::Deserialize;
    use serde_json::json;

    const MAX_CASE_BYTES: u64 = 1_048_576;
    const MAX_IMAGE_BYTES: u64 = 512 * 1024 * 1024;
    const MAX_PIXELS: u64 = 64_000_000;
    const DEFAULT_QUERIES: &[&str] = &["main subject", "subject", "object", "foreground", "foreground object", "person", "rocket", "cup"];

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct Case {
        source: PathBuf,
        model_dir: PathBuf,
        output_dir: PathBuf,
        #[serde(default)]
        queries: Vec<String>,
        #[serde(default)]
        save_masks: bool,
    }

    struct Row {
        text: String,
        presence: f32,
        max_confidence: f32,
        top_bbox: Option<[f32; 4]>,
        above_threshold_count: usize,
        positive_cells: usize,
        elapsed_seconds: f64,
        mask: Option<Vec<f32>>,
        mask_path: Option<PathBuf>,
    }

    pub fn run() -> Result<(), String> {
        let mut args = std::env::args_os();
        let _ = args.next();
        let path = PathBuf::from(args.next().ok_or("provide exactly one JSON case path")?);
        if args.next().is_some() {
            return Err("provide exactly one JSON case path".into());
        }
        let case: Case =
            serde_json::from_slice(&read_limited(&path, MAX_CASE_BYTES)?).map_err(|e| format!("{}: invalid QA case: {e}", path.display()))?;
        let queries: Vec<String> = if case.queries.is_empty() { DEFAULT_QUERIES.iter().map(|s| (*s).into()).collect() } else { case.queries };
        if queries.len() > 16 || queries.iter().any(|q| q.trim().is_empty() || q.len() > 128) {
            return Err("provide 1..16 nonempty queries of at most 128 bytes each".into());
        }
        let source = fs::canonicalize(&case.source).map_err(|e| format!("source {}: {e}", case.source.display()))?;
        if !source.is_file() || source.metadata().map_err(|e| e.to_string())?.len() > MAX_IMAGE_BYTES {
            return Err("source must be a photo file of at most 512 MiB".into());
        }
        let model_dir = fs::canonicalize(&case.model_dir).map_err(|e| format!("modelDir {}: {e}", case.model_dir.display()))?;
        if !lightcraft_segment::is_model_dir(&model_dir) {
            return Err("modelDir needs model.safetensors, vocab.json and merges.txt".into());
        }
        fs::create_dir_all(&case.output_dir).map_err(|e| format!("outputDir {}: {e}", case.output_dir.display()))?;
        let output_dir = fs::canonicalize(&case.output_dir).map_err(|e| e.to_string())?;
        let device = lightcraft_segment::best_device().map_err(|e| e.to_string())?;
        if cfg!(feature = "cuda") && !device.is_cuda() {
            return Err("CUDA QA requires an NVIDIA device; remove LIGHTCRAFT_AI_DEVICE=cpu or repair CUDA".into());
        }
        let device_name = if device.is_cuda() {
            "NVIDIA CUDA"
        } else if device.is_metal() {
            "Metal"
        } else {
            "CPU"
        };

        // Match the production worker's developed, uncropped, mask-free 1008-pixel input.
        let mut session = Session::new().with_fs();
        let imported = session.execute("library.import", &json!({"paths":[source.to_string_lossy()]})).map_err(|e| e.to_string())?;
        let id = imported["imported"].get(0).and_then(serde_json::Value::as_u64).map(PhotoId).ok_or("source was not imported")?;
        session.selection = Selection::single(id);
        let photo = session.catalog.photo(id).ok_or("imported photo vanished")?.clone();
        if u64::from(photo.width).saturating_mul(u64::from(photo.height)) > MAX_PIXELS {
            return Err("source exceeds 64 MP QA limit".into());
        }
        let mut develop = (*photo.develop).clone();
        develop.masks.clear();
        let image = session
            .preview_job(id, lightcraft_segment::INPUT, lightcraft_segment::INPUT, false, &develop)
            .ok_or("could not render model input")?
            .run()
            .rendered
            .map_err(|e| e.to_string())?
            .image;
        let rgb: Vec<u8> = image.data.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
        let load_start = Instant::now();
        let mut model = Sam3::load_on(&model_dir, device).map_err(|e| e.to_string())?;
        lightcraft_segment::sync(&model).map_err(|e| e.to_string())?;
        let load_seconds = load_start.elapsed().as_secs_f64();
        let encode_start = Instant::now();
        let mut encoded = model.encode(&rgb, image.width, image.height).map_err(|e| e.to_string())?;
        lightcraft_segment::sync(&model).map_err(|e| e.to_string())?;
        let encode_seconds = encode_start.elapsed().as_secs_f64();

        let mut rows = Vec::with_capacity(queries.len());
        for text in queries {
            let started = Instant::now();
            let output = model.detect(&mut encoded, &text).map_err(|e| format!("{text}: {e}"))?;
            lightcraft_segment::sync(&model).map_err(|e| e.to_string())?;
            if !output.presence.is_finite() || output.logits.iter().any(|x| !x.is_finite()) || output.boxes.iter().flatten().any(|x| !x.is_finite()) {
                return Err(format!("{text}: detector returned non-finite scores or boxes"));
            }
            let presence = sigmoid(output.presence);
            let scored: Vec<(usize, f32)> = output.logits.iter().enumerate().map(|(i, l)| (i, sigmoid(*l) * presence)).collect();
            let top = scored.iter().copied().max_by(|a, b| a.1.total_cmp(&b.1));
            let above_threshold_count = scored.iter().filter(|(_, score)| *score > 0.5).count();
            let mut merged: Option<Vec<f32>> = None;
            for (i, _) in scored.iter().filter(|(_, score)| *score > 0.5) {
                let mask = output.masks.get(*i).map_err(|e| e.to_string())?.to_vec2::<f32>().map_err(|e| e.to_string())?;
                if mask.len() != MASK_SIDE || mask.iter().any(|row| row.len() != MASK_SIDE || row.iter().any(|v| !v.is_finite())) {
                    return Err(format!("{text}: detector returned an invalid mask"));
                }
                let mask: Vec<f32> = mask.into_iter().flatten().collect();
                if let Some(previous) = &mut merged {
                    for (out, value) in previous.iter_mut().zip(mask) {
                        *out = out.max(value);
                    }
                } else {
                    merged = Some(mask);
                }
            }
            let positive_cells = merged.as_ref().map_or(0, |mask| mask.iter().filter(|v| **v > 0.0).count());
            rows.push(Row {
                text,
                presence,
                max_confidence: top.map_or(0.0, |(_, score)| score),
                top_bbox: top.and_then(|(i, _)| output.boxes.get(i).copied()),
                above_threshold_count,
                positive_cells,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                mask: merged,
                mask_path: None,
            });
        }
        if case.save_masks {
            let mut best: Vec<usize> = (0..rows.len()).filter(|i| rows[*i].positive_cells > 0).collect();
            best.sort_by(|a, b| rows[*b].max_confidence.total_cmp(&rows[*a].max_confidence));
            for (rank, index) in best.into_iter().take(3).enumerate() {
                let row = &mut rows[index];
                let logits = row.mask.as_ref().ok_or("nonempty row lost its mask")?;
                let image = Rgba8::from_fn(MASK_SIDE, MASK_SIDE, |x, y| {
                    let value = if logits[y * MASK_SIDE + x] > 0.0 { 255 } else { 0 };
                    [value, value, value, 255]
                });
                let png = lightcraft_codecs::encode_png(&EncodeImage::rgba8(&image), &EncodeMeta::default()).map_err(|e| e.to_string())?;
                let path = output_dir.join(format!("sam3-prompt-{rank}.png"));
                write_new(&path, &png)?;
                row.mask_path = Some(path);
            }
        }
        let report_rows: Vec<_> = rows
            .iter()
            .map(|r| {
                json!({"text":r.text,"presenceProbability":r.presence,"maxConfidence":r.max_confidence,
                    "topBBox":r.top_bbox,"aboveThresholdCount":r.above_threshold_count,
                    "positiveCells":r.positive_cells,"nonempty":r.positive_cells>0,
                    "seconds":r.elapsed_seconds,"maskPng":r.mask_path})
            })
            .collect();
        let report = json!({"source":source,"modelDir":model_dir,"device":device_name,
            "inputWidth":image.width,"inputHeight":image.height,"maskSide":MASK_SIDE,
            "threshold":0.5,"loadSeconds":load_seconds,"encodeSeconds":encode_seconds,
            "queries":report_rows});
        let path = output_dir.join("sam3-prompt-probe.json");
        write_new(&path, &serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)?;
        println!("{}", path.display());
        Ok(())
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn read_limited(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if file.metadata().map_err(|e| e.to_string())?.len() > limit {
            return Err("case file exceeds 1 MiB".into());
        }
        let mut bytes = Vec::new();
        file.take(limit + 1).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        if bytes.len() as u64 > limit {
            return Err("case file exceeds 1 MiB".into());
        }
        Ok(bytes)
    }

    fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        file.write_all(bytes).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())
    }
}
