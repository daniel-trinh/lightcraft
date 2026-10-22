//! Developer-only Klein end-to-end image probe.
//!
//! Usage: klein_inpaint_image MODEL_DIR FIXTURE_DIR OUTPUT_DIR
//! FIXTURE_DIR contains preprocessed input.png, mask.png and run.json. Outputs
//! generated.png (the raw model canvas) and composite.png (the fixture mask
//! applied to input) for visual review. No weights or images are built in.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use image::{GrayImage, RgbImage};
use lightcraft_inpaint::klein::{KleinConditioning, KleinOptions, generate};

fn image_from_rgb(rgb: Vec<u8>, width: u32, height: u32) -> Result<RgbImage, Box<dyn std::error::Error>> {
    RgbImage::from_raw(width, height, rgb).ok_or_else(|| "invalid generated RGB dimensions".into())
}

fn load_fixture(path: &Path) -> Result<(RgbImage, GrayImage, KleinOptions, PathBuf, PathBuf), Box<dyn std::error::Error>> {
    let input = image::open(path.join("input.png"))?.into_rgb8();
    let mask = image::open(path.join("mask.png"))?.into_luma8();
    if input.dimensions() != mask.dimensions() {
        return Err("fixture image and mask dimensions differ".into());
    }
    let run: serde_json::Value = serde_json::from_slice(&std::fs::read(path.join("run.json"))?)?;
    let prompt = run.get("prompt").and_then(serde_json::Value::as_str).ok_or("fixture prompt missing")?;
    let seed = run.get("seed").and_then(serde_json::Value::as_u64).ok_or("fixture seed missing")?;
    let steps = run.get("steps").and_then(serde_json::Value::as_u64).ok_or("fixture steps missing")?;
    let strength = run.get("strength").and_then(serde_json::Value::as_f64).ok_or("fixture strength missing")?;
    let source = run.get("source").and_then(serde_json::Value::as_str).ok_or("fixture original source missing")?;
    let original_mask = run.get("mask").and_then(serde_json::Value::as_str).ok_or("fixture original mask missing")?;
    let conditioning = match run.get("conditioning").and_then(serde_json::Value::as_str) {
        None | Some("fullImage") => KleinConditioning::FullImage,
        Some("erasedMask") => KleinConditioning::ErasedMask,
        Some(_) => return Err("unknown Klein conditioning mode".into()),
    };
    Ok((
        input,
        mask,
        KleinOptions { prompt: prompt.into(), seed, steps: usize::try_from(steps)?, strength, conditioning },
        source.into(),
        original_mask.into(),
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [model_dir, fixture_dir, output_dir] = args.as_slice() else {
        return Err("usage: klein_inpaint_image MODEL_DIR FIXTURE_DIR OUTPUT_DIR".into());
    };
    let (model_dir, fixture_dir, output_dir) = (PathBuf::from(model_dir), PathBuf::from(fixture_dir), PathBuf::from(output_dir));
    let (input, mask, options, original_path, original_mask_path) = load_fixture(&fixture_dir)?;
    let (width, height) = input.dimensions();
    let cancel = AtomicBool::new(false);
    let started = Instant::now();
    let generated = generate(&model_dir, input.as_raw(), mask.as_raw(), width as usize, height as usize, &options, &cancel, &mut |event| {
        eprintln!("{:.1}s {:?}", started.elapsed().as_secs_f32(), event.phase)
    })?;
    let output = image_from_rgb(generated, width, height)?;
    std::fs::create_dir_all(&output_dir)?;
    output.save(output_dir.join("generated.png"))?;
    let mut composite = input;
    for ((original, generated), selected) in composite.pixels_mut().zip(output.pixels()).zip(mask.pixels()) {
        if selected.0[0] >= 128 {
            *original = *generated;
        }
    }
    composite.save(output_dir.join("composite.png"))?;
    let original = image::open(original_path)?.into_rgb8();
    let original_mask = image::open(original_mask_path)?.into_luma8();
    if original.dimensions() != original_mask.dimensions() {
        return Err("original image and original mask dimensions differ".into());
    }
    let resized = image::imageops::resize(&output, original.width(), original.height(), image::imageops::FilterType::Lanczos3);
    let mut original_composite = original.clone();
    let mut unchanged = true;
    for (((target, source), generated), selected) in
        original_composite.pixels_mut().zip(original.pixels()).zip(resized.pixels()).zip(original_mask.pixels())
    {
        let alpha = u16::from(selected.0[0]);
        for channel in 0..3 {
            target.0[channel] = ((u16::from(source.0[channel]) * (255 - alpha) + u16::from(generated.0[channel]) * alpha + 127) / 255) as u8;
        }
        if alpha == 0 && target != source {
            unchanged = false;
        }
    }
    if !unchanged {
        return Err("unselected original pixels changed".into());
    }
    original_composite.save(output_dir.join("composite-original.png"))?;
    println!("Finished in {:.2}s. Outputs: {}", started.elapsed().as_secs_f32(), output_dir.display());
    Ok(())
}
