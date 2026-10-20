//! Native local inpainting, with an optional advanced ComfyUI backend.
//! The model output is blended into a new PNG beside the source; the source is never changed.

use std::io::{Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use lightcraft_catalog::{PhotoId, Source};
use lightcraft_codecs::{EncodeImage, EncodeMeta};
use lightcraft_pipeline::{MaskView, Overlay};
use lightcraft_raster::Rgba8;
use serde_json::{Value, json};

use crate::{
    Session,
    media::{RenderJob, SourceLevel, SourceRef},
};

const MAX_RESPONSE: usize = 64 * 1024 * 1024;
const MAX_EDGE: usize = 2048;
const NATIVE_MAX_EDGE: usize = 512;
const MAX_NATIVE_PIXELS: usize = 64_000_000;
pub const REMOVE_MASK_MARGIN: usize = 32;
static NEXT_NAME: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
pub struct Options {
    pub backend: Backend,
    pub endpoint: String,
    pub checkpoint: String,
    pub prompt: String,
    pub negative: String,
    pub steps: u32,
    pub guidance: f64,
    pub denoise: f64,
    pub seed: u64,
    pub edge: usize,
    /// Extra conditioning mask pixels on the model canvas. The final blend uses the original mask.
    pub mask_margin: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Native,
    Comfy,
}

impl Options {
    pub fn validate(&self) -> Result<(), String> {
        if self.mask_margin > 64 {
            return Err("maskMargin must be 0..64 model pixels".into());
        }
        if self.prompt.len() > 4096 || self.negative.len() > 4096 {
            return Err("prompt is too long (maximum 4096 bytes)".into());
        }
        if !(1..=100).contains(&self.steps) {
            return Err("steps must be 1..100".into());
        }
        if !self.guidance.is_finite() || !(1.0..=20.0).contains(&self.guidance) {
            return Err("guidance must be 1..20".into());
        }
        match self.backend {
            Backend::Native => {
                if !self.denoise.is_finite() || !(0.0..=1.0).contains(&self.denoise) || self.denoise == 0.0 {
                    return Err("native denoise must be greater than 0 and no more than 1".into());
                }
                if (self.denoise * f64::from(self.steps)).floor() < 1.0 {
                    return Err("native denoise and steps must leave at least one diffusion step".into());
                }
                if !(64..=NATIVE_MAX_EDGE).contains(&self.edge) {
                    return Err("native inpainting supports image edges from 64 to 512 pixels".into());
                }
            }
            Backend::Comfy => {
                if self.mask_margin != 0 {
                    return Err("maskMargin is only available with backend=native".into());
                }
                endpoint(&self.endpoint)?;
                if self.checkpoint.trim().is_empty() || self.checkpoint.chars().any(char::is_control) {
                    return Err("choose an installed ComfyUI checkpoint filename".into());
                }
                if !self.denoise.is_finite() || !(0.01..=1.0).contains(&self.denoise) {
                    return Err("denoise must be 0.01..1".into());
                }
                if !(64..=MAX_EDGE).contains(&self.edge) {
                    return Err("ComfyUI image edge must be 64..2048 pixels".into());
                }
            }
        }
        Ok(())
    }
}

/// Prompts used for contextual removal. The negative prompt supplied by the user is appended.
pub fn removal_prompts(negative: &str) -> (String, String) {
    let prompt = "empty natural background, matching the surrounding texture, lighting and perspective".to_string();
    let avoid = "subject, object, person, text, artifact";
    let negative = if negative.trim().is_empty() { avoid.to_string() } else { format!("{avoid}, {negative}") };
    (prompt, negative)
}

/// Detached work suitable for a UI background task.
pub struct Job {
    pub original: PhotoId,
    pub destination: PathBuf,
    options: Options,
    image: RenderJob,
    mask: RenderJob,
    model_dir: Option<PathBuf>,
}

impl Session {
    pub fn plan_generative(&mut self, id: PhotoId, mask_id: u32, options: Options, demo_dir: Option<&str>) -> Result<Job, String> {
        options.validate()?;
        let photo = self.catalog.photo(id).ok_or("no such photo")?.clone();
        if !photo.develop.masks.iter().any(|m| m.id == mask_id && !m.components.is_empty()) {
            return Err("select a nonempty mask first".into());
        }
        let dir = match &photo.source {
            Source::File { path } => Path::new(path).parent().ok_or("source has no parent folder")?.to_path_buf(),
            Source::Demo { .. } => PathBuf::from(demo_dir.ok_or("demo photos require a destination folder")?),
        };
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let stem = Path::new(&photo.file_name).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "photo".into());
        let model_dir = match options.backend {
            Backend::Native => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    Some(self.generator.model_dir()?)
                }
                #[cfg(target_arch = "wasm32")]
                {
                    return Err("native inpainting is only available in the desktop app".into());
                }
            }
            Backend::Comfy => None,
        };
        let render_edge = if options.backend == Backend::Native {
            if u64::from(photo.width).saturating_mul(u64::from(photo.height)) > MAX_NATIVE_PIXELS as u64 {
                return Err("the source exceeds the 64 MP safety limit; export a smaller copy first".into());
            }
            let (w, h) = lightcraft_pipeline::native_output_size(photo.width.max(1) as usize, photo.height.max(1) as usize, &photo.develop);
            if !w.is_finite() || !h.is_finite() || w < 1.0 || h < 1.0 || w * h > MAX_NATIVE_PIXELS as f64 {
                return Err("the full-resolution edit exceeds the 64 MP safety limit; export a smaller copy first".into());
            }
            let (render_w, render_h) = (w.round() as usize, h.round() as usize);
            if render_w.saturating_mul(render_h) > MAX_NATIVE_PIXELS {
                return Err("the full-resolution edit exceeds the 64 MP safety limit; export a smaller copy first".into());
            }
            render_w.max(render_h).max(1)
        } else {
            options.edge
        };
        let mut image = self.render_job(id, render_edge, render_edge, false, true).ok_or("could not render photo")?;
        let mid = u16::try_from(mask_id).map_err(|_| "mask id is too large for rendering")?;
        let mut mask = self.render_job(id, render_edge, render_edge, false, true).ok_or("could not render mask")?.with_overlay(Overlay::Mask {
            id: mid,
            view: MaskView::WhiteOnBlack,
            color: [255; 3],
            opacity: 100,
        });
        // A tight crop may have a short output edge while still needing native source pixels.
        if options.backend == Backend::Native {
            image.level = SourceLevel::Full;
            image.source = self.media.source_ref(&photo, SourceLevel::Full);
            mask.level = SourceLevel::Full;
            mask.source = image.source.clone();
        }
        let destination = unique_destination(&dir, &stem);
        Ok(Job { original: id, destination, options, image, mask, model_dir })
    }
}

fn unique_destination(dir: &Path, stem: &str) -> PathBuf {
    let first = dir.join(format!("{stem}-AI-Edit.png"));
    if !first.exists() {
        return first;
    }
    for n in 2..u32::MAX {
        let p = dir.join(format!("{stem}-AI-Edit-{n}.png"));
        if !p.exists() {
            return p;
        }
    }
    dir.join(format!("{stem}-AI-Edit-{}.png", std::process::id()))
}

impl Job {
    /// Run the local model and durably write a generated file. It becomes a catalog photo only
    /// after `photo.generativeImport`, so failures cannot replace the original or an earlier edit.
    pub fn run(self) -> Result<PathBuf, String> {
        self.run_with(&std::sync::atomic::AtomicBool::new(false), |_, _| {})
    }

    /// Run generation with cooperative cancellation and model progress reporting.
    pub fn run_with(self, cancel: &std::sync::atomic::AtomicBool, mut progress: impl FnMut(usize, usize)) -> Result<PathBuf, String> {
        if cancel.load(Ordering::Relaxed) {
            return Err("generation cancelled".into());
        }
        let rendered_image = self.image.run();
        let mut mask_job = self.mask;
        if let Some(source) = rendered_image.loaded {
            mask_job.source = SourceRef::Loaded(Box::new(source));
        }
        let mut image = rendered_image.rendered?.image;
        let mask = mask_job.run().rendered?.image;
        if (image.width, image.height) != (mask.width, mask.height) || image.width == 0 || image.height == 0 {
            return Err("mask and photo sizes do not match".into());
        }
        if self.options.backend == Backend::Native && image.width.saturating_mul(image.height) > MAX_NATIVE_PIXELS {
            return Err("the rendered edit exceeds the 64 MP safety limit; export a smaller copy first".into());
        }
        if !mask.data.iter().any(|px| px[0] > 8) {
            return Err("the selected mask has no visible area".into());
        }
        progress(0, 1);
        match self.options.backend {
            Backend::Native => {
                #[cfg(all(feature = "inpaint", not(target_arch = "wasm32")))]
                {
                    let model_dir = self.model_dir.as_deref().ok_or("native model directory is unavailable")?;
                    if cancel.load(Ordering::Relaxed) {
                        return Err("generation cancelled".into());
                    }
                    let mut model = lightcraft_inpaint::Model::load(model_dir).map_err(|e| format!("could not load local inpainting model: {e}"))?;
                    if cancel.load(Ordering::Relaxed) {
                        return Err("generation cancelled".into());
                    }
                    let roi = selected_roi(&mask, self.options.edge).ok_or("the selected mask has no visible area")?;
                    let patch = roi_patch(&image, roi);
                    let mut rgb = Vec::with_capacity(patch.width * patch.height * 3);
                    for px in &patch.data {
                        rgb.extend_from_slice(px.get(..3).ok_or("invalid rendered pixel")?);
                    }
                    let mut mask8 = roi_mask8(&mask, roi);
                    dilate_mask8(&mut mask8, patch.width, patch.height, self.options.mask_margin)?;
                    eprintln!(
                        "Lightcraft inpaint: device={}, full={}x{}, ROI=({}, {}) {}x{}, model={}x{}, maskMargin={}",
                        model.device_description(),
                        image.width,
                        image.height,
                        roi.x,
                        roi.y,
                        roi.w,
                        roi.h,
                        patch.width,
                        patch.height,
                        self.options.mask_margin
                    );
                    let native_options = lightcraft_inpaint::GenerateOptions {
                        prompt: if self.options.prompt.trim().is_empty() {
                            "clean natural background matching the surrounding image".into()
                        } else {
                            self.options.prompt.clone()
                        },
                        negative: self.options.negative.clone(),
                        seed: self.options.seed,
                        steps: self.options.steps as usize,
                        guidance: self.options.guidance,
                        strength: self.options.denoise,
                    };
                    let pixels = model
                        .generate(&rgb, &mask8, patch.width, patch.height, &native_options, cancel, &mut progress)
                        .map_err(|e| format!("native inpainting failed: {e}"))?;
                    if pixels.len() != patch.width.saturating_mul(patch.height).saturating_mul(3) {
                        return Err("native inpainting returned an invalid image size".into());
                    }
                    let mut generated = Rgba8::new(patch.width, patch.height);
                    for (dst, src) in generated.data.iter_mut().zip(pixels.as_chunks::<3>().0) {
                        let rgb = src.get(..3).ok_or("invalid generated pixel")?;
                        dst.get_mut(..3).ok_or("invalid output pixel")?.copy_from_slice(rgb);
                        if let Some(alpha) = dst.get_mut(3) {
                            *alpha = 255;
                        }
                    }
                    composite_roi(&mut image, &generated, &mask, roi);
                }
                #[cfg(not(all(feature = "inpaint", not(target_arch = "wasm32"))))]
                {
                    let _ = self.model_dir.as_deref();
                    return Err("native inpainting is not enabled in this build".into());
                }
            }
            Backend::Comfy => {
                let (w, h) = (image.width.next_multiple_of(8).max(8), image.height.next_multiple_of(8).max(8));
                let padded_image = pad_edges(&image, w, h);
                let mut mask_png = Rgba8::new(mask.width, mask.height);
                for (src, dst) in mask.data.iter().zip(&mut mask_png.data) {
                    // ComfyUI's LoadImageMask inverts the PNG alpha channel.
                    *dst = [0, 0, 0, 255 - src[0]];
                }
                if cancel.load(Ordering::Relaxed) {
                    return Err("generation cancelled".into());
                }
                let input = png(&padded_image)?;
                let alpha = png(&pad_edges(&mask_png, w, h))?;
                let addr = endpoint(&self.options.endpoint)?;
                let nonce = NEXT_NAME.fetch_add(1, Ordering::Relaxed);
                let base = format!(
                    "lightcraft-{}-{}-{nonce}",
                    std::process::id(),
                    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis()
                );
                let image_name = upload(addr, &format!("{base}-photo.png"), &input)?;
                let mask_name = upload(addr, &format!("{base}-mask.png"), &alpha)?;
                let prompt = workflow(&self.options, &image_name, &mask_name, &base);
                let response = http(
                    addr,
                    "POST",
                    "/prompt",
                    &serde_json::to_vec(&json!({"prompt": prompt, "client_id": "lightcraft"})).map_err(|e| e.to_string())?,
                    "application/json",
                    1024 * 1024,
                )?;
                let ticket: Value = serde_json::from_slice(&response).map_err(|e| format!("ComfyUI prompt response: {e}"))?;
                let id = ticket.get("prompt_id").and_then(Value::as_str).ok_or_else(|| format!("ComfyUI rejected the workflow: {ticket}"))?;
                if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
                    return Err("invalid ComfyUI prompt id".into());
                }
                let output = poll_result(addr, id, cancel)?;
                let generated = lightcraft_codecs::decode(
                    &output,
                    lightcraft_codecs::DecodeOptions { max_pixels: (MAX_EDGE * MAX_EDGE) as u64, ..Default::default() },
                )
                .map_err(|e| format!("ComfyUI returned an invalid image: {e}"))?
                .to_srgb8();
                if (generated.width, generated.height) != (w, h) {
                    return Err(format!("ComfyUI output is {}×{}, expected {w}×{h}", generated.width, generated.height));
                }
                let generated = Rgba8::from_fn(image.width, image.height, |x, y| generated.get(x, y));
                composite(&mut image, &generated, &mask);
            }
        }
        if self.options.backend == Backend::Comfy {
            progress(1, 1);
        }
        if cancel.load(Ordering::Relaxed) {
            return Err("generation cancelled".into());
        }
        let output = png(&image)?;
        let path = lightcraft_catalog::safe_file::write_new_unique(
            &mut (0u32..).map(|n| {
                if n == 0 {
                    self.destination.clone()
                } else {
                    let stem = self.destination.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "AI-Edit".into());
                    self.destination.with_file_name(format!("{stem}-{n}.png"))
                }
            }),
            &output,
        )
        .map_err(|e| format!("could not save generated photo: {e}"))?;
        Ok(path)
    }
}

fn pad_edges(src: &Rgba8, w: usize, h: usize) -> Rgba8 {
    Rgba8::from_fn(w, h, |x, y| src.get(x.min(src.width - 1), y.min(src.height - 1)))
}

/// A selected bounding box with native-resolution context. Only the model canvas is resized.
#[cfg(any(all(feature = "inpaint", not(target_arch = "wasm32")), test))]
#[derive(Clone, Copy, Debug)]
struct Roi {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    active_w: usize,
    active_h: usize,
    model_w: usize,
    model_h: usize,
}

#[cfg(any(all(feature = "inpaint", not(target_arch = "wasm32")), test))]
fn selected_roi(mask: &Rgba8, edge: usize) -> Option<Roi> {
    if mask.width == 0 || mask.height == 0 || mask.width.checked_mul(mask.height) != Some(mask.data.len()) || !(64..=512).contains(&edge) {
        return None;
    }
    let (mut x0, mut y0, mut x1, mut y1) = (mask.width, mask.height, 0, 0);
    for (i, px) in mask.data.iter().enumerate() {
        if px[0] > 0 {
            let (x, y) = (i % mask.width, i / mask.width);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    if x0 > x1 || y0 > y1 {
        return None;
    }
    // Give the model surrounding scene context rather than enlarging the hole
    // to most of its canvas. Small object removal otherwise becomes a close-up
    // of the selected object and can invent another object in its place.
    let side = (x1 - x0 + 1).max(y1 - y0 + 1).saturating_mul(3).max(64);
    let (w, h) = (side.min(mask.width), side.min(mask.height));
    let x = ((x0 + x1 + 1).saturating_sub(w) / 2).min(mask.width - w);
    let y = ((y0 + y1 + 1).saturating_sub(h) / 2).min(mask.height - h);
    let scale = edge as f64 / w.max(h) as f64;
    let active_w = ((w as f64 * scale).round() as usize).clamp(1, edge);
    let active_h = ((h as f64 * scale).round() as usize).clamp(1, edge);
    Some(Roi { x, y, w, h, active_w, active_h, model_w: active_w.next_multiple_of(8).max(64), model_h: active_h.next_multiple_of(8).max(64) })
}

#[cfg(any(all(feature = "inpaint", not(target_arch = "wasm32")), test))]
fn sample_bilinear(src: &Rgba8, x: f64, y: f64) -> [u8; 4] {
    let x = x.clamp(0.0, (src.width - 1) as f64);
    let y = y.clamp(0.0, (src.height - 1) as f64);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(src.width - 1), (y0 + 1).min(src.height - 1));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let (a, b, c, d) = (src.get(x0, y0), src.get(x1, y0), src.get(x0, y1), src.get(x1, y1));
    std::array::from_fn(|i| {
        ((a[i] as f64 * (1.0 - fx) + b[i] as f64 * fx) * (1.0 - fy) + (c[i] as f64 * (1.0 - fx) + d[i] as f64 * fx) * fy).round() as u8
    })
}

#[cfg(any(all(feature = "inpaint", not(target_arch = "wasm32")), test))]
fn roi_patch(image: &Rgba8, roi: Roi) -> Rgba8 {
    Rgba8::from_fn(roi.model_w, roi.model_h, |x, y| {
        let x = x.min(roi.active_w - 1);
        let y = y.min(roi.active_h - 1);
        sample_bilinear(
            image,
            roi.x as f64 + (x as f64 + 0.5) * roi.w as f64 / roi.active_w as f64 - 0.5,
            roi.y as f64 + (y as f64 + 0.5) * roi.h as f64 / roi.active_h as f64 - 0.5,
        )
    })
}

#[cfg(any(all(feature = "inpaint", not(target_arch = "wasm32")), test))]
fn roi_mask8(mask: &Rgba8, roi: Roi) -> Vec<u8> {
    let mut out = vec![0; roi.model_w * roi.model_h];
    // Sample at model resolution for contiguous masks when the native patch is enlarged.
    for my in 0..roi.active_h {
        for mx in 0..roi.active_w {
            let x = roi.x + ((mx as f64 + 0.5) * roi.w as f64 / roi.active_w as f64).floor() as usize;
            let y = roi.y + ((my as f64 + 0.5) * roi.h as f64 / roi.active_h as f64).floor() as usize;
            if mask.get(x.min(mask.width - 1), y.min(mask.height - 1))[0] > 8
                && let Some(px) = out.get_mut(my * roi.model_w + mx)
            {
                *px = 255;
            }
        }
    }
    // Project every selected native pixel, so thin selections cannot disappear when reduced.
    for y in 0..roi.h {
        for x in 0..roi.w {
            if mask.get(roi.x + x, roi.y + y)[0] > 8 {
                let mx = x * roi.active_w / roi.w;
                let my = y * roi.active_h / roi.h;
                if let Some(px) = out.get_mut(my * roi.model_w + mx) {
                    *px = 255;
                }
            }
        }
    }
    // Inference dimensions are multiples of eight; extend the last active edge into padding.
    for y in 0..roi.model_h {
        for x in 0..roi.model_w {
            if x >= roi.active_w || y >= roi.active_h {
                let src = out.get(y.min(roi.active_h - 1) * roi.model_w + x.min(roi.active_w - 1)).copied().unwrap_or(0);
                if let Some(dst) = out.get_mut(y * roi.model_w + x) {
                    *dst = src;
                }
            }
        }
    }
    out
}

/// Square binary dilation on the bounded model canvas, in two linear passes. This mask is
/// conditioning only: `composite_roi` always receives the untouched full-resolution mask.
#[cfg(any(all(feature = "inpaint", not(target_arch = "wasm32")), test))]
fn dilate_mask8(mask: &mut [u8], width: usize, height: usize, margin: usize) -> Result<(), String> {
    if width == 0
        || height == 0
        || width > NATIVE_MAX_EDGE
        || height > NATIVE_MAX_EDGE
        || width.checked_mul(height) != Some(mask.len())
        || margin > 64
    {
        return Err("invalid model mask dimensions or margin".into());
    }
    if margin == 0 {
        return Ok(());
    }
    let mut horizontal = vec![0u8; mask.len()];
    let mut sums = Vec::with_capacity(width.max(height) + 1);
    for y in 0..height {
        sums.clear();
        sums.push(0u16);
        for x in 0..width {
            let selected = mask.get(y * width + x).is_some_and(|&px| px != 0);
            sums.push(sums.last().copied().unwrap_or(0) + u16::from(selected));
        }
        for x in 0..width {
            let left = x.saturating_sub(margin);
            let right = x.saturating_add(margin).saturating_add(1).min(width);
            let count = sums.get(right).copied().ok_or("invalid model mask row")? - sums.get(left).copied().ok_or("invalid model mask row")?;
            *horizontal.get_mut(y * width + x).ok_or("invalid model mask row")? = if count > 0 { 255 } else { 0 };
        }
    }
    for x in 0..width {
        sums.clear();
        sums.push(0u16);
        for y in 0..height {
            let selected = horizontal.get(y * width + x).is_some_and(|&px| px != 0);
            sums.push(sums.last().copied().unwrap_or(0) + u16::from(selected));
        }
        for y in 0..height {
            let top = y.saturating_sub(margin);
            let bottom = y.saturating_add(margin).saturating_add(1).min(height);
            let count = sums.get(bottom).copied().ok_or("invalid model mask column")? - sums.get(top).copied().ok_or("invalid model mask column")?;
            *mask.get_mut(y * width + x).ok_or("invalid model mask column")? = if count > 0 { 255 } else { 0 };
        }
    }
    Ok(())
}

#[cfg(any(all(feature = "inpaint", not(target_arch = "wasm32")), test))]
fn composite_roi(image: &mut Rgba8, generated: &Rgba8, mask: &Rgba8, roi: Roi) {
    for y in 0..roi.h {
        for x in 0..roi.w {
            let index = (roi.y + y) * image.width + roi.x + x;
            let alpha = mask.data.get(index).map_or(0, |px| px[0]) as u16;
            if alpha == 0 {
                continue;
            }
            let gx = (x as f64 + 0.5) * roi.active_w as f64 / roi.w as f64 - 0.5;
            let gy = (y as f64 + 0.5) * roi.active_h as f64 / roi.h as f64 - 0.5;
            let px = sample_bilinear(generated, gx, gy);
            if let Some(dst) = image.data.get_mut(index) {
                for c in 0..3 {
                    dst[c] = ((u16::from(dst[c]) * (255 - alpha) + u16::from(px[c]) * alpha + 127) / 255) as u8;
                }
            }
        }
    }
}

fn composite(image: &mut Rgba8, generated: &Rgba8, mask: &Rgba8) {
    for ((dst, generated_px), m) in image.data.iter_mut().zip(&generated.data).zip(&mask.data) {
        let a = m[0] as u16;
        for c in 0..3 {
            dst[c] = ((u16::from(dst[c]) * (255 - a) + u16::from(generated_px[c]) * a + 127) / 255) as u8;
        }
    }
}

fn png(img: &Rgba8) -> Result<Vec<u8>, String> {
    lightcraft_codecs::encode_png(&EncodeImage::rgba8(img), &EncodeMeta::default()).map_err(|e| e.to_string())
}

fn endpoint(value: &str) -> Result<SocketAddr, String> {
    let authority = value.strip_prefix("http://").ok_or("ComfyUI URL must start with http://")?;
    if authority.contains(['/', '?', '#', '@']) {
        return Err("ComfyUI URL must be an origin without a path or credentials".into());
    }
    let addr: SocketAddr = authority.parse().map_err(|_| "use a loopback IP and port, such as http://127.0.0.1:8188")?;
    if !matches!(addr.ip(), IpAddr::V4(a) if a.is_loopback()) && !matches!(addr.ip(), IpAddr::V6(a) if a.is_loopback()) {
        return Err("ComfyUI must be on this computer (loopback address)".into());
    }
    Ok(addr)
}

fn workflow(o: &Options, image: &str, mask: &str, prefix: &str) -> Value {
    let prompt = if o.prompt.trim().is_empty() { "clean natural background matching the surrounding image" } else { o.prompt.as_str() };
    json!({
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "LoadImageMask", "inputs": {"image": mask, "channel": "alpha"}},
        "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": o.checkpoint}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 1], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 1], "text": o.negative}},
        "6": {"class_type": "InpaintModelConditioning", "inputs": {"positive": ["4", 0], "negative": ["5", 0], "pixels": ["1", 0], "mask": ["2", 0], "vae": ["3", 2], "noise_mask": true}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["3", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["6", 2], "seed": o.seed, "steps": o.steps, "cfg": o.guidance, "sampler_name": "euler", "scheduler": "normal", "denoise": o.denoise}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    })
}

fn upload(addr: SocketAddr, name: &str, bytes: &[u8]) -> Result<String, String> {
    let boundary = "LightcraftInpaintBoundary";
    let mut body = format!("--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\nContent-Type: image/png\r\n\r\n")
        .into_bytes();
    body.extend_from_slice(bytes);
    body.extend_from_slice(
        format!("\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n--{boundary}--\r\n").as_bytes(),
    );
    let response = http(addr, "POST", "/upload/image", &body, &format!("multipart/form-data; boundary={boundary}"), 1024 * 1024)?;
    let info: Value = serde_json::from_slice(&response).map_err(|e| format!("ComfyUI upload response: {e}"))?;
    let returned = info.get("name").and_then(Value::as_str).ok_or_else(|| format!("ComfyUI rejected image upload: {info}"))?;
    if returned != name {
        return Err("ComfyUI renamed the uploaded image unexpectedly".into());
    }
    Ok(returned.to_string())
}

fn poll_result(addr: SocketAddr, id: &str, cancel: &std::sync::atomic::AtomicBool) -> Result<Vec<u8>, String> {
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(10 * 60) {
        if cancel.load(Ordering::Relaxed) {
            return Err("generation cancelled".into());
        }
        let bytes = http(addr, "GET", &format!("/history/{id}"), &[], "", 4 * 1024 * 1024)?;
        let history: Value = serde_json::from_slice(&bytes).map_err(|e| format!("ComfyUI history response: {e}"))?;
        if let Some(record) = history.get(id) {
            if record.pointer("/status/status_str").and_then(Value::as_str) == Some("error") {
                return Err(format!("ComfyUI failed: {}", record.get("status").unwrap_or(&Value::Null)));
            }
            if let Some(file) = record.pointer("/outputs/9/images/0") {
                let filename = file.get("filename").and_then(Value::as_str).ok_or("ComfyUI output has no filename")?;
                let subfolder = file.get("subfolder").and_then(Value::as_str).unwrap_or("");
                let kind = file.get("type").and_then(Value::as_str).unwrap_or("output");
                let path = format!("/view?filename={}&subfolder={}&type={}", url_encode(filename), url_encode(subfolder), url_encode(kind));
                return http(addr, "GET", &path, &[], "", MAX_RESPONSE);
            }
            if record.pointer("/status/completed").and_then(Value::as_bool) == Some(true) {
                return Err("ComfyUI finished without an image from SaveImage".into());
            }
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    Err("ComfyUI did not finish within 10 minutes".into())
}

fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.') {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn http(addr: SocketAddr, method: &str, path: &str, body: &[u8], content_type: &str, limit: usize) -> Result<Vec<u8>, String> {
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(5)).map_err(|e| format!("could not connect to local ComfyUI: {e}"))?;
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Length: {}\r\nContent-Type: {content_type}\r\n\r\n",
        body.len()
    );
    write_deadline(&mut stream, head.as_bytes(), deadline)?;
    write_deadline(&mut stream, body, deadline)?;
    let mut response = Vec::new();
    let mut buffer = [0u8; 16384];
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("ComfyUI response exceeded 60 seconds".into());
        }
        stream.set_read_timeout(Some(remaining.min(Duration::from_secs(30)))).map_err(|e| e.to_string())?;
        let n = stream.read(&mut buffer).map_err(|e| format!("ComfyUI response failed: {e}"))?;
        if n == 0 {
            break;
        }
        if response.len().saturating_add(n) > limit.saturating_add(65536) {
            return Err("ComfyUI response is too large".into());
        }
        response.extend_from_slice(&buffer[..n]);
    }
    parse_response(&response, limit)
}

fn write_deadline(stream: &mut TcpStream, data: &[u8], deadline: Instant) -> Result<(), String> {
    let mut offset = 0;
    while offset < data.len() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("ComfyUI request exceeded 60 seconds".into());
        }
        stream.set_write_timeout(Some(remaining.min(Duration::from_secs(30)))).map_err(|e| e.to_string())?;
        let n = stream.write(&data[offset..]).map_err(|e| format!("ComfyUI request failed: {e}"))?;
        if n == 0 {
            return Err("ComfyUI request stopped accepting data".into());
        }
        offset += n;
    }
    Ok(())
}

fn parse_response(response: &[u8], limit: usize) -> Result<Vec<u8>, String> {
    let split = response.windows(4).position(|w| w == b"\r\n\r\n").ok_or("ComfyUI sent an incomplete HTTP response")?;
    let header = std::str::from_utf8(response.get(..split).ok_or("bad HTTP header")?).map_err(|_| "bad HTTP header")?;
    let status = header.lines().next().and_then(|s| s.split_whitespace().nth(1)).and_then(|s| s.parse::<u16>().ok()).ok_or("bad HTTP status")?;
    let body = response.get(split + 4..).ok_or("bad HTTP body")?;
    if !(200..300).contains(&status) {
        return Err(format!("ComfyUI HTTP {status}: {}", String::from_utf8_lossy(body).chars().take(600).collect::<String>()));
    }
    if header.lines().any(|h| h.eq_ignore_ascii_case("transfer-encoding: chunked")) {
        return decode_chunked(body, limit);
    }
    if body.len() > limit {
        return Err("ComfyUI response is too large".into());
    }
    if let Some(n) = header.lines().find_map(|h| h.to_ascii_lowercase().strip_prefix("content-length:").and_then(|v| v.trim().parse::<usize>().ok()))
        && n != body.len()
    {
        return Err("ComfyUI response length does not match its header".into());
    }
    Ok(body.to_vec())
}

fn decode_chunked(body: &[u8], limit: usize) -> Result<Vec<u8>, String> {
    let mut pos = 0usize;
    let mut out = Vec::new();
    loop {
        let line = body.get(pos..).ok_or("invalid chunk")?;
        let end = line.windows(2).position(|w| w == b"\r\n").ok_or("invalid chunk size")?;
        let size = std::str::from_utf8(line.get(..end).ok_or("invalid chunk size")?).map_err(|_| "invalid chunk size")?;
        let n = usize::from_str_radix(size.split(';').next().unwrap_or(""), 16).map_err(|_| "invalid chunk size")?;
        pos = pos.checked_add(end + 2).ok_or("invalid chunk offset")?;
        if n == 0 {
            return Ok(out);
        }
        if out.len().saturating_add(n) > limit {
            return Err("ComfyUI response is too large".into());
        }
        out.extend_from_slice(body.get(pos..pos.saturating_add(n)).ok_or("truncated chunk")?);
        pos = pos.checked_add(n + 2).ok_or("invalid chunk offset")?;
        if body.get(pos - 2..pos) != Some(&b"\r\n"[..]) {
            return Err("invalid chunk ending".into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    #[test]
    fn loopback_only_and_http_body_bounds() {
        assert!(endpoint("http://127.0.0.1:8188").is_ok());
        assert!(endpoint("http://[::1]:8188").is_ok());
        assert!(endpoint("http://192.168.1.2:8188").is_err());
        assert!(endpoint("https://127.0.0.1:8188").is_err());
        assert_eq!(parse_response(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok", 2).unwrap(), b"ok");
        assert!(parse_response(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nok", 2).is_err());
        assert_eq!(parse_response(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nok\r\n0\r\n\r\n", 2).unwrap(), b"ok");
    }
    #[test]
    fn stock_comfy_graph_has_inpaint_path() {
        let o = Options {
            backend: Backend::Comfy,
            endpoint: "http://127.0.0.1:8188".into(),
            checkpoint: "inpaint.safetensors".into(),
            prompt: "remove person".into(),
            negative: String::new(),
            steps: 20,
            guidance: 7.5,
            denoise: 1.0,
            seed: 1,
            edge: 1024,
            mask_margin: 0,
        };
        o.validate().unwrap();
        let g = workflow(&o, "photo.png", "mask.png", "edit");
        assert_eq!(g["6"]["class_type"], "InpaintModelConditioning");
        assert_eq!(g["6"]["inputs"]["mask"], json!(["2", 0]));
        assert_eq!(g["9"]["inputs"]["images"], json!(["8", 0]));
    }
    #[test]
    fn generated_pixels_affect_only_selected_area() {
        let mut original = Rgba8 { width: 3, height: 1, data: vec![[10, 20, 30, 255]; 3] };
        let generated = Rgba8 { width: 3, height: 1, data: vec![[210, 220, 230, 255]; 3] };
        let mask = Rgba8 { width: 3, height: 1, data: vec![[0, 0, 0, 255], [128, 128, 128, 255], [255, 255, 255, 255]] };
        composite(&mut original, &generated, &mask);
        assert_eq!(original.data[0], [10, 20, 30, 255]);
        assert_eq!(original.data[1], [110, 120, 130, 255]);
        assert_eq!(original.data[2], [210, 220, 230, 255]);
    }

    #[test]
    fn native_roi_keeps_full_photo_and_unselected_pixels() {
        let mut image = Rgba8::from_fn(2048, 1024, |x, y| [(x % 251) as u8, (y % 241) as u8, 60, 255]);
        let original = image.clone();
        let mask = Rgba8::from_fn(2048, 1024, |x, y| {
            let alpha = if x == 1899 && (20..100).contains(&y) {
                128
            } else if (1900..1990).contains(&x) && (20..100).contains(&y) {
                255
            } else {
                0
            };
            [alpha, alpha, alpha, 255]
        });
        let roi = selected_roi(&mask, 512).unwrap();
        assert!(roi.x > 0 && roi.x + roi.w <= image.width);
        assert!(roi.y + roi.h <= image.height);
        assert!(roi.model_w <= 512 && roi.model_h <= 512);
        let patch = roi_patch(&image, roi);
        assert_eq!((patch.width, patch.height), (roi.model_w, roi.model_h));
        let mut conditioning = roi_mask8(&mask, roi);
        let selected_before = conditioning.iter().filter(|&&px| px == 255).count();
        assert!(selected_before > 1000);
        dilate_mask8(&mut conditioning, roi.model_w, roi.model_h, REMOVE_MASK_MARGIN).unwrap();
        assert!(conditioning.iter().filter(|&&px| px == 255).count() > selected_before);
        let generated = Rgba8::filled(roi.model_w, roi.model_h, [250, 10, 20, 255]);
        composite_roi(&mut image, &generated, &mask, roi);
        assert_eq!((image.width, image.height), (2048, 1024));
        let fill = [250u8, 10, 20];
        for (before, (after, selection)) in original.data.iter().zip(image.data.iter().zip(&mask.data)) {
            if selection[0] == 0 {
                assert_eq!(before, after);
            } else if selection[0] == 128 {
                for c in 0..3 {
                    assert_eq!(after[c], ((u16::from(before[c]) * 127 + u16::from(fill[c]) * 128 + 127) / 255) as u8);
                }
                assert_eq!(after[3], before[3]);
            } else {
                assert_eq!(after, &[250, 10, 20, 255]);
            }
        }
    }

    #[test]
    fn conditioning_margin_is_bounded_at_canvas_edges() {
        let mut mask = vec![0u8; 8 * 8];
        mask[0] = 255;
        dilate_mask8(&mut mask, 8, 8, 2).unwrap();
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(mask[y * 8 + x], if x <= 2 && y <= 2 { 255 } else { 0 });
            }
        }
        let unchanged = mask.clone();
        dilate_mask8(&mut mask, 8, 8, 0).unwrap();
        assert_eq!(mask, unchanged);
        assert!(dilate_mask8(&mut mask, 8, 8, 65).is_err());
        assert!(dilate_mask8(&mut mask, 9, 8, 1).is_err());
        dilate_mask8(&mut mask, 8, 8, 64).unwrap();
        assert!(mask.iter().all(|&px| px == 255));
    }

    #[test]
    fn native_roi_at_frame_edge_keeps_thin_mask_in_model() {
        let mask = Rgba8::from_fn(3000, 1800, |x, y| {
            let a = if x == 0 && (0..1200).contains(&y) { 255 } else { 0 };
            [a, a, a, 255]
        });
        let roi = selected_roi(&mask, 512).unwrap();
        assert_eq!(roi.x, 0);
        assert!(roi.w <= 3000 && roi.h <= 1800);
        assert!(roi.model_w <= 512 && roi.model_h <= 512);
        assert!(roi_mask8(&mask, roi).contains(&255));
    }

    #[test]
    fn cropped_rotated_native_render_matches_mask_geometry() {
        let mut session = Session::with_demo();
        let id = session.active().unwrap();
        session.execute("photo.rotateRight", &json!({})).unwrap();
        session.execute("crop.set", &json!({"rect": [0.1, 0.2, 0.8, 0.9]})).unwrap();
        session.execute("mask.add", &json!({"kind": "radial", "center": [0.6, 0.5], "rx": 0.12, "ry": 0.15})).unwrap();
        let mask_id = session.active_mask.unwrap();
        let photo = session.catalog.photo(id).unwrap();
        let (w, h) = lightcraft_pipeline::native_output_size(photo.width as usize, photo.height as usize, &photo.develop);
        let edge = (w.round() as usize).max(h.round() as usize);
        let image = session.render_job(id, edge, edge, false, true).unwrap().run().rendered.unwrap().image;
        let mask = session
            .render_job(id, edge, edge, false, true)
            .unwrap()
            .with_overlay(Overlay::Mask { id: mask_id as u16, view: MaskView::WhiteOnBlack, color: [255; 3], opacity: 100 })
            .run()
            .rendered
            .unwrap()
            .image;
        assert_eq!((image.width, image.height), (w.round() as usize, h.round() as usize));
        assert_eq!((image.width, image.height), (mask.width, mask.height));
        let roi = selected_roi(&mask, 64).unwrap();
        assert!(image.width.max(image.height) > roi.model_w.max(roi.model_h));
        assert_eq!((roi_patch(&image, roi).width, roi_patch(&image, roi).height), (roi.model_w, roi.model_h));
    }
    #[test]
    fn removal_uses_context_prompt_and_keeps_user_negative_terms() {
        let (prompt, negative) = removal_prompts("red car");
        assert!(prompt.contains("background"));
        assert!(negative.contains("object"));
        assert!(negative.contains("red car"));
    }

    #[test]
    fn local_comfy_transport_uploads_and_reads_history_image() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            for expected in ["POST /upload/image", "POST /prompt", "GET /history/test-id", "GET /view?"] {
                let (mut stream, _) = listener.accept().unwrap();
                stream.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
                let mut request = Vec::new();
                let header_end = loop {
                    let mut part = [0u8; 1024];
                    let n = stream.read(&mut part).unwrap();
                    assert!(n > 0);
                    request.extend_from_slice(&part[..n]);
                    if let Some(i) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        break i + 4;
                    }
                };
                let headers = std::str::from_utf8(&request[..header_end]).unwrap().to_string();
                assert!(headers.starts_with(expected), "{headers}");
                let length = headers
                    .lines()
                    .find_map(|h| h.to_ascii_lowercase().strip_prefix("content-length:").and_then(|v| v.trim().parse::<usize>().ok()))
                    .unwrap();
                while request.len() < header_end + length {
                    let mut part = [0u8; 1024];
                    let n = stream.read(&mut part).unwrap();
                    assert!(n > 0);
                    request.extend_from_slice(&part[..n]);
                }
                let body = match expected {
                    "POST /upload/image" => {
                        assert!(request.windows(7).any(|w| w == b"PNGDATA"));
                        br#"{"name":"mask.png","subfolder":"","type":"input"}"#.to_vec()
                    }
                    "POST /prompt" => {
                        let v: Value = serde_json::from_slice(&request[header_end..]).unwrap();
                        assert!(v["prompt"].is_object());
                        br#"{"prompt_id":"test-id"}"#.to_vec()
                    }
                    "GET /history/test-id" => br#"{"test-id":{"status":{"completed":true},"outputs":{"9":{"images":[{"filename":"edit 1.png","subfolder":"","type":"output"}]}}}}"#.to_vec(),
                    _ => {
                        assert!(headers.contains("filename=edit%201.png"));
                        b"GENERATED".to_vec()
                    }
                };
                let reply = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                stream.write_all(reply.as_bytes()).unwrap();
                stream.write_all(&body).unwrap();
            }
        });
        assert_eq!(upload(addr, "mask.png", b"PNGDATA").unwrap(), "mask.png");
        let response = http(addr, "POST", "/prompt", br#"{"prompt":{}}"#, "application/json", 1024).unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&response).unwrap()["prompt_id"], "test-id");
        assert_eq!(poll_result(addr, "test-id", &std::sync::atomic::AtomicBool::new(false)).unwrap(), b"GENERATED");
        server.join().unwrap();
    }

    #[test]
    fn native_generation_options_accept_partial_denoise_with_active_steps() {
        let options = Options {
            backend: Backend::Native,
            endpoint: String::new(),
            checkpoint: String::new(),
            prompt: "fill the selected area".into(),
            negative: String::new(),
            steps: 24,
            guidance: 7.5,
            denoise: 0.99,
            seed: 4,
            edge: 512,
            mask_margin: 0,
        };
        assert!(options.validate().is_ok());
        let mut one_step = options.clone();
        one_step.steps = 100;
        one_step.denoise = 0.01;
        assert!(one_step.validate().is_ok());
        let mut zero = options.clone();
        zero.denoise = 0.0;
        assert!(zero.validate().unwrap_err().contains("denoise"));
        let mut nan = options.clone();
        nan.denoise = f64::NAN;
        assert!(nan.validate().is_err());
        let mut no_steps = options.clone();
        no_steps.steps = 2;
        no_steps.denoise = 0.49;
        assert!(no_steps.validate().unwrap_err().contains("one diffusion step"));
        let mut too_large = options;
        too_large.edge = 1024;
        assert!(too_large.validate().unwrap_err().contains("512"));
        too_large.edge = 512;
        too_large.mask_margin = 65;
        assert!(too_large.validate().unwrap_err().contains("maskMargin"));
    }
}
