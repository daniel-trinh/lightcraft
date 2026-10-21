//! Point-prompt SAM 1 ViT-B using Candle's Apache-2.0 Segment Anything implementation.
//!
//! The image is resized proportionally to a 1024-pixel long edge and padded on its bottom and
//! right by Candle. Decoder logits still cover that padded square. [`Prediction::logits`] removes
//! the padding geometrically so the stored square grid covers the *whole photograph*.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::sam::{IMAGE_SIZE, Sam};

use crate::weights::{Backend, Weights};
use crate::{Error, Result};

/// Side of the stored, image-covering logit grid. Compatible with `SegMask::from_logits`.
pub const MASK_SIDE: usize = 256;
const MAX_SOURCE_PIXELS: usize = 64 * 1024 * 1024;
const MAX_WEIGHT_BYTES: u64 = 600_000_000;
const MAX_CLICKS: usize = 64;

/// A normalized click in the upright, uncropped photograph or detail region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Click {
    pub x: f64,
    pub y: f64,
    pub positive: bool,
}

/// SAM 1 ViT-B parameters are 768-wide, 12 layers, 12 heads, with global attention in these
/// four blocks. The file must be an F32 safetensors checkpoint with original SAM tensor names.
pub struct Sam1 {
    model: Sam,
    device: Device,
}

/// Image encoder output and the proportional dimensions used before Candle's bottom/right pad.
pub struct Encoded {
    embedding: Tensor,
    width: usize,
    height: usize,
}

/// Selected mask logits over the *un-padded image*, stretched to a square for `SegMask`.
pub struct Prediction {
    pub logits: Vec<f32>,
    pub score: f32,
}

impl Sam1 {
    /// Load a checkpoint onto an explicitly selected device. File size and safetensors layout
    /// are bounded before any model tensor is allocated.
    pub fn load(path: &Path, device: Device) -> Result<Self> {
        let size = std::fs::metadata(path).map_err(|e| Error::Sam1(format!("{}: {e}", path.display())))?.len();
        if !(1..=MAX_WEIGHT_BYTES).contains(&size) {
            return Err(Error::Sam1(format!("checkpoint has an unsupported size of {size} bytes")));
        }
        let weights = Weights::open(path).map_err(|e| Error::Sam1(e.to_string()))?;
        let vb = VarBuilder::from_backend(Box::new(Backend(Arc::clone(&weights))), DType::F32, device.clone());
        let model = Sam::new(768, 12, 12, &[2, 5, 8, 11], vb)?;
        Ok(Self { model, device })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Encode interleaved 8-bit RGB. This can take a while; cancellation is checked on both
    /// sides of the encoder call because Candle does not support interrupting an active kernel.
    pub fn encode(&self, rgb: &[u8], width: usize, height: usize, cancel: &AtomicBool) -> Result<Encoded> {
        cancelled(cancel)?;
        validate_rgb(rgb, width, height)?;
        let (w, h) = resized_dims(width, height)?;
        let chw = resize_rgb_chw(rgb, width, height, w, h)?;
        cancelled(cancel)?;
        let image = Tensor::from_vec(chw, (3, h, w), &self.device)?;
        let embedding = self.model.embeddings(&image)?;
        cancelled(cancel)?;
        Ok(Encoded { embedding, width: w, height: h })
    }

    /// Decode positive/negative clicks, reusing the image embedding. One click requests three
    /// candidate masks and selects the greatest predicted IoU; later clicks request one mask.
    pub fn segment_clicks(&self, encoded: &Encoded, clicks: &[Click], cancel: &AtomicBool) -> Result<Prediction> {
        cancelled(cancel)?;
        if clicks.is_empty() || clicks.len() > MAX_CLICKS || !clicks.iter().any(|c| c.positive) {
            return Err(Error::Sam1("needs 1–64 clicks including a positive click".into()));
        }
        if clicks.iter().any(|c| !c.x.is_finite() || !c.y.is_finite() || !(0.0..=1.0).contains(&c.x) || !(0.0..=1.0).contains(&c.y)) {
            return Err(Error::Sam1("click coordinates must be finite and inside the image".into()));
        }
        let prompts: Vec<_> = clicks.iter().map(|c| (c.x, c.y, c.positive)).collect();
        let (masks, scores) = self.model.forward_for_embeddings(&encoded.embedding, encoded.height, encoded.width, &prompts, clicks.len() == 1)?;
        cancelled(cancel)?;
        let scores = scores.get(0)?.to_vec1::<f32>()?;
        let (index, score) = best_score(&scores)?;
        let padded = masks.get(0)?.get(index)?.to_vec2::<f32>()?;
        let logits = unpad_logits(&padded, encoded.width, encoded.height)?;
        Ok(Prediction { logits, score })
    }
}

fn cancelled(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) { Err(Error::Sam1("request cancelled".into())) } else { Ok(()) }
}

fn validate_rgb(rgb: &[u8], w: usize, h: usize) -> Result<()> {
    let pixels = w
        .checked_mul(h)
        .filter(|n| *n > 0 && *n <= MAX_SOURCE_PIXELS)
        .ok_or_else(|| Error::Sam1("image dimensions are invalid or too large".into()))?;
    if pixels.checked_mul(3) != Some(rgb.len()) {
        return Err(Error::Sam1("RGB buffer size does not match its dimensions".into()));
    }
    Ok(())
}

/// Match ResizeLongestSide's rounded integer dimensions, never exceeding 1024.
fn resized_dims(w: usize, h: usize) -> Result<(usize, usize)> {
    let long = w.max(h);
    if w == 0 || h == 0 || w.checked_mul(h).is_none_or(|n| n > MAX_SOURCE_PIXELS) {
        return Err(Error::Sam1("image dimensions are invalid or too large".into()));
    }
    let scaled = |n: usize| (((n as f64) * IMAGE_SIZE as f64 / long as f64) + 0.5).floor() as usize;
    Ok((scaled(w).clamp(1, IMAGE_SIZE), scaled(h).clamp(1, IMAGE_SIZE)))
}

/// Bilinear half-pixel resize into channel-first RGB values 0..255. The engine already gives a
/// reduced preview, so this typically adjusts only 1008→1024 pixels.
fn resize_rgb_chw(rgb: &[u8], w: usize, h: usize, out_w: usize, out_h: usize) -> Result<Vec<f32>> {
    let mut out = vec![0f32; 3 * out_w * out_h];
    for oy in 0..out_h {
        let fy = ((oy as f64 + 0.5) * h as f64 / out_h as f64 - 0.5).clamp(0.0, (h - 1) as f64);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(h - 1);
        let ty = (fy - y0 as f64) as f32;
        for ox in 0..out_w {
            let fx = ((ox as f64 + 0.5) * w as f64 / out_w as f64 - 0.5).clamp(0.0, (w - 1) as f64);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let tx = (fx - x0 as f64) as f32;
            for c in 0..3 {
                let at = |x: usize, y: usize| {
                    rgb.get((y * w + x) * 3 + c).copied().map(f32::from).ok_or_else(|| Error::Sam1("RGB sample is outside the image buffer".into()))
                };
                let a = at(x0, y0)? * (1.0 - tx) + at(x1, y0)? * tx;
                let b = at(x0, y1)? * (1.0 - tx) + at(x1, y1)? * tx;
                let dst = out
                    .get_mut(c * out_w * out_h + oy * out_w + ox)
                    .ok_or_else(|| Error::Sam1("resized RGB sample is outside the image buffer".into()))?;
                *dst = a * (1.0 - ty) + b * ty;
            }
        }
    }
    Ok(out)
}

fn best_score(scores: &[f32]) -> Result<(usize, f32)> {
    scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, s)| s.is_finite())
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or_else(|| Error::Sam1("returned no finite mask score".into()))
}

/// Sample the *valid* region of the 256-square decoder grid into an image-covering square.
/// The decoder's coordinate of image pixel `(x,y)` is `(x*256/1024,y*256/1024)`, independent
/// of image aspect. Half-pixel centres preserve an exact identity for square inputs.
fn unpad_logits(src: &[Vec<f32>], valid_w: usize, valid_h: usize) -> Result<Vec<f32>> {
    if valid_w == 0
        || valid_h == 0
        || valid_w > IMAGE_SIZE
        || valid_h > IMAGE_SIZE
        || src.len() != MASK_SIDE
        || src.iter().any(|row| row.len() != MASK_SIDE || row.iter().any(|v| !v.is_finite()))
    {
        return Err(Error::Sam1("returned an invalid mask grid".into()));
    }
    let mut out = vec![0f32; MASK_SIDE * MASK_SIDE];
    for y in 0..MASK_SIDE {
        let fy = ((y as f32 + 0.5) * valid_h as f32 / IMAGE_SIZE as f32 - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(MASK_SIDE - 1);
        let y1 = (y0 + 1).min(MASK_SIDE - 1);
        let ty = fy - y0 as f32;
        for x in 0..MASK_SIDE {
            let fx = ((x as f32 + 0.5) * valid_w as f32 / IMAGE_SIZE as f32 - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(MASK_SIDE - 1);
            let x1 = (x0 + 1).min(MASK_SIDE - 1);
            let tx = fx - x0 as f32;
            let sample = |px: usize, py: usize| {
                src.get(py).and_then(|row| row.get(px)).copied().ok_or_else(|| Error::Sam1("mask sample is outside the decoder grid".into()))
            };
            let a = sample(x0, y0)? * (1.0 - tx) + sample(x1, y0)? * tx;
            let b = sample(x0, y1)? * (1.0 - tx) + sample(x1, y1)? * tx;
            let dst = out.get_mut(y * MASK_SIDE + x).ok_or_else(|| Error::Sam1("mask sample is outside the output grid".into()))?;
            *dst = a * (1.0 - ty) + b * ty;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_and_bad_input_are_bounded() {
        assert_eq!(resized_dims(4000, 2000).unwrap(), (1024, 512));
        assert_eq!(resized_dims(2000, 4000).unwrap(), (512, 1024));
        assert_eq!(resized_dims(900, 900).unwrap(), (1024, 1024));
        assert!(resized_dims(0, 1).is_err());
        assert!(validate_rgb(&[0; 5], 2, 1).is_err());
        assert!(validate_rgb(&[], usize::MAX, 3).is_err());
    }

    #[test]
    fn unpadding_maps_portrait_and_landscape_edges() {
        // A vertical stripe in the last valid column of a portrait must reach the right edge,
        // even though padded columns beyond the portrait contain a different value.
        let cols: Vec<Vec<f32>> = (0..MASK_SIDE).map(|_| (0..MASK_SIDE).map(|x| x as f32).collect()).collect();
        let portrait = unpad_logits(&cols, 512, 1024).unwrap();
        assert!((portrait[MASK_SIDE - 1] - 127.25).abs() < 0.01);
        let square = unpad_logits(&cols, 1024, 1024).unwrap();
        assert_eq!(square[MASK_SIDE - 1], 255.0);
        let rows: Vec<Vec<f32>> = (0..MASK_SIDE).map(|y| vec![y as f32; MASK_SIDE]).collect();
        let landscape = unpad_logits(&rows, 1024, 512).unwrap();
        assert!((landscape[(MASK_SIDE - 1) * MASK_SIDE] - 127.25).abs() < 0.01);
        let mut corrupt = cols;
        corrupt[0][0] = f32::NAN;
        assert!(unpad_logits(&corrupt, 1024, 1024).is_err());
    }

    #[test]
    fn selects_best_finite_iou_and_cancels() {
        assert_eq!(best_score(&[0.2, 0.9, 0.4]).unwrap(), (1, 0.9));
        assert_eq!(best_score(&[f32::NAN, 0.5]).unwrap(), (1, 0.5));
        assert!(best_score(&[f32::NAN]).is_err());
        let cancel = AtomicBool::new(true);
        assert!(cancelled(&cancel).is_err());
    }
}
