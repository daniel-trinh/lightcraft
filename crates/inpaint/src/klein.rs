//! Native, batch-one FLUX.2 Klein 4B inpainting. The model files are supplied by
//! the caller; this module has no Python, service, download or CPU fallback.
//!
//! Sampling follows the Apache-2.0 Diffusers 0.38 `Flux2KleinInpaintPipeline`:
//! full-source VAE conditioning, a second copy of its latents as reference
//! tokens, flow-matching Euler steps, then latent preservation outside the mask.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::klein_sampling::{self, KleinSchedule};
use crate::klein_text::KleinTextEncoder;
use crate::klein_tokenizer::KleinTokenizer;
use crate::klein_transformer::{KleinConfig, KleinTransformer};
use crate::klein_vae::{KleinVae, KleinVaeConfig};
use crate::model::{Noise, best_device};
use crate::weights::{Backend, MultiBackend, TensorLimit, Weights};
use crate::{Error, Result};

const MAX_PROMPT_BYTES: usize = 4096;
const DISTILLED_STEPS: usize = 4;

/// How the original image is supplied as transformer reference tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KleinConditioning {
    /// Reference-faithful full source image. This remains the product default.
    #[default]
    FullImage,
    /// Experimental: replace selected source RGB pixels with white before
    /// encoding *only* the condition reference; preserve original clean latents.
    ErasedMask,
}

/// Parameters for the pinned, four-step Klein 4B checkpoint.
#[derive(Clone, Debug)]
pub struct KleinOptions {
    pub prompt: String,
    pub seed: u64,
    /// This distilled checkpoint supports exactly four scheduler steps.
    pub steps: usize,
    /// Fraction of the four-step schedule to run, in `(0, 1]`.
    pub strength: f64,
    pub conditioning: KleinConditioning,
}

impl Default for KleinOptions {
    fn default() -> Self {
        Self { prompt: String::new(), seed: 0, steps: DISTILLED_STEPS, strength: 1.0, conditioning: KleinConditioning::FullImage }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KleinPhase {
    Text,
    Encode,
    Denoise { step: usize, total: usize },
    Decode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KleinProgress {
    pub phase: KleinPhase,
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) { Err(Error::Cancelled) } else { Ok(()) }
}

fn validate_input(rgb: &[u8], mask: &[u8], width: usize, height: usize, options: &KleinOptions) -> Result<()> {
    if !(64..=1024).contains(&width) || !(64..=1024).contains(&height) || !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(Error::Model("Klein image dimensions must be 64..=1024 and divisible by 16".into()));
    }
    let pixels = width.checked_mul(height).ok_or_else(|| Error::Model("Klein image dimensions overflow".into()))?;
    let rgb_bytes = pixels.checked_mul(3).ok_or_else(|| Error::Model("Klein RGB dimensions overflow".into()))?;
    if rgb.len() != rgb_bytes || mask.len() != pixels {
        return Err(Error::Model("Klein RGB or mask byte length differs from image dimensions".into()));
    }
    if !mask.iter().any(|value| *value >= 128) {
        return Err(Error::Model("Klein mask contains no selected pixels".into()));
    }
    if options.prompt.len() > MAX_PROMPT_BYTES {
        return Err(Error::Model("Klein prompt exceeds 4096 UTF-8 bytes".into()));
    }
    if options.steps != DISTILLED_STEPS {
        return Err(Error::Model("Klein 4B is distilled for exactly four scheduler steps".into()));
    }
    if !options.strength.is_finite() || !(0.0..=1.0).contains(&options.strength) || options.strength == 0.0 {
        return Err(Error::Model("Klein strength must be finite and in (0,1]".into()));
    }
    Ok(())
}

fn rgb_to_tensor(rgb: &[u8], width: usize, height: usize, device: &Device) -> Result<Tensor> {
    let pixels = width.checked_mul(height).ok_or_else(|| Error::Model("Klein RGB dimensions overflow".into()))?;
    if pixels.checked_mul(3) != Some(rgb.len()) {
        return Err(Error::Model("Klein RGB byte length differs from dimensions".into()));
    }
    let mut chw = vec![0.0f32; rgb.len()];
    for (pixel, channels) in rgb.as_chunks::<3>().0.iter().enumerate() {
        for (channel, value) in channels.iter().enumerate() {
            let index = channel * pixels + pixel;
            let destination = chw.get_mut(index).ok_or_else(|| Error::Model("Klein RGB layout overflow".into()))?;
            *destination = f32::from(*value) * (2.0 / 255.0) - 1.0;
        }
    }
    Ok(Tensor::from_vec(chw, (1, 3, height, width), device)?)
}

fn erase_selected_rgb(rgb: &[u8], mask: &[u8]) -> Result<Vec<u8>> {
    if rgb.len() != mask.len().checked_mul(3).ok_or_else(|| Error::Model("Klein erased-mask dimensions overflow".into()))? {
        return Err(Error::Model("Klein erased-mask RGB and mask lengths differ".into()));
    }
    let mut erased = rgb.to_vec();
    for (channels, selected) in erased.as_chunks_mut::<3>().0.iter_mut().zip(mask) {
        if *selected >= 128 {
            *channels = [255; 3];
        }
    }
    Ok(erased)
}

fn tensor_to_rgb(image: &Tensor, width: usize, height: usize) -> Result<Vec<u8>> {
    if image.dims4()? != (1, 3, height, width) {
        return Err(Error::Model("Klein VAE returned an unexpected RGB shape".into()));
    }
    let values = image.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let pixels = width.checked_mul(height).ok_or_else(|| Error::Model("Klein RGB dimensions overflow".into()))?;
    let rgb_bytes = pixels.checked_mul(3).ok_or_else(|| Error::Model("Klein RGB dimensions overflow".into()))?;
    let mut rgb = vec![0u8; rgb_bytes];
    for pixel in 0..pixels {
        for channel in 0..3 {
            let value = *values.get(channel * pixels + pixel).ok_or_else(|| Error::Model("Klein VAE RGB layout is incomplete".into()))?;
            if !value.is_finite() {
                return Err(Error::Model("Klein VAE produced a non-finite RGB value".into()));
            }
            let destination = rgb.get_mut(pixel * 3 + channel).ok_or_else(|| Error::Model("Klein RGB output layout overflow".into()))?;
            *destination = ((value.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8;
        }
    }
    Ok(rgb)
}

fn vae_builder(model_dir: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    let weights = Weights::open(&model_dir.join("vae").join("diffusion_pytorch_model.safetensors"))?;
    Ok(VarBuilder::from_backend(Box::new(Backend(weights)), DType::F32, device.clone()))
}

fn transformer_builder(model_dir: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    let weights =
        Weights::open_with_limit(&model_dir.join("transformer").join("diffusion_pytorch_model.safetensors"), TensorLimit::KleinTransformer)?;
    Ok(VarBuilder::from_backend(Box::new(Backend(weights)), DType::BF16, device.clone()))
}

fn text_builder(model_dir: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    let shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        .iter()
        .map(|name| Weights::open_with_limit(&model_dir.join("text_encoder").join(name), TensorLimit::KleinText))
        .collect::<Result<Vec<_>>>()?;
    Ok(VarBuilder::from_backend(Box::new(MultiBackend::new(shards)?), DType::BF16, device.clone()))
}

fn timestep_for_transformer(sigma: f32, device: &Device) -> Result<Tensor> {
    // Diffusers computes scheduler.timesteps = sigma(F32) * 1000(F32), then
    // converts to BF16 before dividing by 1000. Direct BF16 sigma rounding
    // differs at later steps and changes the transformer's time embedding.
    let timestep = Tensor::new(sigma * 1000.0, device)?.to_dtype(DType::BF16)?.reshape((1,))?;
    Ok((timestep / 1000.0)?)
}

fn run_steps(
    model_dir: &Path,
    source: &Tensor,
    source_condition: &Tensor,
    text_features: &Tensor,
    mask: &[u8],
    width: usize,
    height: usize,
    schedule: &KleinSchedule,
    seed: u64,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(KleinProgress),
    device: &Device,
) -> Result<Tensor> {
    let mut rng = Noise::new(seed);
    let initial_noise = klein_sampling::pack_latents(&rng.tensor((1, 128, height / 16, width / 16), device, DType::BF16)?)?;
    let initial_sigma = *schedule.sigmas.first().ok_or_else(|| Error::Model("Klein schedule has no starting sigma".into()))?;
    let total = schedule
        .sigmas
        .len()
        .checked_sub(1)
        .filter(|count| *count > 0)
        .ok_or_else(|| Error::Model("Klein schedule has no denoising steps".into()))?;
    let mut latents = klein_sampling::scale_noise(source, &initial_noise, initial_sigma)?;
    let image_ids = klein_sampling::combined_image_ids(width, height, device)?;
    let text_ids = klein_sampling::text_ids(text_features.dim(1)?, device)?;
    let mask = klein_sampling::packed_mask(mask, width, height, DType::BF16, device)?;
    check_cancel(cancel)?;
    let transformer = KleinTransformer::new(&KleinConfig::default(), transformer_builder(model_dir, device)?)?;
    for (index, sigmas) in schedule.sigmas.windows(2).enumerate() {
        check_cancel(cancel)?;
        progress(KleinProgress { phase: KleinPhase::Denoise { step: index + 1, total } });
        let timestep = timestep_for_transformer(sigmas[0], device)?;
        let model_input = Tensor::cat(&[latents.clone(), source_condition.clone()], 1)?;
        let predicted = match transformer.forward_cancel(&model_input, text_features, &timestep, &image_ids, &text_ids, cancel) {
            Ok(predicted) => predicted,
            Err(_) if cancel.load(Ordering::Relaxed) => return Err(Error::Cancelled),
            Err(error) => return Err(error.into()),
        };
        let predicted = predicted.narrow(1, 0, source.dim(1)?)?;
        latents = klein_sampling::euler_inpaint_step(&latents, &predicted, source, &initial_noise, &mask, sigmas[0], sigmas[1])?;
    }
    check_cancel(cancel)?;
    Ok(latents)
}

/// Generate an RGB8 HWC image using the local Klein 4B model. `mask` is one
/// grayscale byte per pixel: values >=128 request replacement. The returned
/// image is the whole generated canvas. The caller owns compositing original
/// pixels outside the selected mask, and any crop/resize before this call.
pub fn generate(
    model_dir: &Path,
    rgb: &[u8],
    mask: &[u8],
    width: usize,
    height: usize,
    options: &KleinOptions,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(KleinProgress),
) -> Result<Vec<u8>> {
    validate_input(rgb, mask, width, height, options)?;
    check_cancel(cancel)?;
    let device = best_device()?;
    if !matches!(device, Device::Cuda(_)) {
        return Err(Error::Model("Klein 4B generation requires an NVIDIA CUDA GPU; CPU inference is unsupported".into()));
    }
    let schedule = klein_sampling::schedule(width, height, options.steps, options.strength)?;

    progress(KleinProgress { phase: KleinPhase::Text });
    let text_features = {
        let tokenizer = KleinTokenizer::load(model_dir.join("tokenizer"))?;
        let tokens = tokenizer.encode(&options.prompt, 512)?;
        check_cancel(cancel)?;
        let text = KleinTextEncoder::new(text_builder(model_dir, &device)?)?;
        text.encode(&tokens.input_ids, &tokens.attention_mask, cancel)?
    };
    check_cancel(cancel)?;

    progress(KleinProgress { phase: KleinPhase::Encode });
    let (source, source_condition) = {
        let vae = KleinVae::new(KleinVaeConfig::default(), vae_builder(model_dir, &device)?)?;
        let image = rgb_to_tensor(rgb, width, height, &device)?;
        let encoded = vae.encode(&image)?.to_dtype(DType::BF16)?;
        let source = klein_sampling::pack_latents(&encoded)?;
        let source_condition = match options.conditioning {
            KleinConditioning::FullImage => source.clone(),
            KleinConditioning::ErasedMask => {
                check_cancel(cancel)?;
                let erased_rgb = erase_selected_rgb(rgb, mask)?;
                let erased_image = rgb_to_tensor(&erased_rgb, width, height, &device)?;
                let encoded = vae.encode(&erased_image)?.to_dtype(DType::BF16)?;
                klein_sampling::pack_latents(&encoded)?
            }
        };
        (source, source_condition)
    };
    check_cancel(cancel)?;

    let latents =
        run_steps(model_dir, &source, &source_condition, &text_features, mask, width, height, &schedule, options.seed, cancel, progress, &device)?;
    drop(text_features);
    drop(source);
    drop(source_condition);
    check_cancel(cancel)?;

    progress(KleinProgress { phase: KleinPhase::Decode });
    let decoded = {
        let vae = KleinVae::new(KleinVaeConfig::default(), vae_builder(model_dir, &device)?)?;
        let spatial = klein_sampling::unpack_latents(&latents.to_dtype(DType::F32)?, width, height)?;
        vae.decode(&spatial)?
    };
    check_cancel(cancel)?;
    tensor_to_rgb(&decoded, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_inputs_before_model_or_gpu_access() {
        let cancel = AtomicBool::new(false);
        let rgb = vec![127; 64 * 64 * 3];
        let mut mask = vec![0; 64 * 64];
        let mut events = |_| {};
        let options = KleinOptions::default();
        let absent = Path::new("absent-model");
        assert!(generate(absent, &rgb, &mask, 64, 64, &options, &cancel, &mut events).is_err());
        mask[0] = 255;
        assert!(generate(absent, &rgb, &mask, 63, 64, &options, &cancel, &mut events).is_err());
        assert!(generate(absent, &rgb[..rgb.len() - 1], &mask, 64, 64, &options, &cancel, &mut events).is_err());
        assert!(generate(absent, &rgb, &mask, 64, 64, &KleinOptions { steps: 5, ..options.clone() }, &cancel, &mut events).is_err());
        assert!(generate(absent, &rgb, &mask, 64, 64, &KleinOptions { strength: f64::NAN, ..options.clone() }, &cancel, &mut events).is_err());
        assert!(generate(absent, &rgb, &mask, 64, 64, &KleinOptions { prompt: "x".repeat(4097), ..options }, &cancel, &mut events).is_err());
    }

    #[test]
    fn rgb_roundtrip_keeps_channel_order() -> Result<()> {
        let rgb = [0, 127, 255, 12, 180, 80];
        let image = rgb_to_tensor(&rgb, 2, 1, &Device::Cpu)?;
        assert_eq!(tensor_to_rgb(&image, 2, 1)?, rgb);
        Ok(())
    }

    #[test]
    fn erased_condition_changes_only_selected_pixels() -> Result<()> {
        let rgb = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mask = [0, 128, 127];
        let erased = erase_selected_rgb(&rgb, &mask)?;
        assert_eq!(erased, [1, 2, 3, 255, 255, 255, 7, 8, 9]);
        assert_eq!(rgb, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert!(erase_selected_rgb(&rgb[..8], &mask).is_err());
        Ok(())
    }
}
