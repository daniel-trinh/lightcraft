//! SD 1.5 9-channel latent inpainting on Candle CPU or Metal.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::{clip, ddim, schedulers::SchedulerConfig, unet_2d, vae};

use crate::{
    download::installed,
    tokenizer::Tokenizer,
    weights::{Backend, Weights},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Model(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Candle(#[from] candle_core::Error),
    #[error("inpainting cancelled")]
    Cancelled,
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug)]
pub struct GenerateOptions {
    pub prompt: String,
    pub negative: String,
    pub seed: u64,
    pub steps: usize,
    pub guidance: f64,
}

impl GenerateOptions {
    pub fn validate(&self) -> Result<()> {
        if self.prompt.len() > 4096 || self.negative.len() > 4096 {
            return Err(Error::Model("prompt exceeds 4096 bytes".into()));
        }
        if !(1..=100).contains(&self.steps) {
            return Err(Error::Model("steps must be 1..100".into()));
        }
        if !self.guidance.is_finite() || !(1.0..=20.0).contains(&self.guidance) {
            return Err(Error::Model("guidance must be 1..20".into()));
        }
        Ok(())
    }
}

pub struct Model {
    device: Device,
    dtype: DType,
    tokenizer: Tokenizer,
    clip: clip::ClipTextTransformer,
    vae: vae::AutoEncoderKL,
    unet: unet_2d::UNet2DConditionModel,
}

fn best_device() -> Device {
    #[cfg(target_os = "macos")]
    if let Ok(device) = Device::new_metal(0) {
        return device;
    }
    Device::Cpu
}

fn builder(dir: &Path, component: &str, file: &str, dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
    let weights = Weights::open(&dir.join(component).join(file))?;
    Ok(VarBuilder::from_backend(Box::new(Backend(weights)), dtype, device.clone()))
}

fn unet_config() -> unet_2d::UNet2DConditionModelConfig {
    let block = |out_channels, use_cross_attn| unet_2d::BlockConfig { out_channels, use_cross_attn, attention_head_dim: 8 };
    unet_2d::UNet2DConditionModelConfig {
        blocks: vec![block(320, Some(1)), block(640, Some(1)), block(1280, Some(1)), block(1280, None)],
        center_input_sample: false,
        cross_attention_dim: 768,
        downsample_padding: 1,
        flip_sin_to_cos: true,
        freq_shift: 0.0,
        layers_per_block: 2,
        mid_block_scale_factor: 1.0,
        norm_eps: 1e-5,
        norm_num_groups: 32,
        // Candle 0.9.2's sliced path stacks batches into rank 4 before a rank-3
        // reshape. Use its working unsliced attention path for the bounded canvas.
        sliced_attention_size: None,
        use_linear_projection: false,
    }
}

fn vae_config() -> vae::AutoEncoderKLConfig {
    vae::AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: 4,
        norm_num_groups: 32,
        use_quant_conv: true,
        use_post_quant_conv: true,
    }
}

impl Model {
    pub fn load(dir: &Path) -> Result<Self> {
        if !installed(dir) {
            return Err(Error::Model(format!("inpainting checkpoint is missing or incomplete in {}", dir.display())));
        }
        let device = best_device();
        let dtype = if matches!(device, Device::Cpu) { DType::F32 } else { DType::F16 };
        let tokenizer = Tokenizer::load(&dir.join("tokenizer"))?;
        let clip = clip::ClipTextTransformer::new(builder(dir, "text_encoder", "model.fp16.safetensors", dtype, &device)?, &clip::Config::v1_5())?;
        let vae = vae::AutoEncoderKL::new(builder(dir, "vae", "diffusion_pytorch_model.fp16.safetensors", dtype, &device)?, 3, 3, vae_config())?;
        let unet = unet_2d::UNet2DConditionModel::new(
            builder(dir, "unet", "diffusion_pytorch_model.fp16.safetensors", dtype, &device)?,
            9,
            4,
            false,
            unet_config(),
        )?;
        Ok(Self { device, dtype, tokenizer, clip, vae, unet })
    }

    /// `rgb8` is HWC RGB; `mask8` is one byte/pixel, white requests replacement.
    /// Dimensions must be divisible by 8. Caller owns padding, framing and final composite.
    pub fn generate(
        &mut self,
        rgb8: &[u8],
        mask8: &[u8],
        width: usize,
        height: usize,
        options: &GenerateOptions,
        cancel: &AtomicBool,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Vec<u8>> {
        options.validate()?;
        validate_input(rgb8, mask8, width, height)?;
        check_cancel(cancel)?;
        let (mask, masked_image) = conditioning_inputs(rgb8, mask8, width, height, &self.device, self.dtype)?;
        let mask_latents = (self.vae.encode(&masked_image)?.sample()? * 0.18215)?.to_dtype(self.dtype)?;
        let low_mask = mask.interpolate2d(height / 8, width / 8)?;
        let (positive, negative) = (self.embed(&options.prompt)?, self.embed(&options.negative)?);
        let embeddings = Tensor::cat(&[negative, positive], 0)?;
        let mut scheduler = ddim::DDIMSchedulerConfig::default().build(options.steps)?;
        let mut rng = Noise::new(options.seed);
        let noise = rng.tensor((1, 4, height / 8, width / 8), &self.device, self.dtype)?;
        let mut latents = (noise * scheduler.init_noise_sigma())?;
        for (index, &timestep) in scheduler.timesteps().to_vec().iter().enumerate() {
            check_cancel(cancel)?;
            let duplicated = Tensor::cat(&[&latents, &latents], 0)?;
            let scaled = scheduler.scale_model_input(duplicated, timestep)?;
            let masks = Tensor::cat(&[&low_mask, &low_mask], 0)?;
            let masked = Tensor::cat(&[&mask_latents, &mask_latents], 0)?;
            let conditioned = Tensor::cat(&[scaled, masks, masked], 1)?;
            let predicted = self.unet.forward(&conditioned, timestep as f64, &embeddings)?;
            let chunks = predicted.chunk(2, 0)?;
            let (Some(uncond), Some(text)) = (chunks.first(), chunks.get(1)) else {
                return Err(Error::Model("UNet returned an invalid batch".into()));
            };
            let guided = (uncond + ((text - uncond)? * options.guidance)?)?;
            latents = scheduler.step(&guided, timestep, &latents)?;
            progress(index + 1, options.steps);
        }
        check_cancel(cancel)?;
        let image = self.vae.decode(&(latents / 0.18215)?)?;
        let image = ((image / 2.0)? + 0.5)?.clamp(0f32, 1.0)?.permute((0, 2, 3, 1))?;
        Ok((image * 255.0)?.to_dtype(DType::U8)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?)
    }

    fn embed(&self, text: &str) -> Result<Tensor> {
        let ids = self.tokenizer.encode(text)?;
        let ids = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        Ok(candle_core::Module::forward(&self.clip, &ids)?.to_dtype(self.dtype)?)
    }
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) { Err(Error::Cancelled) } else { Ok(()) }
}

fn conditioning_inputs(rgb8: &[u8], mask8: &[u8], width: usize, height: usize, device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
    let image = Tensor::from_vec(rgb8.to_vec(), (1, height, width, 3), &Device::Cpu)?
        .permute((0, 3, 1, 2))?
        .to_dtype(DType::F32)?
        .affine(2.0 / 255.0, -1.0)?
        .to_device(device)?
        .to_dtype(dtype)?;
    let mask = Tensor::from_vec(mask8.to_vec(), (1, 1, height, width), &Device::Cpu)?.to_device(device)?.ge(128u8)?.to_dtype(dtype)?;
    let keep = mask.le(0.5)?.repeat((1, 3, 1, 1))?.to_dtype(dtype)?;
    let masked_image = image.broadcast_mul(&keep)?;
    Ok((mask, masked_image))
}

fn validate_input(rgb: &[u8], mask: &[u8], width: usize, height: usize) -> Result<()> {
    if !(64..=512).contains(&width) || !(64..=512).contains(&height) || !width.is_multiple_of(8) || !height.is_multiple_of(8) {
        return Err(Error::Model("inpainting dimensions must be 64..512 and multiples of 8".into()));
    }
    let pixels = width.checked_mul(height).ok_or_else(|| Error::Model("image too large".into()))?;
    if mask.len() != pixels || rgb.len() != pixels * 3 {
        return Err(Error::Model("image or mask byte length does not match dimensions".into()));
    }
    if !mask.iter().any(|&a| a > 127) {
        return Err(Error::Model("mask has no selected pixels".into()));
    }
    Ok(())
}

/// Small deterministic normal generator for initial latent noise; avoids a native RNG library.
struct Noise(u64);
impl Noise {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed })
    }
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let n = self.0.wrapping_mul(0x2545F4914F6CDD1D);
        ((n >> 11) as f64 + 0.5) * (1.0 / ((1u64 << 53) as f64))
    }
    fn tensor(&mut self, shape: (usize, usize, usize, usize), device: &Device, dtype: DType) -> Result<Tensor> {
        let count = shape.0 * shape.1 * shape.2 * shape.3;
        let mut values = Vec::with_capacity(count);
        while values.len() < count {
            let u = self.uniform();
            let v = self.uniform();
            let r = (-2.0 * u.ln()).sqrt();
            let a = std::f64::consts::TAU * v;
            values.push((r * a.cos()) as f32);
            if values.len() < count {
                values.push((r * a.sin()) as f32);
            }
        }
        Ok(Tensor::from_vec(values, shape, device)?.to_dtype(dtype)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn white_mask_removes_pixels_from_conditioning_and_cancellation_is_reported() {
        let mut rgb = vec![0u8; 64 * 64 * 3];
        rgb[..3].copy_from_slice(&[255, 128, 0]);
        rgb[3..6].copy_from_slice(&[255, 128, 0]);
        let mut mask = vec![0u8; 64 * 64];
        mask[0] = 255;
        let (mask_t, condition) = conditioning_inputs(&rgb, &mask, 64, 64, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(mask_t.dims(), [1, 1, 64, 64]);
        assert_eq!(condition.dims(), [1, 3, 64, 64]);
        assert_eq!(condition.get(0).unwrap().get(0).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()[0], 0.0);
        assert_eq!(condition.get(0).unwrap().get(0).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()[1], 1.0);
        let cancelled = AtomicBool::new(true);
        assert!(matches!(check_cancel(&cancelled), Err(Error::Cancelled)));
    }
    #[test]
    fn rejects_invalid_input_before_loading_a_model() {
        assert!(validate_input(&[], &[], 64, 64).is_err());
        assert!(validate_input(&[0; 64 * 64 * 3], &[0; 64 * 64], 64, 64).is_err());
        assert!(validate_input(&[0; 64 * 64 * 3], &[255; 64 * 64], 64, 64).is_ok());
        assert!(validate_input(&[], &[], 513, 64).is_err());
    }
    #[test]
    fn initial_noise_is_seeded_and_finite() {
        let mut a = Noise::new(3);
        let mut b = Noise::new(3);
        let x = a.tensor((1, 4, 8, 8), &Device::Cpu, DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let y = b.tensor((1, 4, 8, 8), &Device::Cpu, DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(x, y);
        assert!(x.iter().all(|v| v.is_finite()));
    }
}
