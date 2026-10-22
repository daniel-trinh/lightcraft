//! Bounded FLUX.2 Klein inpaint sampling helpers.
//!
//! The scheduler, IDs and mask layout follow the Apache-2.0 Diffusers 0.38.0
//! `Flux2KleinInpaintPipeline` and `FlowMatchEulerDiscreteScheduler`.
//! Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team.
//! Adapted to Rust/Candle for LightCraft (2026); no checkpoint is bundled here.

use candle_core::{DType, Device, Result, Tensor};

const MIN_EDGE: usize = 64;
const MAX_EDGE: usize = 1024;
const PACKED_CHANNELS: usize = 128;
const MAX_IMAGE_TOKENS: usize = 4096;
const MAX_STEPS: usize = 100;
const MAX_TEXT_TOKENS: usize = 512;

fn image_grid(width: usize, height: usize) -> Result<(usize, usize, usize)> {
    if !(MIN_EDGE..=MAX_EDGE).contains(&width) || !(MIN_EDGE..=MAX_EDGE).contains(&height) || !width.is_multiple_of(16) || !height.is_multiple_of(16)
    {
        candle_core::bail!("Klein image dimensions must be 64..=1024 and divisible by 16");
    }
    let rows = height / 16;
    let cols = width / 16;
    let count = rows * cols;
    if count > MAX_IMAGE_TOKENS {
        candle_core::bail!("too many Klein image tokens");
    }
    Ok((rows, cols, count))
}

/// Diffusers' empirical schedule shift for the number of *generated* image
/// tokens. The current 1024-pixel bound caps a single image at 4096 tokens.
pub fn empirical_mu(image_seq_len: usize, num_steps: usize) -> Result<f64> {
    if image_seq_len == 0 || image_seq_len > MAX_IMAGE_TOKENS || !(1..=MAX_STEPS).contains(&num_steps) {
        candle_core::bail!("invalid Klein schedule dimensions or step count");
    }
    let length = image_seq_len as f64;
    let m_200 = 0.00016927 * length + 0.45666666;
    let m_10 = 8.73809524e-05 * length + 1.89833333;
    let slope = (m_200 - m_10) / 190.0;
    Ok(slope * num_steps as f64 + m_200 - 200.0 * slope)
}

/// Active shifted sigmas, including the terminal zero. `sigmas[i]` and
/// `sigmas[i+1]` drive one transformer call and Euler update. Strength follows
/// Diffusers' `get_timesteps`: it skips `floor(steps * (1-strength))` entries.
#[derive(Clone, Debug)]
pub struct KleinSchedule {
    pub sigmas: Vec<f32>,
    pub mu: f64,
    pub begin_index: usize,
}

pub fn schedule(width: usize, height: usize, num_steps: usize, strength: f64) -> Result<KleinSchedule> {
    let (_, _, tokens) = image_grid(width, height)?;
    if !(1..=MAX_STEPS).contains(&num_steps) || !strength.is_finite() || !(0.0..=1.0).contains(&strength) || strength == 0.0 {
        candle_core::bail!("Klein steps must be 1..=100 and strength must be in (0,1]");
    }
    let mu = empirical_mu(tokens, num_steps)?;
    let shift = mu.exp();
    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for index in 0..num_steps {
        // np.linspace(1, 1/steps, steps).astype(np.float32), followed by
        // FlowMatchEuler's exponential time shift and float32 tensor cast.
        let base = (1.0 - index as f64 / num_steps as f64) as f32;
        let shifted = shift / (shift + (1.0 / f64::from(base) - 1.0));
        sigmas.push(shifted as f32);
    }
    sigmas.push(0.0);
    let begin_index = ((num_steps as f64 * (1.0 - strength)).max(0.0) as usize).min(num_steps - 1);
    Ok(KleinSchedule { sigmas: sigmas[begin_index..].to_vec(), mu, begin_index })
}

/// `[1,128,H/16,W/16]` to `[1,(H/16)*(W/16),128]` in row-major order.
pub fn pack_latents(latents: &Tensor) -> Result<Tensor> {
    let (batch, channels, rows, cols) = latents.dims4()?;
    let width = cols.checked_mul(16).ok_or_else(|| candle_core::Error::Msg("Klein latent width overflow".into()))?;
    let height = rows.checked_mul(16).ok_or_else(|| candle_core::Error::Msg("Klein latent height overflow".into()))?;
    image_grid(width, height)?;
    if batch != 1 || channels != PACKED_CHANNELS {
        candle_core::bail!("Klein latents must have shape [1,128,H/16,W/16]");
    }
    latents.flatten_from(2)?.transpose(1, 2)
}

/// Reverse `pack_latents` for a known output image size.
pub fn unpack_latents(tokens: &Tensor, width: usize, height: usize) -> Result<Tensor> {
    let (rows, cols, count) = image_grid(width, height)?;
    if tokens.dims3()? != (1, count, PACKED_CHANNELS) {
        candle_core::bail!("Klein packed latent shape does not match image dimensions");
    }
    tokens.transpose(1, 2)?.contiguous()?.reshape((1, PACKED_CHANNELS, rows, cols))
}

fn make_image_ids(width: usize, height: usize, time: i64, device: &Device) -> Result<Tensor> {
    let (rows, cols, count) = image_grid(width, height)?;
    let mut ids = Vec::with_capacity(count * 4);
    for row in 0..rows {
        for col in 0..cols {
            ids.extend_from_slice(&[time, row as i64, col as i64, 0]);
        }
    }
    Tensor::from_vec(ids, (1, count, 4), device)
}

/// Generated image IDs `[0,h,w,0]`, then condition image IDs `[10,h,w,0]`.
pub fn combined_image_ids(width: usize, height: usize, device: &Device) -> Result<Tensor> {
    Tensor::cat(&[make_image_ids(width, height, 0, device)?, make_image_ids(width, height, 10, device)?], 1)
}

/// Qwen text IDs `[0,0,0,l]` for each token, capped at the pinned 512-token context.
pub fn text_ids(length: usize, device: &Device) -> Result<Tensor> {
    if length == 0 || length > MAX_TEXT_TOKENS {
        candle_core::bail!("Klein text token count must be 1..=512");
    }
    let mut ids = Vec::with_capacity(length * 4);
    for position in 0..length {
        ids.extend_from_slice(&[0_i64, 0, 0, position as i64]);
    }
    Tensor::from_vec(ids, (1, length, 4), device)
}

/// Source-resolution grayscale mask: 255 means repaint; 0 means preserve.
/// Binarizes at 128 (Diffusers' >=0.5 rule), then bilinearly resizes with
/// `align_corners=false` to the packed latent grid, returning `[1,N,1]`.
pub fn packed_mask(mask: &[u8], width: usize, height: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let (rows, cols, _) = image_grid(width, height)?;
    if mask.len() != width * height || !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        candle_core::bail!("Klein mask length or output dtype is invalid");
    }
    let binary: Vec<f32> = mask.iter().map(|value| f32::from(*value >= 128)).collect();
    Tensor::from_vec(binary, (1, 1, height, width), device)?.upsample_bilinear2d(rows, cols, false)?.flatten_from(2)?.transpose(1, 2)?.to_dtype(dtype)
}

fn checked_latents(tensor: &Tensor) -> Result<usize> {
    let (batch, tokens, channels) = tensor.dims3()?;
    if batch != 1
        || tokens == 0
        || tokens > MAX_IMAGE_TOKENS
        || channels != PACKED_CHANNELS
        || !matches!(tensor.dtype(), DType::F32 | DType::F16 | DType::BF16)
    {
        candle_core::bail!("invalid Klein packed latents");
    }
    Ok(tokens)
}

fn same_latents(a: &Tensor, b: &Tensor) -> Result<()> {
    if a.shape() != b.shape() || a.dtype() != b.dtype() || a.device().location() != b.device().location() {
        candle_core::bail!("Klein latents, noise and predictions must share shape, dtype and device");
    }
    Ok(())
}

fn valid_sigma(sigma: f32) -> bool {
    sigma.is_finite() && (0.0..=1.0).contains(&sigma)
}

/// Diffusers' flow-matching noise scale `(1-sigma)*clean + sigma*noise`.
pub fn scale_noise(clean: &Tensor, initial_noise: &Tensor, sigma: f32) -> Result<Tensor> {
    checked_latents(clean)?;
    same_latents(clean, initial_noise)?;
    if !valid_sigma(sigma) {
        candle_core::bail!("invalid Klein noise sigma");
    }
    // FlowMatchEuler casts its sigma tensor to the source dtype first. BF16
    // multiplication and addition therefore round at each operation.
    let sigma = Tensor::new(sigma, clean.device())?.to_dtype(clean.dtype())?.reshape((1, 1, 1))?;
    let noisy = sigma.broadcast_mul(initial_noise)?;
    let preserved = (&sigma.neg()? + 1.0)?.broadcast_mul(clean)?;
    noisy.broadcast_add(&preserved)
}

/// One FlowMatchEuler step, followed by the inpaint preservation blend.
/// `mask` has `[1,N,1]`; white/1 selects generated pixels. Outside the mask,
/// the original source latents receive the *same* initial noise at `next_sigma`.
/// The terminal sigma zero restores the exact clean source latents.
pub fn euler_inpaint_step(
    latents: &Tensor,
    model_output: &Tensor,
    clean: &Tensor,
    initial_noise: &Tensor,
    mask: &Tensor,
    sigma: f32,
    next_sigma: f32,
) -> Result<Tensor> {
    let tokens = checked_latents(latents)?;
    same_latents(latents, model_output)?;
    same_latents(latents, clean)?;
    same_latents(latents, initial_noise)?;
    if mask.dims3()? != (1, tokens, 1)
        || !matches!(mask.dtype(), DType::F32 | DType::F16 | DType::BF16)
        || mask.device().location() != latents.device().location()
        || !valid_sigma(sigma)
        || !valid_sigma(next_sigma)
        || next_sigma > sigma
    {
        candle_core::bail!("invalid Klein inpaint mask or sigma transition");
    }
    let validation_mask = mask.to_dtype(DType::F32)?;
    let minimum = validation_mask.min_all()?.to_scalar::<f32>()?;
    let maximum = validation_mask.max_all()?.to_scalar::<f32>()?;
    if !minimum.is_finite() || !maximum.is_finite() || minimum < 0.0 || maximum > 1.0 {
        candle_core::bail!("Klein inpaint mask must contain finite values in [0,1]");
    }
    let dtype = latents.dtype();
    // Scheduler sample accumulation is F32, but Torch's scalar dt is
    // converted to the BF16 model-output dtype for its multiplication.
    let dt = Tensor::new(next_sigma - sigma, latents.device())?.to_dtype(dtype)?.reshape((1, 1, 1))?;
    let delta = model_output.broadcast_mul(&dt)?;
    let previous = latents.to_dtype(DType::F32)?.broadcast_add(&delta.to_dtype(DType::F32)?)?.to_dtype(dtype)?;
    let source = scale_noise(clean, initial_noise, next_sigma)?;
    let mask = mask.to_dtype(dtype)?;
    let preserved = source.broadcast_mul(&(&mask.neg()? + 1.0)?)?;
    preserved.broadcast_add(&previous.broadcast_mul(&mask)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip_and_ids_are_row_major() -> Result<()> {
        let device = Device::Cpu;
        let values: Vec<f32> = (0..(128 * 4 * 4)).map(|i| i as f32).collect();
        let original = Tensor::from_vec(values, (1, 128, 4, 4), &device)?;
        let packed = pack_latents(&original)?;
        assert_eq!(packed.dims(), &[1, 16, 128]);
        assert_eq!(packed.narrow(2, 0, 1)?.flatten_all()?.to_vec1::<f32>()?[0..4], [0.0, 1.0, 2.0, 3.0]);
        let error = unpack_latents(&packed, 64, 64)?.broadcast_sub(&original)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(error, 0.0);
        let ids = combined_image_ids(64, 64, &device)?.to_vec3::<i64>()?;
        assert_eq!(ids[0][0], [0, 0, 0, 0]);
        assert_eq!(ids[0][15], [0, 3, 3, 0]);
        assert_eq!(ids[0][16], [10, 0, 0, 0]);
        assert_eq!(ids[0][31], [10, 3, 3, 0]);
        assert_eq!(text_ids(3, &device)?.to_vec3::<i64>()?[0], [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]);
        Ok(())
    }

    #[test]
    fn binary_mask_resamples_at_half_pixel_centres() -> Result<()> {
        let device = Device::Cpu;
        let mut mask = vec![0u8; 64 * 64];
        for value in &mut mask[..8 * 64] {
            *value = 255;
        }
        let packed = packed_mask(&mask, 64, 64, DType::F32, &device)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(&packed[..4], &[0.5; 4]);
        assert!(packed[4..].iter().all(|value| *value == 0.0));
        Ok(())
    }

    #[test]
    fn strength_truncates_shifted_schedule_but_keeps_terminal_zero() -> Result<()> {
        let full = schedule(1024, 1024, 4, 1.0)?;
        let reference = [1.0, 0.967_384_04, 0.908_143_94, 0.767_199_93, 0.0];
        assert!((full.mu - 2.291_179_894_115_571).abs() < 1e-12);
        for (actual, expected) in full.sigmas.iter().zip(reference) {
            assert!((*actual - expected).abs() < 1e-7, "Diffusers sigma mismatch: {actual} vs {expected}");
        }
        assert_eq!(full.begin_index, 0);
        assert_eq!(full.sigmas.len(), 5);
        assert_eq!(full.sigmas[0], 1.0);
        assert_eq!(full.sigmas[4], 0.0);
        assert!(full.sigmas.windows(2).all(|pair| pair[0] > pair[1]));
        let half = schedule(1024, 1024, 4, 0.5)?;
        assert_eq!(half.begin_index, 2);
        assert_eq!(half.sigmas, full.sigmas[2..]);
        assert_eq!(schedule(1024, 1024, 4, 0.01)?.sigmas.len(), 2);
        Ok(())
    }

    #[test]
    fn euler_blend_renoisifies_only_preserved_tokens() -> Result<()> {
        let device = Device::Cpu;
        let clean = Tensor::full(2f32, (1, 16, 128), &device)?;
        let noise = Tensor::full(4f32, (1, 16, 128), &device)?;
        let prediction = Tensor::full(1f32, (1, 16, 128), &device)?;
        let mask = Tensor::from_vec([vec![0f32; 8], vec![1f32; 8]].concat(), (1, 16, 1), &device)?;
        let first = euler_inpaint_step(&noise, &prediction, &clean, &noise, &mask, 1.0, 0.5)?;
        let first_values = first.narrow(2, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(&first_values[..8], &[3.0; 8]);
        assert_eq!(&first_values[8..], &[3.5; 8]);
        let last = euler_inpaint_step(&first, &prediction, &clean, &noise, &mask, 0.5, 0.0)?;
        let last_values = last.narrow(2, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(&last_values[..8], &[2.0; 8]);
        assert_eq!(&last_values[8..], &[3.0; 8]);
        Ok(())
    }

    #[test]
    fn rejects_oversized_and_mismatched_inputs() -> Result<()> {
        let device = Device::Cpu;
        assert!(schedule(1025, 1024, 4, 1.0).is_err());
        assert!(schedule(64, 64, 0, 1.0).is_err());
        assert!(schedule(64, 64, 4, 0.0).is_err());
        assert!(text_ids(513, &device).is_err());
        assert!(packed_mask(&[255], 64, 64, DType::F32, &device).is_err());
        assert!(unpack_latents(&Tensor::zeros((1, 2, 128), DType::F32, &device)?, 64, 64).is_err());
        let latents = Tensor::zeros((1, 16, 128), DType::F32, &device)?;
        let bad_mask = Tensor::full(1.1f32, (1, 16, 1), &device)?;
        assert!(euler_inpaint_step(&latents, &latents, &latents, &latents, &bad_mask, 1.0, 0.5).is_err());
        Ok(())
    }

    /// Run with `KLEIN_FIXTURE` set to the captured NASA first-call tensors.
    #[test]
    #[ignore = "requires external captured Diffusers first-call fixture"]
    fn captured_first_call_ids_match_diffusers() -> Result<()> {
        use crate::weights::Weights;
        use std::path::Path;

        let path = std::env::var("KLEIN_FIXTURE").map_err(|_| candle_core::Error::Msg("set KLEIN_FIXTURE for Klein ID parity".into()))?;
        let fixture = Weights::open(Path::new(&path)).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let device = Device::Cpu;
        let actual_image = combined_image_ids(1024, 1024, &device)?.to_vec3::<i64>()?;
        let expected_image = fixture.read("img_ids", DType::I64, &device)?.to_vec3::<i64>()?;
        assert!(actual_image == expected_image, "Klein image position IDs differ from Diffusers");
        let actual_text = text_ids(512, &device)?.to_vec3::<i64>()?;
        let expected_text = fixture.read("txt_ids", DType::I64, &device)?.to_vec3::<i64>()?;
        assert!(actual_text == expected_text, "Klein text position IDs differ from Diffusers");
        let first_timestep = fixture.read("timestep", DType::F32, &device)?.to_vec1::<f32>()?[0];
        assert!((schedule(1024, 1024, 4, 1.0)?.sigmas[0] - first_timestep).abs() < 1e-6);
        Ok(())
    }

    /// Run with `KLEIN_SAMPLING_FIXTURE` pointing to the small Diffusers
    /// scheduler/mask export; no model weights or GPU are needed.
    #[test]
    #[ignore = "requires external captured Diffusers sampling fixture"]
    fn captured_masks_match_diffusers_bilinear() -> Result<()> {
        let path =
            std::env::var("KLEIN_SAMPLING_FIXTURE").map_err(|_| candle_core::Error::Msg("set KLEIN_SAMPLING_FIXTURE for mask parity".into()))?;
        let device = Device::Cpu;
        let fixture = candle_core::safetensors::load(path, &device)?;
        for (input_name, expected_name, width, height) in
            [("nasa_mask_input_u8", "mask_packed", 1024, 1024), ("small_mask_u8", "small_mask_packed", 96, 64)]
        {
            let source = fixture.get(input_name).ok_or_else(|| candle_core::Error::Msg(format!("missing {input_name}")))?;
            let expected = fixture.get(expected_name).ok_or_else(|| candle_core::Error::Msg(format!("missing {expected_name}")))?;
            let pixels = source.flatten_all()?.to_vec1::<u8>()?;
            let actual = packed_mask(&pixels, width, height, DType::F32, &device)?;
            let error = actual.broadcast_sub(expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(error < 1e-6, "{input_name} mask differs from Diffusers by {error}");
        }
        Ok(())
    }

    /// Run with `KLEIN_SAMPLING_DIR` pointing to the small CPU-only Diffusers
    /// exports. Repeats the tiny 4×8 synthetic tensors across independent
    /// tokens/channels to exercise the production [1,N,128] input contract.
    #[test]
    #[ignore = "requires external captured Diffusers BF16 step fixture"]
    fn captured_bf16_step_matches_diffusers() -> Result<()> {
        use std::path::Path;

        let dir = std::env::var("KLEIN_SAMPLING_DIR").map_err(|_| candle_core::Error::Msg("set KLEIN_SAMPLING_DIR for BF16 step parity".into()))?;
        let dir = Path::new(&dir);
        let device = Device::Cpu;
        let schedule_fixture = candle_core::safetensors::load(dir.join("klein-sampling-f32-goldens.safetensors"), &device)?;
        let reference_sigmas = schedule_fixture.get("sigmas").ok_or_else(|| candle_core::Error::Msg("missing sigmas".into()))?.to_vec1::<f32>()?;
        let computed_sigmas = schedule(1024, 1024, 4, 1.0)?.sigmas;
        assert_eq!(computed_sigmas.len(), reference_sigmas.len());
        for (actual, expected) in computed_sigmas.iter().zip(reference_sigmas) {
            assert!((*actual - expected).abs() < 1e-7, "shifted sigma differs from Diffusers");
        }
        let fixture = candle_core::safetensors::load(dir.join("klein-step-bf16-goldens.safetensors"), &device)?;
        let field = |name: &str| fixture.get(name).ok_or_else(|| candle_core::Error::Msg(format!("missing {name}")));
        let expand = |small: &Tensor| -> Result<Tensor> {
            let (_, _, channels) = small.dims3()?;
            let expanded = Tensor::cat(&vec![small.clone(); 4], 1)?;
            if channels == 1 { Ok(expanded) } else { Tensor::cat(&vec![expanded; 16], 2) }
        };
        let maximum_error = |actual: &Tensor, expected: &Tensor| -> Result<f32> {
            actual.to_dtype(DType::F32)?.broadcast_sub(&expected.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()
        };
        let clean = expand(field("step_sample_bf16")?)?;
        let noise = expand(field("step_noise_bf16")?)?;
        let prediction = expand(field("step_model_output_bf16")?)?;
        let mask = expand(field("step_blend_mask_bf16")?)?;
        let sigma = computed_sigmas[0];
        let next = computed_sigmas[1];
        let initial = scale_noise(&clean, &noise, sigma)?;
        let initial_error = maximum_error(&initial, &expand(field("step_initial_noisy_bf16")?)?)?;
        let next_source_error = maximum_error(&scale_noise(&clean, &noise, next)?, &expand(field("step_next_source_noisy_bf16")?)?)?;
        let dt = Tensor::new(next - sigma, &device)?.to_dtype(DType::BF16)?.reshape((1, 1, 1))?;
        let delta = prediction.broadcast_mul(&dt)?;
        let previous = initial.to_dtype(DType::F32)?.broadcast_add(&delta.to_dtype(DType::F32)?)?.to_dtype(DType::BF16)?;
        let previous_error = maximum_error(&previous, &expand(field("step_scheduler_output_bf16")?)?)?;
        let source = scale_noise(&clean, &noise, next)?;
        let source_product = source.broadcast_mul(&(&mask.neg()? + 1.0)?)?;
        let generated_product = previous.broadcast_mul(&mask)?;
        let source_product_error = maximum_error(&source_product, &expand(field("step_blend_source_product_bf16")?)?)?;
        let generated_product_error = maximum_error(&generated_product, &expand(field("step_blend_generated_product_bf16")?)?)?;
        let blended = euler_inpaint_step(&initial, &prediction, &clean, &noise, &mask, sigma, next)?;
        let blend_error = maximum_error(&blended, &expand(field("step_blended_output_bf16")?)?)?;
        eprintln!(
            "Klein BF16 scale_noise initial error={initial_error}, next source error={next_source_error}, Euler error={previous_error}, source product={source_product_error}, generated product={generated_product_error}, blend error={blend_error}"
        );
        assert_eq!(initial_error, 0.0);
        assert_eq!(next_source_error, 0.0);
        assert_eq!(previous_error, 0.0);
        assert_eq!(source_product_error, 0.0);
        assert_eq!(generated_product_error, 0.0);
        assert_eq!(blend_error, 0.0);
        Ok(())
    }
}
