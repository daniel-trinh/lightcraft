//! FLUX.2 Klein image autoencoder, matching the public Diffusers VAE layout.
//!
//! The VAE consumes RGB tensors in `[-1, 1]` and uses the posterior mean. Its
//! diffusion latents have 128 channels: the 32 VAE channels are patchified 2x2
//! and normalized by the checkpoint's frozen batch-normalization statistics.
//! The reverse path unnormalizes, unpatchifies, and decodes to RGB `[-1, 1]`.
//! All weights and operations remain on the caller's Candle device.
//! The current Candle CPU convolution diverges from PyTorch for a 128-channel
//! 3x3 kernel at 128-pixel spatial resolution. CPU calls above 64 pixels fail
//! explicitly until that backend has a verified fix; CUDA parity is tested.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, GroupNorm, VarBuilder};
use candle_transformers::models::stable_diffusion::unet_2d_blocks::{
    DownEncoderBlock2D, DownEncoderBlock2DConfig, UNetMidBlock2D, UNetMidBlock2DConfig, UpDecoderBlock2D, UpDecoderBlock2DConfig,
};

const PATCH: usize = 2;
const LATENT_CHANNELS: usize = 32;
const PACKED_CHANNELS: usize = LATENT_CHANNELS * PATCH * PATCH;
const MAX_IMAGE_SIDE: usize = 1024;
const MAX_CPU_IMAGE_SIDE: usize = 64;

fn check_backend_resolution(device: &Device, height: usize, width: usize) -> Result<()> {
    if device.is_cpu() && (height > MAX_CPU_IMAGE_SIDE || width > MAX_CPU_IMAGE_SIDE) {
        candle_core::bail!(
            "FLUX.2 Klein VAE images above 64 pixels require NVIDIA CUDA: the current CPU 3x3 convolution does not match the reference at this size"
        )
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct KleinVaeConfig {
    pub block_out_channels: [usize; 4],
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub batch_norm_eps: f64,
}

impl Default for KleinVaeConfig {
    fn default() -> Self {
        Self { block_out_channels: [128, 256, 512, 512], layers_per_block: 2, norm_num_groups: 32, batch_norm_eps: 1e-4 }
    }
}

impl KleinVaeConfig {
    fn validate(&self) -> Result<()> {
        if self.block_out_channels != [128, 256, 512, 512] || self.layers_per_block != 2 || self.norm_num_groups != 32 {
            candle_core::bail!("unsupported FLUX.2 Klein VAE architecture")
        }
        if !self.batch_norm_eps.is_finite() || self.batch_norm_eps <= 0.0 || self.batch_norm_eps > 1.0 {
            candle_core::bail!("invalid FLUX.2 Klein VAE batch-normalization epsilon")
        }
        Ok(())
    }
}

struct Encoder {
    conv_in: Conv2d,
    down_blocks: Vec<DownEncoderBlock2D>,
    mid_block: UNetMidBlock2D,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
    quant_conv: Conv2d,
}

impl Encoder {
    fn new(config: &KleinVaeConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let e = vb.pp("encoder");
        let p3 = Conv2dConfig { padding: 1, ..Default::default() };
        let conv_in = candle_nn::conv2d(3, config.block_out_channels[0], 3, p3, e.pp("conv_in"))?;
        let mut down_blocks = Vec::with_capacity(config.block_out_channels.len());
        for (index, &output) in config.block_out_channels.iter().enumerate() {
            let input = if index == 0 { output } else { config.block_out_channels[index - 1] };
            let block = DownEncoderBlock2D::new(
                e.pp("down_blocks").pp(index.to_string()),
                input,
                output,
                DownEncoderBlock2DConfig {
                    num_layers: config.layers_per_block,
                    resnet_eps: 1e-6,
                    resnet_groups: config.norm_num_groups,
                    add_downsample: index + 1 != config.block_out_channels.len(),
                    downsample_padding: 0,
                    ..Default::default()
                },
            )?;
            down_blocks.push(block);
        }
        let deepest = config.block_out_channels[3];
        let mid_block = UNetMidBlock2D::new(
            e.pp("mid_block"),
            deepest,
            None,
            UNetMidBlock2DConfig {
                resnet_eps: 1e-6,
                resnet_groups: Some(config.norm_num_groups),
                attn_num_head_channels: None,
                ..Default::default()
            },
        )?;
        let conv_norm_out = candle_nn::group_norm(config.norm_num_groups, deepest, 1e-6, e.pp("conv_norm_out"))?;
        let conv_out = candle_nn::conv2d(deepest, LATENT_CHANNELS * 2, 3, p3, e.pp("conv_out"))?;
        let quant_conv = candle_nn::conv2d(LATENT_CHANNELS * 2, LATENT_CHANNELS * 2, 1, Conv2dConfig::default(), vb.pp("quant_conv"))?;
        Ok(Self { conv_in, down_blocks, mid_block, conv_norm_out, conv_out, quant_conv })
    }

    fn mean(&self, rgb: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(rgb)?;
        for block in &self.down_blocks {
            x = block.forward(&x)?;
        }
        x = self.mid_block.forward(&x, None)?;
        x = candle_nn::ops::silu(&self.conv_norm_out.forward(&x)?)?;
        x = self.quant_conv.forward(&self.conv_out.forward(&x)?)?;
        x.narrow(1, 0, LATENT_CHANNELS)
    }
}

struct Decoder {
    post_quant_conv: Conv2d,
    conv_in: Conv2d,
    mid_block: UNetMidBlock2D,
    up_blocks: Vec<UpDecoderBlock2D>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Decoder {
    fn new(config: &KleinVaeConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let d = vb.pp("decoder");
        let deepest = config.block_out_channels[3];
        let p3 = Conv2dConfig { padding: 1, ..Default::default() };
        let post_quant_conv = candle_nn::conv2d(LATENT_CHANNELS, LATENT_CHANNELS, 1, Conv2dConfig::default(), vb.pp("post_quant_conv"))?;
        let conv_in = candle_nn::conv2d(LATENT_CHANNELS, deepest, 3, p3, d.pp("conv_in"))?;
        let mid_block = UNetMidBlock2D::new(
            d.pp("mid_block"),
            deepest,
            None,
            UNetMidBlock2DConfig {
                resnet_eps: 1e-6,
                resnet_groups: Some(config.norm_num_groups),
                attn_num_head_channels: None,
                ..Default::default()
            },
        )?;
        let reversed = [config.block_out_channels[3], config.block_out_channels[2], config.block_out_channels[1], config.block_out_channels[0]];
        let mut up_blocks = Vec::with_capacity(reversed.len());
        for (index, &output) in reversed.iter().enumerate() {
            let input = if index == 0 { output } else { reversed[index - 1] };
            let block = UpDecoderBlock2D::new(
                d.pp("up_blocks").pp(index.to_string()),
                input,
                output,
                UpDecoderBlock2DConfig {
                    num_layers: config.layers_per_block + 1,
                    resnet_eps: 1e-6,
                    resnet_groups: config.norm_num_groups,
                    add_upsample: index + 1 != reversed.len(),
                    ..Default::default()
                },
            )?;
            up_blocks.push(block);
        }
        let conv_norm_out = candle_nn::group_norm(config.norm_num_groups, config.block_out_channels[0], 1e-6, d.pp("conv_norm_out"))?;
        let conv_out = candle_nn::conv2d(config.block_out_channels[0], 3, 3, p3, d.pp("conv_out"))?;
        Ok(Self { post_quant_conv, conv_in, mid_block, up_blocks, conv_norm_out, conv_out })
    }

    fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        let z = self.post_quant_conv.forward(latent)?;
        let mut x = self.conv_in.forward(&z)?;
        x = self.mid_block.forward(&x, None)?;
        for block in &self.up_blocks {
            x = block.forward(&x)?;
        }
        x = candle_nn::ops::silu(&self.conv_norm_out.forward(&x)?)?;
        self.conv_out.forward(&x)
    }
}

/// The public FLUX.2 Klein VAE in the checkpoint's Diffusers key layout.
pub struct KleinVae {
    encoder: Encoder,
    decoder: Decoder,
    bn_mean: Tensor,
    bn_std: Tensor,
}

impl KleinVae {
    pub fn new(config: KleinVaeConfig, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        if vb.dtype() != DType::F32 {
            candle_core::bail!("FLUX.2 Klein VAE requires F32 for its force-upcast configuration")
        }
        let encoder = Encoder::new(&config, vb.clone())?;
        let decoder = Decoder::new(&config, vb.clone())?;
        let mean = vb.pp("bn").get((PACKED_CHANNELS,), "running_mean")?.reshape((1, PACKED_CHANNELS, 1, 1))?;
        let variance = vb.pp("bn").get((PACKED_CHANNELS,), "running_var")?;
        let std = variance.affine(1.0, config.batch_norm_eps)?.sqrt()?.reshape((1, PACKED_CHANNELS, 1, 1))?;
        Ok(Self { encoder, decoder, bn_mean: mean, bn_std: std })
    }

    /// Deterministic posterior mean `[1,32,H/8,W/8]` before patching and BN.
    pub fn posterior_mean(&self, rgb: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = rgb.dims4()?;
        if channels != 3 || height < 32 || width < 32 || height > MAX_IMAGE_SIDE || width > MAX_IMAGE_SIDE || height % 16 != 0 || width % 16 != 0 {
            candle_core::bail!("FLUX.2 Klein VAE RGB input must be 3 channels, 32..1024 pixels per side, and divisible by 16")
        }
        if batch != 1 || rgb.dtype() != DType::F32 {
            candle_core::bail!("FLUX.2 Klein VAE expects one F32 image per encode call")
        }
        check_backend_resolution(rgb.device(), height, width)?;
        self.encoder.mean(rgb)
    }

    /// RGB `[B,3,H,W]` in `[-1,1]` -> packed normalized `[B,128,H/16,W/16]`.
    pub fn encode(&self, rgb: &Tensor) -> Result<Tensor> {
        let packed = patchify(&self.posterior_mean(rgb)?)?;
        packed.broadcast_sub(&self.bn_mean)?.broadcast_div(&self.bn_std)
    }

    /// Packed normalized `[B,128,h,w]` -> RGB `[B,3,16h,16w]` in `[-1,1]`.
    pub fn decode(&self, packed: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = packed.dims4()?;
        if batch != 1 || channels != PACKED_CHANNELS || height < 2 || width < 2 || height > MAX_IMAGE_SIDE / 16 || width > MAX_IMAGE_SIDE / 16 {
            candle_core::bail!("FLUX.2 Klein VAE packed latent must be [1,128,h,w] with h,w in 2..64")
        }
        if packed.dtype() != DType::F32 {
            candle_core::bail!("FLUX.2 Klein VAE expects F32 packed latents")
        }
        check_backend_resolution(packed.device(), height * 16, width * 16)?;
        let unnormalized = packed.broadcast_mul(&self.bn_std)?.broadcast_add(&self.bn_mean)?;
        self.decoder.forward(&unpatchify(&unnormalized)?)
    }
}

/// `[B,32,2h,2w] -> [B,128,h,w]`, channel order `(channel,row-in-patch,column-in-patch)`.
fn patchify(latent: &Tensor) -> Result<Tensor> {
    let (batch, channels, height, width) = latent.dims4()?;
    if channels != LATENT_CHANNELS || height == 0 || width == 0 || height % PATCH != 0 || width % PATCH != 0 {
        candle_core::bail!("FLUX.2 Klein VAE latent needs 32 channels and even spatial dimensions")
    }
    latent.reshape((batch, channels, height / PATCH, PATCH, width / PATCH, PATCH))?.permute((0, 1, 3, 5, 2, 4))?.reshape((
        batch,
        PACKED_CHANNELS,
        height / PATCH,
        width / PATCH,
    ))
}

/// `[B,128,h,w] -> [B,32,2h,2w]`, exact inverse of `patchify`.
fn unpatchify(packed: &Tensor) -> Result<Tensor> {
    let (batch, channels, height, width) = packed.dims4()?;
    if channels != PACKED_CHANNELS || height == 0 || width == 0 {
        candle_core::bail!("FLUX.2 Klein VAE packed latent needs 128 channels and positive spatial dimensions")
    }
    packed.reshape((batch, LATENT_CHANNELS, PATCH, PATCH, height, width))?.permute((0, 1, 4, 2, 5, 3))?.reshape((
        batch,
        LATENT_CHANNELS,
        height * PATCH,
        width * PATCH,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::path::PathBuf;

    #[test]
    fn patchify_preserves_channel_and_spatial_order() -> Result<()> {
        let input = Tensor::arange(0f32, (LATENT_CHANNELS * 4 * 6) as f32, &Device::Cpu)?.reshape((1, LATENT_CHANNELS, 4, 6))?;
        let packed = patchify(&input)?;
        assert_eq!(packed.dims(), &[1, PACKED_CHANNELS, 2, 3]);
        assert_eq!(packed.flatten_all()?.to_vec1::<f32>()?[0..6], [0., 2., 4., 12., 14., 16.]);
        assert_eq!(unpatchify(&packed)?.flatten_all()?.to_vec1::<f32>()?, input.flatten_all()?.to_vec1::<f32>()?);
        Ok(())
    }

    #[test]
    fn invalid_patch_shape_is_rejected() -> Result<()> {
        let odd = Tensor::zeros((1, LATENT_CHANNELS, 3, 4), DType::F32, &Device::Cpu)?;
        assert!(patchify(&odd).is_err());
        let wrong = Tensor::zeros((1, 32, 2, 2), DType::F32, &Device::Cpu)?;
        assert!(unpatchify(&wrong).is_err());
        Ok(())
    }

    #[test]
    fn cpu_resolution_guard_prevents_known_wrong_convolution() {
        assert!(check_backend_resolution(&Device::Cpu, 64, 64).is_ok());
        assert!(check_backend_resolution(&Device::Cpu, 128, 64).is_err());
        assert!(check_backend_resolution(&Device::Cpu, 64, 128).is_err());
    }

    #[test]
    #[ignore = "requires separately generated public checkpoint parity fixtures"]
    fn reference_intermediate_stages() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let checkpoint = PathBuf::from(std::env::var_os("LIGHTCRAFT_KLEIN_VAE_CHECKPOINT").ok_or("missing checkpoint path")?);
        let fixture = PathBuf::from(std::env::var_os("LIGHTCRAFT_KLEIN_VAE_FIXTURE").ok_or("missing fixture path")?);
        let intermediates = PathBuf::from(std::env::var_os("LIGHTCRAFT_KLEIN_VAE_INTERMEDIATES").ok_or("missing intermediate path")?);
        let device = Device::Cpu;
        let model_bytes = std::fs::read(checkpoint)?;
        let vb = VarBuilder::from_buffered_safetensors(model_bytes, DType::F32, &device)?;
        let vae = KleinVae::new(KleinVaeConfig::default(), vb.clone())?;
        let fv = VarBuilder::from_buffered_safetensors(std::fs::read(fixture)?, DType::F32, &device)?;
        let trace = VarBuilder::from_buffered_safetensors(std::fs::read(intermediates)?, DType::F32, &device)?;
        let nested_path = PathBuf::from(std::env::var_os("LIGHTCRAFT_KLEIN_VAE_RESNET_TRACE").ok_or("missing nested trace path")?);
        let nested = VarBuilder::from_buffered_safetensors(std::fs::read(nested_path)?, DType::F32, &device)?;
        let compare = |name: &str, tensor: &Tensor| -> std::result::Result<(), Box<dyn std::error::Error>> {
            let reference = trace.get_unchecked(name)?;
            let delta = (tensor - &reference)?.abs()?;
            eprintln!("{name}: MAE={} max={}", delta.mean_all()?.to_scalar::<f32>()?, delta.max_all()?.to_scalar::<f32>()?);
            Ok(())
        };
        let encoder = &vae.encoder;
        let mut x = encoder.conv_in.forward(&fv.get_unchecked("input_nchw_minus1_plus1_f32")?)?;
        compare("encode_encoder_conv_in_out", &x)?;
        let e0 = vb.pp("encoder.down_blocks.0.resnets.0");
        let en1 = candle_nn::group_norm(32, 128, 1e-6, e0.pp("norm1"))?.forward(&x)?;
        let en1_delta = (&en1 - &nested.get_unchecked("encode_encoder_down_blocks_0_resnets_0_norm1")?)?.abs()?;
        eprintln!("encode first norm1: MAE={} max={}", en1_delta.mean_all()?.to_scalar::<f32>()?, en1_delta.max_all()?.to_scalar::<f32>()?);
        let ec1 = candle_nn::conv2d(128, 128, 3, Conv2dConfig { padding: 1, ..Default::default() }, e0.pp("conv1"))?
            .forward(&candle_nn::ops::silu(&en1)?)?;
        let ec1_delta = (&ec1 - &nested.get_unchecked("encode_encoder_down_blocks_0_resnets_0_conv1")?)?.abs()?;
        eprintln!("encode first conv1: MAE={} max={}", ec1_delta.mean_all()?.to_scalar::<f32>()?, ec1_delta.max_all()?.to_scalar::<f32>()?);
        for (index, block) in encoder.down_blocks.iter().enumerate() {
            x = block.forward(&x)?;
            compare(&format!("encode_encoder_down_blocks_{index}_out"), &x)?;
        }
        x = encoder.mid_block.forward(&x, None)?;
        compare("encode_encoder_mid_block_out", &x)?;
        x = encoder.conv_out.forward(&candle_nn::ops::silu(&encoder.conv_norm_out.forward(&x)?)?)?;
        compare("encode_encoder_conv_out_out", &x)?;
        x = encoder.quant_conv.forward(&x)?;
        compare("encode_quant_conv_out", &x)?;
        let decoder = &vae.decoder;
        let mut y = decoder.post_quant_conv.forward(&fv.get_unchecked("decoder_input_unpacked")?)?;
        compare("decode_post_quant_conv_out", &y)?;
        y = decoder.conv_in.forward(&y)?;
        compare("decode_decoder_conv_in_out", &y)?;
        y = decoder.mid_block.forward(&y, None)?;
        compare("decode_decoder_mid_block_out", &y)?;
        for (index, block) in decoder.up_blocks.iter().enumerate() {
            if index == 3 {
                let d0 = vb.pp("decoder.up_blocks.3.resnets.0");
                let dn1 = candle_nn::group_norm(32, 256, 1e-6, d0.pp("norm1"))?.forward(&y)?;
                let dn1_delta = (&dn1 - &nested.get_unchecked("decode_decoder_up_blocks_3_resnets_0_norm1")?)?.abs()?;
                eprintln!("decode first norm1: MAE={} max={}", dn1_delta.mean_all()?.to_scalar::<f32>()?, dn1_delta.max_all()?.to_scalar::<f32>()?);
                let dc1 = candle_nn::conv2d(256, 128, 3, Conv2dConfig { padding: 1, ..Default::default() }, d0.pp("conv1"))?
                    .forward(&candle_nn::ops::silu(&dn1)?)?;
                let dc1_delta = (&dc1 - &nested.get_unchecked("decode_decoder_up_blocks_3_resnets_0_conv1")?)?.abs()?;
                eprintln!("decode first conv1: MAE={} max={}", dc1_delta.mean_all()?.to_scalar::<f32>()?, dc1_delta.max_all()?.to_scalar::<f32>()?);
            }
            y = block.forward(&y)?;
            compare(&format!("decode_decoder_up_blocks_{index}_out"), &y)?;
        }
        y = decoder.conv_out.forward(&candle_nn::ops::silu(&decoder.conv_norm_out.forward(&y)?)?)?;
        compare("decode_decoder_conv_out_out", &y)?;
        Ok(())
    }
}
