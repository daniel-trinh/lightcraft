//! MI-GAN 512 inference on Candle. Architecture follows the MIT-licensed
//! Picsart-AI-Research/MI-GAN `lib/model_zoo/migan_inference.py` at
//! revision 2b793c5ece43f4253e32d4afc257120a5deed6f5.
//! The network weights are separately licensed under that project's
//! `LICENSE-WEIGHTS`; neither weights nor ONNX Runtime are bundled here.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module};

use crate::weights::Weights;
use crate::{Error, Result};

pub const SIZE: usize = 512;
pub const AUTHOR_ONNX_SHA256: &str = "593eba0b7e04730f1b61c0a3cbca68d97d8d6a7ff5c6a44a7b9d7fcd880fc5ae";
pub const CONVERTED_SAFETENSORS_SHA256: &str = "365c3680e7e33b140761d11a0e62d7ed596572d036bbc3e670540c2baaf9c845";
const CHANNEL_BASE: usize = 32768;
const CHANNEL_MAX: usize = 512;

fn channels(resolution: usize) -> usize {
    (CHANNEL_BASE / resolution).min(CHANNEL_MAX)
}

// Names are ONNX graph initializers in the pinned author `migan.onnx` export.
// Each pair is the constant-folded first/second synthesis-conv noise image.
fn folded_noise_key(resolution: usize, second: bool) -> Option<&'static str> {
    match (resolution, second) {
        (8, false) => Some("1433"),
        (8, true) => Some("1438"),
        (16, false) => Some("1455"),
        (16, true) => Some("1460"),
        (32, false) => Some("1477"),
        (32, true) => Some("1482"),
        (64, false) => Some("1499"),
        (64, true) => Some("1504"),
        (128, false) => Some("1521"),
        (128, true) => Some("1526"),
        (256, false) => Some("1543"),
        (256, true) => Some("1548"),
        (512, false) => Some("1565"),
        (512, true) => Some("1570"),
        _ => None,
    }
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) { Err(Error::Cancelled) } else { Ok(()) }
}

struct Reader {
    weights: Arc<Weights>,
    device: Device,
    prefix: &'static str,
}

impl Reader {
    fn name(&self, name: &str) -> String {
        format!("{}{name}", self.prefix)
    }

    fn tensor(&self, name: &str) -> Result<Tensor> {
        Ok(self.weights.read(&self.name(name), DType::F32, &self.device)?)
    }

    fn conv(
        &self,
        name: &str,
        input: usize,
        output: usize,
        kernel: usize,
        groups: usize,
        padding: usize,
        stride: usize,
        bias: bool,
    ) -> Result<Conv2d> {
        let weight = self.tensor(&format!("{name}.weight"))?;
        if weight.dims() != [output, input / groups, kernel, kernel] {
            return Err(Error::Model(format!(
                "MI-GAN {name}.weight has shape {:?}, expected [{output}, {}, {kernel}, {kernel}]",
                weight.dims(),
                input / groups
            )));
        }
        let bias = if bias {
            let value = self.tensor(&format!("{name}.bias"))?;
            if value.dims() != [output] {
                return Err(Error::Model(format!("MI-GAN {name}.bias has incorrect shape")));
            }
            Some(value)
        } else {
            None
        };
        Ok(Conv2d::new(weight, bias, Conv2dConfig { padding, stride, groups, ..Default::default() }))
    }

    fn depthwise(&self, name: &str, channels: usize, kernel: usize, padding: usize, stride: usize, bias: bool) -> Result<DepthwiseConv> {
        let weight = self.tensor(&format!("{name}.weight"))?;
        if weight.dims() != [channels, 1, kernel, kernel] {
            return Err(Error::Model(format!("MI-GAN {name}.weight has incorrect shape")));
        }
        let bias = if bias {
            let value = self.tensor(&format!("{name}.bias"))?;
            if value.dims() != [channels] {
                return Err(Error::Model(format!("MI-GAN {name}.bias has incorrect shape")));
            }
            Some(value)
        } else {
            None
        };
        Ok(DepthwiseConv { weight, bias, padding, stride })
    }
}

/// Candle expands grouped convolution into one kernel launch per channel.
/// On CUDA, turn each bounded group of output rows into image patches and use
/// one batched matrix multiplication over the depthwise filter taps.
struct DepthwiseConv {
    weight: Tensor,
    bias: Option<Tensor>,
    padding: usize,
    stride: usize,
}

impl DepthwiseConv {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        #[cfg(test)]
        let reference = std::env::var("MIGAN_TEST_GROUPED").as_deref() == Ok("1");
        #[cfg(not(test))]
        let reference = false;
        if !matches!(xs.device(), Device::Cuda(_)) || reference {
            let channels = self.weight.dims4()?.0;
            let conv = Conv2d::new(
                self.weight.clone(),
                self.bias.clone(),
                Conv2dConfig { padding: self.padding, stride: self.stride, groups: channels, ..Default::default() },
            );
            return Ok(conv.forward(xs)?);
        }
        self.forward_tiled(xs)
    }

    fn forward_tiled(&self, xs: &Tensor) -> Result<Tensor> {
        const MAX_PATCH_BYTES: usize = 256 * 1024 * 1024;
        const MAX_TILE_ROWS: usize = 128;
        let (batch, channels, height, width) = xs.dims4()?;
        let (weight_channels, one, kernel_h, kernel_w) = self.weight.dims4()?;
        if channels != weight_channels || one != 1 || kernel_h != kernel_w || self.stride == 0 {
            return Err(Error::Model("MI-GAN depthwise convolution shape mismatch".into()));
        }
        let padding = self.padding.checked_mul(2).ok_or_else(|| Error::Model("MI-GAN depthwise padding overflow".into()))?;
        let padded_h = height.checked_add(padding).ok_or_else(|| Error::Model("MI-GAN depthwise height overflow".into()))?;
        let padded_w = width.checked_add(padding).ok_or_else(|| Error::Model("MI-GAN depthwise width overflow".into()))?;
        let out_h = padded_h.checked_sub(kernel_h).ok_or_else(|| Error::Model("MI-GAN depthwise kernel exceeds height".into()))? / self.stride + 1;
        let out_w = padded_w.checked_sub(kernel_w).ok_or_else(|| Error::Model("MI-GAN depthwise kernel exceeds width".into()))? / self.stride + 1;
        let kernel_pixels = kernel_h.checked_mul(kernel_w).ok_or_else(|| Error::Model("MI-GAN depthwise kernel size overflow".into()))?;
        let patch_row_bytes = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(out_w))
            .and_then(|n| n.checked_mul(kernel_pixels))
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::Model("MI-GAN depthwise patch size overflow".into()))?;
        if patch_row_bytes == 0 || patch_row_bytes > MAX_PATCH_BYTES {
            return Err(Error::Model("MI-GAN depthwise patch exceeds the 256 MiB limit".into()));
        }
        let tile_rows = (MAX_PATCH_BYTES / patch_row_bytes).clamp(1, MAX_TILE_ROWS);
        let padded = xs.pad_with_zeros(2, self.padding, self.padding)?.pad_with_zeros(3, self.padding, self.padding)?;
        let kernel = self.weight.reshape((1, channels, kernel_pixels, 1))?;
        let mut tiles = Vec::with_capacity(out_h.div_ceil(tile_rows));
        for row in (0..out_h).step_by(tile_rows) {
            let rows = (out_h - row).min(tile_rows);
            let first = row.checked_mul(self.stride).ok_or_else(|| Error::Model("MI-GAN depthwise row overflow".into()))?;
            let input_rows = (rows - 1)
                .checked_mul(self.stride)
                .and_then(|n| n.checked_add(kernel_h))
                .ok_or_else(|| Error::Model("MI-GAN depthwise tile overflow".into()))?;
            let slice = padded.narrow(2, first, input_rows)?;
            // unfold(2) appends Kh, then unfold(3) appends Kw. Their final
            // flattened order is exactly the checkpoint's [Kh,Kw] order.
            let patch_pixels = rows.checked_mul(out_w).ok_or_else(|| Error::Model("MI-GAN depthwise tile size overflow".into()))?;
            let patches =
                slice.unfold(2, kernel_h, self.stride)?.unfold(3, kernel_w, self.stride)?.reshape((batch, channels, patch_pixels, kernel_pixels))?;
            let tile = patches.broadcast_matmul(&kernel)?.reshape((batch, channels, rows, out_w))?;
            tiles.push(tile);
        }
        let output = Tensor::cat(&tiles, 2)?;
        if let Some(bias) = &self.bias { Ok(output.broadcast_add(&bias.reshape((1, channels, 1, 1))?)?) } else { Ok(output) }
    }
}

fn activate(xs: Tensor) -> Result<Tensor> {
    Ok((candle_nn::ops::leaky_relu(&xs, 0.2)? * std::f64::consts::SQRT_2)?.clamp(-256f64, 256f64)?)
}

struct SeparableConv {
    depth: DepthwiseConv,
    point: Conv2d,
    down: Option<DepthwiseConv>,
    up: Option<DepthwiseConv>,
    noise: Option<Tensor>,
    activation: bool,
}

impl SeparableConv {
    fn load(
        reader: &Reader,
        name: &str,
        input: usize,
        output: usize,
        down: bool,
        up_resolution: Option<usize>,
        noise_resolution: Option<usize>,
        folded_noise_key: Option<&str>,
    ) -> Result<Self> {
        let depth = reader.depthwise(&format!("{name}.conv1"), input, 3, 1, 1, true)?;
        let point = reader.conv(&format!("{name}.conv2"), input, output, 1, 1, 0, 1, false)?;
        let down = if down { Some(reader.depthwise(&format!("{name}.downsample.filter"), input, 4, 1, 2, false)?) } else { None };
        let up = if up_resolution.is_some() { Some(reader.depthwise(&format!("{name}.upsample.filter"), output, 4, 0, 1, false)?) } else { None };
        let noise = match (noise_resolution, folded_noise_key) {
            (Some(resolution), Some(key)) => {
                // The author's pinned ONNX export constant-folds noise_const *
                // noise_strength into numbered graph initializers.
                let fixed = reader.weights.read(key, DType::F32, &reader.device)?;
                if fixed.dims() != [resolution, resolution] {
                    return Err(Error::Model(format!("MI-GAN folded noise {key} has incorrect shape")));
                }
                Some(fixed.reshape((1, 1, resolution, resolution))?)
            }
            (Some(resolution), None) => {
                let fixed = reader.tensor(&format!("{name}.noise_const"))?;
                let strength = reader.tensor(&format!("{name}.noise_strength"))?;
                if fixed.dims() != [resolution, resolution] || strength.elem_count() != 1 {
                    return Err(Error::Model(format!("MI-GAN {name} noise has incorrect shape")));
                }
                Some(fixed.broadcast_mul(&strength.reshape((1, 1))?)?.reshape((1, 1, resolution, resolution))?)
            }
            (None, _) => None,
        };
        Ok(Self { depth, point, down, up, noise, activation: true })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.depth.forward(xs)?;
        if self.activation {
            xs = activate(xs)?;
        }
        if let Some(filter) = &self.down {
            xs = filter.forward(&xs)?;
        }
        xs = self.point.forward(&xs)?;
        if let Some(filter) = &self.up {
            xs = upsample(&xs, filter)?;
        }
        if let Some(noise) = &self.noise {
            xs = xs.broadcast_add(noise)?;
        }
        if self.activation {
            xs = activate(xs)?;
        }
        Ok(xs)
    }
}

/// Author's upsample: nearest 2x, keep even/even sites, asymmetric zero pad,
/// then depthwise 4x4 [1,3,3,1] filtering. The checkerboard is deterministic
/// and need not be stored as a 512x512 safetensors buffer.
fn upsample(xs: &Tensor, filter: &DepthwiseConv) -> Result<Tensor> {
    let (_, _, h, w) = xs.dims4()?;
    let up_h = h.checked_mul(2).ok_or_else(|| Error::Model("MI-GAN height overflow".into()))?;
    let up_w = w.checked_mul(2).ok_or_else(|| Error::Model("MI-GAN width overflow".into()))?;
    let mut xs = xs.upsample_nearest2d(up_h, up_w)?;
    let mut checker = vec![0f32; up_h * up_w];
    for y in (0..up_h).step_by(2) {
        for x in (0..up_w).step_by(2) {
            let index = y * up_w + x;
            *checker.get_mut(index).ok_or_else(|| Error::Model("MI-GAN checkerboard index overflow".into()))? = 1.0;
        }
    }
    let checker = Tensor::from_vec(checker, (1, 1, up_h, up_w), xs.device())?;
    xs = xs.broadcast_mul(&checker)?;
    xs = xs.pad_with_zeros(2, 2, 1)?.pad_with_zeros(3, 2, 1)?;
    filter.forward(&xs)
}

struct EncoderBlock {
    from_rgb: Option<Conv2d>,
    conv1: SeparableConv,
    conv2: SeparableConv,
}

impl EncoderBlock {
    fn load(reader: &Reader, resolution: usize) -> Result<Self> {
        let input = channels(resolution);
        let output = channels((resolution / 2).max(4));
        let name = format!("encoder.b{resolution}");
        let from_rgb = if resolution == SIZE { Some(reader.conv(&format!("{name}.fromrgb"), 4, input, 1, 1, 0, 1, true)?) } else { None };
        Ok(Self {
            from_rgb,
            conv1: SeparableConv::load(reader, &format!("{name}.conv1"), input, input, false, None, None, None)?,
            conv2: SeparableConv::load(reader, &format!("{name}.conv2"), input, output, resolution != 4, None, None, None)?,
        })
    }

    fn forward(&self, previous: Option<&Tensor>, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut xs = if let Some(from_rgb) = &self.from_rgb {
            let initial = activate(from_rgb.forward(input)?)?;
            if let Some(previous) = previous { (previous + initial)? } else { initial }
        } else {
            previous.ok_or_else(|| Error::Model("MI-GAN encoder lost its feature tensor".into()))?.clone()
        };
        let skip = self.conv1.forward(&xs)?;
        xs = self.conv2.forward(&skip)?;
        Ok((xs, skip))
    }
}

struct SynthesisBlock {
    conv1: SeparableConv,
    conv2: SeparableConv,
    to_rgb: Conv2d,
    rgb_up: Option<DepthwiseConv>,
}

impl SynthesisBlock {
    fn load(reader: &Reader, resolution: usize) -> Result<Self> {
        let name = format!("synthesis.b{resolution}");
        let input = channels((resolution / 2).max(4));
        let output = channels(resolution);
        Ok(Self {
            conv1: SeparableConv::load(
                reader,
                &format!("{name}.conv1"),
                input,
                output,
                false,
                (resolution != 4).then_some(resolution),
                (resolution != 4).then_some(resolution),
                folded_noise_key(resolution, false),
            )?,
            conv2: SeparableConv::load(
                reader,
                &format!("{name}.conv2"),
                output,
                output,
                false,
                None,
                (resolution != 4).then_some(resolution),
                folded_noise_key(resolution, true),
            )?,
            to_rgb: reader.conv(&format!("{name}.torgb"), output, 3, 1, 1, 0, 1, true)?,
            rgb_up: if resolution == 4 { None } else { Some(reader.depthwise(&format!("{name}.upsample.filter"), 3, 4, 0, 1, false)?) },
        })
    }

    fn forward(&self, previous: &Tensor, skip: &Tensor, previous_rgb: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let xs = self.conv1.forward(previous)?;
        let xs = self.conv2.forward(&(&xs + skip)?)?;
        let rgb = self.to_rgb.forward(&xs)?;
        let rgb = match (previous_rgb, &self.rgb_up) {
            (Some(previous_rgb), Some(filter)) => (upsample(previous_rgb, filter)? + rgb)?,
            (None, None) => rgb,
            _ => return Err(Error::Model("MI-GAN synthesis RGB skip mismatch".into())),
        };
        Ok((xs, rgb))
    }
}

/// Standalone MI-GAN-512 network. `infer_raw` returns unclipped float RGB in
/// NCHW [-1,1]-approximately; the caller performs final compositing.
pub struct Migan {
    device: Device,
    encoder: Vec<(usize, EncoderBlock)>,
    synthesis: Vec<(usize, SynthesisBlock)>,
}

impl Migan {
    pub fn load_local(path: &Path) -> Result<Self> {
        Self::load(path, crate::model::best_device()?)
    }

    pub fn device_description(&self) -> &'static str {
        match self.device {
            Device::Cpu => "CPU",
            Device::Cuda(_) => "NVIDIA CUDA",
            Device::Metal(_) => "Metal",
        }
    }

    pub fn load(path: &Path, device: Device) -> Result<Self> {
        let weights = Weights::open(path)?;
        let prefix = if weights.contains("model.encoder.b512.fromrgb.weight") {
            "model."
        } else if weights.contains("encoder.b512.fromrgb.weight") {
            ""
        } else {
            return Err(Error::Model("MI-GAN safetensors lacks encoder.b512.fromrgb.weight".into()));
        };
        let reader = Reader { weights, device: device.clone(), prefix };
        let mut encoder = Vec::with_capacity(8);
        let mut resolution = SIZE;
        loop {
            encoder.push((resolution, EncoderBlock::load(&reader, resolution)?));
            if resolution == 4 {
                break;
            }
            resolution /= 2;
        }
        let mut synthesis = Vec::with_capacity(8);
        for resolution in [4, 8, 16, 32, 64, 128, 256, 512] {
            synthesis.push((resolution, SynthesisBlock::load(&reader, resolution)?));
        }
        Ok(Self { device, encoder, synthesis })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `conditioned`: F32 [1,4,512,512] as `[known_mask-.5,
    /// normalized_rgb*known_mask]`, matching the author's inference network.
    pub fn infer_raw(&self, conditioned: &Tensor, cancel: &AtomicBool) -> Result<Tensor> {
        if conditioned.dims() != [1, 4, SIZE, SIZE] || conditioned.dtype() != DType::F32 || !conditioned.device().same_device(&self.device) {
            return Err(Error::Model("MI-GAN expects F32 [1,4,512,512] on its model device".into()));
        }
        check_cancel(cancel)?;
        let mut current: Option<Tensor> = None;
        let mut skips = Vec::with_capacity(self.encoder.len());
        for (resolution, block) in &self.encoder {
            check_cancel(cancel)?;
            let (next, skip) = block.forward(current.as_ref(), conditioned)?;
            skips.push((*resolution, skip));
            current = Some(next);
        }
        let mut current = current.ok_or_else(|| Error::Model("MI-GAN has no encoder blocks".into()))?;
        let mut rgb: Option<Tensor> = None;
        for (resolution, block) in &self.synthesis {
            check_cancel(cancel)?;
            let skip = skips
                .iter()
                .find(|(level, _)| level == resolution)
                .map(|(_, tensor)| tensor)
                .ok_or_else(|| Error::Model("MI-GAN missing skip connection".into()))?;
            let (next, next_rgb) = block.forward(&current, skip, rgb.as_ref())?;
            current = next;
            rgb = Some(next_rgb);
        }
        check_cancel(cancel)?;
        rgb.ok_or_else(|| Error::Model("MI-GAN has no synthesis blocks".into()))
    }

    /// Input is HWC RGB and white requests removal. Returns the network's
    /// complete RGB proposal; the engine must composite inside the mask only.
    pub fn generate_rgb8(&self, rgb8: &[u8], mask8: &[u8], cancel: &AtomicBool) -> Result<Vec<u8>> {
        let conditioned = conditioned_input(rgb8, mask8, &self.device)?;
        let output = self.infer_raw(&conditioned, cancel)?.to_device(&Device::Cpu)?.squeeze(0)?.to_vec3::<f32>()?;
        let mut rgb = vec![0u8; SIZE * SIZE * 3];
        for (channel, plane) in output.iter().enumerate() {
            for (index, &value) in plane.iter().flatten().enumerate() {
                if !value.is_finite() {
                    return Err(Error::Model("MI-GAN returned non-finite pixels; check the model file".into()));
                }
                let target = index * 3 + channel;
                if let Some(pixel) = rgb.get_mut(target) {
                    *pixel = ((value * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
                }
            }
        }
        Ok(rgb)
    }
}

pub fn conditioned_input(rgb8: &[u8], mask8: &[u8], device: &Device) -> Result<Tensor> {
    let pixels = SIZE * SIZE;
    if rgb8.len() != pixels * 3 || mask8.len() != pixels {
        return Err(Error::Model("MI-GAN requires a 512x512 RGB image and one-byte mask".into()));
    }
    let mut data = vec![0f32; pixels * 4];
    for (index, &mask) in mask8.iter().enumerate() {
        let known = if mask > 8 { 0.0 } else { 1.0 };
        *data.get_mut(index).ok_or_else(|| Error::Model("MI-GAN condition index overflow".into()))? = known - 0.5;
        for channel in 0..3 {
            if let Some(&pixel) = rgb8.get(index * 3 + channel) {
                *data.get_mut((channel + 1) * pixels + index).ok_or_else(|| Error::Model("MI-GAN condition index overflow".into()))? =
                    (pixel as f32 * (2.0 / 255.0) - 1.0) * known;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, 4, SIZE, SIZE), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditioning_uses_white_as_hole_and_rgb_in_channel_order() {
        let mut rgb = vec![0u8; SIZE * SIZE * 3];
        let mut mask = vec![0u8; SIZE * SIZE];
        rgb[..6].copy_from_slice(&[255, 0, 127, 4, 5, 6]);
        mask[1] = 255;
        let tensor = conditioned_input(&rgb, &mask, &Device::Cpu).unwrap().squeeze(0).unwrap().to_vec3::<f32>().unwrap();
        assert_eq!(tensor[0][0][0], 0.5);
        assert_eq!(tensor[0][0][1], -0.5);
        assert_eq!(tensor[1][0][0], 1.0);
        assert_eq!(tensor[2][0][0], -1.0);
        assert_eq!(tensor[1][0][1], 0.0);
        assert!(conditioned_input(&rgb[..3], &mask, &Device::Cpu).is_err());
    }

    #[test]
    fn upsample_checkerboard_matches_even_even_support() {
        let input = Tensor::ones((1, 1, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let weight = Tensor::ones((1, 1, 4, 4), DType::F32, &Device::Cpu).unwrap();
        let conv = DepthwiseConv { weight, bias: None, padding: 0, stride: 1 };
        let output = upsample(&input, &conv).unwrap();
        assert_eq!(output.dims(), [1, 1, 4, 4]);
    }

    #[test]
    fn tiled_depthwise_matches_grouped_convolution() {
        for (height, width) in [(7, 8), (133, 9)] {
            let input =
                Tensor::from_vec((0..2 * 3 * height * width).map(|i| (i % 97) as f32 / 47.0 - 1.0).collect(), (2, 3, height, width), &Device::Cpu)
                    .unwrap();
            for (kernel, padding, stride) in [(3, 1, 1), (4, 1, 2), (4, 0, 1)] {
                let weight =
                    Tensor::from_vec((0..3 * kernel * kernel).map(|i| (i as f32 - 10.0) / 31.0).collect(), (3, 1, kernel, kernel), &Device::Cpu)
                        .unwrap();
                let bias = Tensor::from_vec(vec![0.1f32, -0.2, 0.3], 3, &Device::Cpu).unwrap();
                let reference = Conv2d::new(weight.clone(), Some(bias.clone()), Conv2dConfig { padding, stride, groups: 3, ..Default::default() });
                let optimized = DepthwiseConv { weight, bias: Some(bias), padding, stride };
                let expected = reference.forward(&input).unwrap();
                let actual = optimized.forward_tiled(&input).unwrap();
                assert_eq!(actual.dims(), expected.dims());
                let error = (actual - expected).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
                assert!(error < 0.0001, "{height}x{width}, kernel {kernel}, padding {padding}, stride {stride}: max {error}");
            }
        }
    }

    #[test]
    fn raw_network_rejects_wrong_shape_and_observes_cancellation() {
        let model = Migan { device: Device::Cpu, encoder: Vec::new(), synthesis: Vec::new() };
        let wrong = Tensor::zeros((1, 4, 256, 256), DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(model.infer_raw(&wrong, &AtomicBool::new(false)), Err(Error::Model(_))));
        let input = Tensor::zeros((1, 4, SIZE, SIZE), DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(model.infer_raw(&input, &AtomicBool::new(true)), Err(Error::Cancelled)));
    }

    /// Set MIGAN_TEST_WEIGHTS and MIGAN_TEST_PARITY to the author's converted
    /// ONNX initializers and a safetensors file with `input` and `output`.
    #[test]
    #[ignore = "requires separately downloaded author model and ONNX Runtime parity fixture"]
    fn author_raw_network_parity() {
        let weights = std::env::var_os("MIGAN_TEST_WEIGHTS").unwrap();
        let fixture = std::env::var_os("MIGAN_TEST_PARITY").unwrap();
        let device = match std::env::var("MIGAN_TEST_DEVICE").as_deref() {
            Ok("cuda") => Device::new_cuda(0).unwrap(),
            _ => Device::Cpu,
        };
        let started = std::time::Instant::now();
        let model = Migan::load(Path::new(&weights), device.clone()).unwrap();
        eprintln!("MI-GAN load {:?}: {:?}", device, started.elapsed());
        let fixture = Weights::open(Path::new(&fixture)).unwrap();
        let input = fixture.read("input", DType::F32, &device).unwrap();
        let expected = fixture.read("output", DType::F32, &Device::Cpu).unwrap();
        let runs = std::env::var("MIGAN_TEST_RUNS").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(1).clamp(1, 3);
        for run in 1..=runs {
            let started = std::time::Instant::now();
            let actual = model.infer_raw(&input, &AtomicBool::new(false)).unwrap().to_device(&Device::Cpu).unwrap();
            eprintln!("MI-GAN inference {run}: {:?}", started.elapsed());
            let error = (actual - &expected).unwrap().abs().unwrap();
            let max_error = error.flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
            let mean_error = error.mean_all().unwrap().to_scalar::<f32>().unwrap();
            eprintln!("MI-GAN raw parity: max {max_error}, mean {mean_error}");
            assert!(max_error < 0.0001 && mean_error < 0.000001, "max {max_error}, mean {mean_error}");
        }
    }
}
