//! FLUX.2 Klein transformer inference, ported from Apache-2.0 Diffusers 0.38.0.
//!
//! Reference: `Flux2Transformer2DModel` in Hugging Face Diffusers 0.38.0 and
//! Black Forest Labs FLUX.2 source commit 50fe5162777813d869182b139e83b10743caef15.
//! The checkpoint remains outside the repository. This module reads its public
//! Diffusers tensor names through a caller-supplied, bounded `VarBuilder`.

use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

const QUERY_TILE: usize = 128;
const MAX_TOKENS: usize = 9216;

/// Model architecture. `Default` matches public FLUX.2 Klein distilled 4B.
#[derive(Clone, Debug)]
pub struct KleinConfig {
    pub in_channels: usize,
    pub joint_attention_dim: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub mlp_ratio: usize,
    pub axes_dims_rope: [usize; 4],
    pub rope_theta: f64,
    pub eps: f64,
    pub timestep_guidance_channels: usize,
}

impl Default for KleinConfig {
    fn default() -> Self {
        Self {
            in_channels: 128,
            joint_attention_dim: 7680,
            num_attention_heads: 24,
            attention_head_dim: 128,
            num_layers: 5,
            num_single_layers: 20,
            mlp_ratio: 3,
            axes_dims_rope: [32; 4],
            rope_theta: 2000.0,
            eps: 1e-6,
            timestep_guidance_channels: 256,
        }
    }
}

impl KleinConfig {
    fn validate(&self) -> Result<usize> {
        let hidden = self
            .num_attention_heads
            .checked_mul(self.attention_head_dim)
            .ok_or_else(|| candle_core::Error::Msg("Klein hidden dimension overflow".into()))?;
        if self.in_channels == 0
            || self.in_channels > 128
            || self.joint_attention_dim == 0
            || self.joint_attention_dim > 7680
            || hidden == 0
            || hidden > 3072
            || self.num_layers > 5
            || self.num_single_layers > 20
            || self.mlp_ratio == 0
            || self.mlp_ratio > 3
            || self.timestep_guidance_channels == 0
            || self.timestep_guidance_channels > 256
            || !self.timestep_guidance_channels.is_multiple_of(2)
            || self.axes_dims_rope.iter().any(|n| *n == 0 || *n > 128 || !n.is_multiple_of(2))
            || self.axes_dims_rope.iter().sum::<usize>() != self.attention_head_dim
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 1.0
            || !self.eps.is_finite()
            || self.eps <= 0.0
        {
            candle_core::bail!("unsupported or malformed Klein transformer configuration");
        }
        Ok(hidden)
    }
}

fn linear(vb: VarBuilder<'_>, input: usize, output: usize) -> Result<Linear> {
    linear_no_bias(input, output, vb)
}

fn layer_norm(xs: &Tensor, eps: f64) -> Result<Tensor> {
    // PyTorch LayerNorm accumulates statistics in F32 for BF16 model activations.
    let dtype = xs.dtype();
    let x = xs.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
    centered.broadcast_mul(&(&variance + eps)?.sqrt()?.recip()?)?.to_dtype(dtype)
}

fn rms_norm(xs: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = xs.dtype();
    let x = xs.to_dtype(DType::F32)?;
    let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
    x.broadcast_mul(&(&variance + eps)?.sqrt()?.recip()?)?.to_dtype(dtype)?.broadcast_mul(weight)
}

fn silu(xs: &Tensor) -> Result<Tensor> {
    candle_nn::ops::silu(xs)
}

fn mod_slice(modulation: &Tensor, index: usize, hidden: usize) -> Result<Tensor> {
    modulation.narrow(1, index * hidden, hidden)?.unsqueeze(1)
}

fn modulated(xs: &Tensor, shift: &Tensor, scale: &Tensor, eps: f64) -> Result<Tensor> {
    layer_norm(xs, eps)?.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

fn swiglu(xs: &Tensor, inner: usize) -> Result<Tensor> {
    let a = xs.narrow(D::Minus1, 0, inner)?;
    let b = xs.narrow(D::Minus1, inner, inner)?;
    silu(&a)?.mul(&b)
}

struct FeedForward {
    linear_in: Linear,
    linear_out: Linear,
    inner: usize,
}

impl FeedForward {
    fn new(vb: VarBuilder<'_>, hidden: usize, ratio: usize) -> Result<Self> {
        let inner = hidden * ratio;
        Ok(Self { linear_in: linear(vb.pp("linear_in"), hidden, inner * 2)?, linear_out: linear(vb.pp("linear_out"), inner, hidden)?, inner })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear_out.forward(&swiglu(&self.linear_in.forward(xs)?, self.inner)?)
    }
}

struct DoubleAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    add_q_proj: Linear,
    add_k_proj: Linear,
    add_v_proj: Linear,
    to_out: Linear,
    to_add_out: Linear,
    norm_q: Tensor,
    norm_k: Tensor,
    norm_added_q: Tensor,
    norm_added_k: Tensor,
}

impl DoubleAttention {
    fn new(vb: VarBuilder<'_>, hidden: usize, head_dim: usize) -> Result<Self> {
        Ok(Self {
            to_q: linear(vb.pp("to_q"), hidden, hidden)?,
            to_k: linear(vb.pp("to_k"), hidden, hidden)?,
            to_v: linear(vb.pp("to_v"), hidden, hidden)?,
            add_q_proj: linear(vb.pp("add_q_proj"), hidden, hidden)?,
            add_k_proj: linear(vb.pp("add_k_proj"), hidden, hidden)?,
            add_v_proj: linear(vb.pp("add_v_proj"), hidden, hidden)?,
            to_out: linear(vb.pp("to_out").pp("0"), hidden, hidden)?,
            to_add_out: linear(vb.pp("to_add_out"), hidden, hidden)?,
            norm_q: vb.get((head_dim,), "norm_q.weight")?,
            norm_k: vb.get((head_dim,), "norm_k.weight")?,
            norm_added_q: vb.get((head_dim,), "norm_added_q.weight")?,
            norm_added_k: vb.get((head_dim,), "norm_added_k.weight")?,
        })
    }

    fn forward(&self, image: &Tensor, text: &Tensor, rope: &Rope, heads: usize, head_dim: usize, eps: f64) -> Result<(Tensor, Tensor)> {
        let (batch, image_len, _) = image.dims3()?;
        let text_len = text.dim(1)?;
        let query = rms_norm(&self.to_q.forward(image)?.reshape((batch, image_len, heads, head_dim))?, &self.norm_q, eps)?;
        let key = rms_norm(&self.to_k.forward(image)?.reshape((batch, image_len, heads, head_dim))?, &self.norm_k, eps)?;
        let value = self.to_v.forward(image)?.reshape((batch, image_len, heads, head_dim))?;
        let text_query = rms_norm(&self.add_q_proj.forward(text)?.reshape((batch, text_len, heads, head_dim))?, &self.norm_added_q, eps)?;
        let text_key = rms_norm(&self.add_k_proj.forward(text)?.reshape((batch, text_len, heads, head_dim))?, &self.norm_added_k, eps)?;
        let text_value = self.add_v_proj.forward(text)?.reshape((batch, text_len, heads, head_dim))?;
        let query = apply_rope(&Tensor::cat(&[text_query, query], 1)?, rope)?;
        let key = apply_rope(&Tensor::cat(&[text_key, key], 1)?, rope)?;
        let value = Tensor::cat(&[text_value, value], 1)?;
        let output = tiled_attention(&query, &key, &value, head_dim)?;
        let text_out = output.narrow(1, 0, text_len)?.flatten_from(2)?;
        let image_out = output.narrow(1, text_len, image_len)?.flatten_from(2)?;
        Ok((self.to_out.forward(&image_out)?, self.to_add_out.forward(&text_out)?))
    }
}

struct DoubleBlock {
    attn: DoubleAttention,
    ff: FeedForward,
    ff_context: FeedForward,
}

impl DoubleBlock {
    fn new(vb: VarBuilder<'_>, cfg: &KleinConfig, hidden: usize) -> Result<Self> {
        Ok(Self {
            attn: DoubleAttention::new(vb.pp("attn"), hidden, cfg.attention_head_dim)?,
            ff: FeedForward::new(vb.pp("ff"), hidden, cfg.mlp_ratio)?,
            ff_context: FeedForward::new(vb.pp("ff_context"), hidden, cfg.mlp_ratio)?,
        })
    }

    fn forward(
        &self,
        image: &Tensor,
        text: &Tensor,
        img_mod: &Tensor,
        txt_mod: &Tensor,
        rope: &Rope,
        cfg: &KleinConfig,
        hidden: usize,
    ) -> Result<(Tensor, Tensor)> {
        let image_norm = modulated(image, &mod_slice(img_mod, 0, hidden)?, &mod_slice(img_mod, 1, hidden)?, cfg.eps)?;
        let text_norm = modulated(text, &mod_slice(txt_mod, 0, hidden)?, &mod_slice(txt_mod, 1, hidden)?, cfg.eps)?;
        let (image_attn, text_attn) = self.attn.forward(&image_norm, &text_norm, rope, cfg.num_attention_heads, cfg.attention_head_dim, cfg.eps)?;
        let image = image.broadcast_add(&image_attn.broadcast_mul(&mod_slice(img_mod, 2, hidden)?)?)?;
        let image_ff = self.ff.forward(&modulated(&image, &mod_slice(img_mod, 3, hidden)?, &mod_slice(img_mod, 4, hidden)?, cfg.eps)?)?;
        let image = image.broadcast_add(&image_ff.broadcast_mul(&mod_slice(img_mod, 5, hidden)?)?)?;
        let text = text.broadcast_add(&text_attn.broadcast_mul(&mod_slice(txt_mod, 2, hidden)?)?)?;
        let text_ff = self.ff_context.forward(&modulated(&text, &mod_slice(txt_mod, 3, hidden)?, &mod_slice(txt_mod, 4, hidden)?, cfg.eps)?)?;
        let text = text.broadcast_add(&text_ff.broadcast_mul(&mod_slice(txt_mod, 5, hidden)?)?)?;
        Ok((image, text))
    }
}

struct SingleBlock {
    qkv_mlp: Linear,
    to_out: Linear,
    norm_q: Tensor,
    norm_k: Tensor,
}

impl SingleBlock {
    fn new(vb: VarBuilder<'_>, hidden: usize, head_dim: usize, ratio: usize) -> Result<Self> {
        let vb = vb.pp("attn");
        Ok(Self {
            qkv_mlp: linear(vb.pp("to_qkv_mlp_proj"), hidden, hidden * (3 + 2 * ratio))?,
            to_out: linear(vb.pp("to_out"), hidden * (1 + ratio), hidden)?,
            norm_q: vb.get((head_dim,), "norm_q.weight")?,
            norm_k: vb.get((head_dim,), "norm_k.weight")?,
        })
    }

    fn forward(&self, xs: &Tensor, modulation: &Tensor, rope: &Rope, cfg: &KleinConfig, hidden: usize) -> Result<Tensor> {
        let normalized = modulated(xs, &mod_slice(modulation, 0, hidden)?, &mod_slice(modulation, 1, hidden)?, cfg.eps)?;
        let projected = self.qkv_mlp.forward(&normalized)?;
        let (batch, tokens, _) = projected.dims3()?;
        let query = rms_norm(
            &projected.narrow(2, 0, hidden)?.reshape((batch, tokens, cfg.num_attention_heads, cfg.attention_head_dim))?,
            &self.norm_q,
            cfg.eps,
        )?;
        let key = rms_norm(
            &projected.narrow(2, hidden, hidden)?.reshape((batch, tokens, cfg.num_attention_heads, cfg.attention_head_dim))?,
            &self.norm_k,
            cfg.eps,
        )?;
        let value = projected.narrow(2, 2 * hidden, hidden)?.reshape((batch, tokens, cfg.num_attention_heads, cfg.attention_head_dim))?;
        let attention = tiled_attention(&apply_rope(&query, rope)?, &apply_rope(&key, rope)?, &value, cfg.attention_head_dim)?.flatten_from(2)?;
        let inner = hidden * cfg.mlp_ratio;
        let mlp = swiglu(&projected.narrow(2, 3 * hidden, 2 * inner)?, inner)?;
        let output = self.to_out.forward(&Tensor::cat(&[attention, mlp], 2)?)?;
        xs.broadcast_add(&output.broadcast_mul(&mod_slice(modulation, 2, hidden)?)?)
    }
}

struct Rope {
    cos: Tensor,
    sin: Tensor,
}

fn rope(ids: &Tensor, cfg: &KleinConfig, device: &Device) -> Result<Rope> {
    let shape = ids.dims();
    let ids = match shape {
        [1, _, 4] => ids.squeeze(0)?,
        [_, 4] => ids.clone(),
        _ => candle_core::bail!("Klein position IDs must have shape [N,4] or [1,N,4]"),
    };
    let (tokens, _) = ids.dims2()?;
    let coordinates = ids.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let mut cos = Vec::with_capacity(tokens * cfg.attention_head_dim);
    let mut sin = Vec::with_capacity(tokens * cfg.attention_head_dim);
    for point in coordinates {
        for (axis, dim) in cfg.axes_dims_rope.iter().enumerate() {
            let position = f64::from(point[axis]);
            if !position.is_finite() || position.abs() > 1_000_000.0 {
                candle_core::bail!("invalid Klein position ID");
            }
            for index in 0..(dim / 2) {
                let angle = position * cfg.rope_theta.powf(-((2 * index) as f64) / *dim as f64);
                let (c, s) = (angle.cos() as f32, angle.sin() as f32);
                cos.extend_from_slice(&[c, c]);
                sin.extend_from_slice(&[s, s]);
            }
        }
    }
    Ok(Rope {
        cos: Tensor::from_vec(cos, (1, tokens, 1, cfg.attention_head_dim), device)?,
        sin: Tensor::from_vec(sin, (1, tokens, 1, cfg.attention_head_dim), device)?,
    })
}

fn apply_rope(xs: &Tensor, rope: &Rope) -> Result<Tensor> {
    let (batch, tokens, heads, head_dim) = xs.dims4()?;
    if !head_dim.is_multiple_of(2) || rope.cos.dim(1)? != tokens {
        candle_core::bail!("Klein RoPE shape mismatch");
    }
    let pair = xs.to_dtype(DType::F32)?.reshape((batch, tokens, heads, head_dim / 2, 2))?;
    let real = pair.narrow(4, 0, 1)?.squeeze(4)?;
    let imag = pair.narrow(4, 1, 1)?.squeeze(4)?;
    let rotated = Tensor::stack(&[imag.neg()?, real], 4)?.reshape((batch, tokens, heads, head_dim))?;
    xs.to_dtype(DType::F32)?.broadcast_mul(&rope.cos)?.broadcast_add(&rotated.broadcast_mul(&rope.sin)?)?.to_dtype(xs.dtype())
}

/// Exact full-key softmax, tiled only along queries. Peak score storage is
/// O(batch * heads * QUERY_TILE * keys), never O(tokens²).
fn tiled_attention(query: &Tensor, key: &Tensor, value: &Tensor, head_dim: usize) -> Result<Tensor> {
    let (batch, tokens, heads, dim) = query.dims4()?;
    if dim != head_dim
        || tokens == 0
        || key.dims4()? != (batch, tokens, heads, dim)
        || value.dims4()? != (batch, tokens, heads, dim)
        || tokens > MAX_TOKENS
    {
        candle_core::bail!("invalid or excessive Klein attention shape");
    }
    let q = query.transpose(1, 2)?.to_dtype(DType::F32)?;
    let k = key.transpose(1, 2)?.to_dtype(DType::F32)?.transpose(2, 3)?;
    let v = value.transpose(1, 2)?.to_dtype(DType::F32)?;
    let mut chunks = Vec::with_capacity(tokens.div_ceil(QUERY_TILE));
    let scale = 1.0 / (head_dim as f64).sqrt();
    for start in (0..tokens).step_by(QUERY_TILE) {
        let len = QUERY_TILE.min(tokens - start);
        let scores = (q.narrow(2, start, len)?.matmul(&k)? * scale)?;
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
        chunks.push(probabilities.matmul(&v)?);
    }
    Tensor::cat(&chunks, 2)?.transpose(1, 2)?.to_dtype(query.dtype())
}

fn timestep_embedding(timestep: &Tensor, cfg: &KleinConfig, dtype: DType, device: &Device) -> Result<Tensor> {
    let batch = timestep.dim(0)?;
    // Diffusers casts the input to model dtype, multiplies by 1000, then
    // computes sinusoidal frequencies in F32 before the learned projection.
    let values = (timestep.to_dtype(dtype)? * 1000.0)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let half = cfg.timestep_guidance_channels / 2;
    let mut embedding = Vec::with_capacity(batch * cfg.timestep_guidance_channels);
    for t in values {
        if !t.is_finite() || !(0.0..=1000.0).contains(&t) {
            candle_core::bail!("Klein scaled timestep must be finite and between 0 and 1000");
        }
        for index in 0..half {
            let angle = f64::from(t) * (-10000.0_f64.ln() * index as f64 / half as f64).exp();
            embedding.push(angle.cos() as f32);
        }
        for index in 0..half {
            let angle = f64::from(t) * (-10000.0_f64.ln() * index as f64 / half as f64).exp();
            embedding.push(angle.sin() as f32);
        }
    }
    Tensor::from_vec(embedding, (batch, cfg.timestep_guidance_channels), device)?.to_dtype(dtype)
}

/// Batch-one inference-only transformer for the public distilled Klein 4B
/// checkpoint. Caller supplies a safe bounded weight backend via `VarBuilder`.
pub struct KleinTransformer {
    cfg: KleinConfig,
    hidden: usize,
    x_embedder: Linear,
    context_embedder: Linear,
    time_linear_1: Linear,
    time_linear_2: Linear,
    double_mod_img: Linear,
    double_mod_txt: Linear,
    single_mod: Linear,
    double_blocks: Vec<DoubleBlock>,
    single_blocks: Vec<SingleBlock>,
    norm_out: Linear,
    proj_out: Linear,
}

impl KleinTransformer {
    pub fn new(cfg: &KleinConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let hidden = cfg.validate()?;
        let mut double_blocks = Vec::with_capacity(cfg.num_layers);
        for index in 0..cfg.num_layers {
            double_blocks.push(DoubleBlock::new(vb.pp(format!("transformer_blocks.{index}")), cfg, hidden)?);
        }
        let mut single_blocks = Vec::with_capacity(cfg.num_single_layers);
        for index in 0..cfg.num_single_layers {
            single_blocks.push(SingleBlock::new(vb.pp(format!("single_transformer_blocks.{index}")), hidden, cfg.attention_head_dim, cfg.mlp_ratio)?);
        }
        Ok(Self {
            cfg: cfg.clone(),
            hidden,
            x_embedder: linear(vb.pp("x_embedder"), cfg.in_channels, hidden)?,
            context_embedder: linear(vb.pp("context_embedder"), cfg.joint_attention_dim, hidden)?,
            time_linear_1: linear(vb.pp("time_guidance_embed.timestep_embedder.linear_1"), cfg.timestep_guidance_channels, hidden)?,
            time_linear_2: linear(vb.pp("time_guidance_embed.timestep_embedder.linear_2"), hidden, hidden)?,
            double_mod_img: linear(vb.pp("double_stream_modulation_img.linear"), hidden, hidden * 6)?,
            double_mod_txt: linear(vb.pp("double_stream_modulation_txt.linear"), hidden, hidden * 6)?,
            single_mod: linear(vb.pp("single_stream_modulation.linear"), hidden, hidden * 3)?,
            double_blocks,
            single_blocks,
            norm_out: linear(vb.pp("norm_out.linear"), hidden, hidden * 2)?,
            proj_out: linear(vb.pp("proj_out"), hidden, cfg.in_channels)?,
        })
    }

    /// `hidden_states`: [1, N_image, 128]; `encoder_hidden_states`:
    /// [1, N_text, 7680]; `timestep`: [1] in 0..1; `img_ids` and `txt_ids`:
    /// [1, N, 4] or [N, 4]. Returns [1, N_image, 128].
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
    ) -> Result<Tensor> {
        self.forward_impl(hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids, None)
    }

    /// Like `forward`, with cooperative cancellation before and between blocks.
    pub fn forward_cancel(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
        cancel: &AtomicBool,
    ) -> Result<Tensor> {
        self.forward_impl(hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids, Some(cancel))
    }

    fn forward_impl(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
        cancel: Option<&AtomicBool>,
    ) -> Result<Tensor> {
        let check_cancel = || -> Result<()> {
            if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                candle_core::bail!("Klein generation cancelled");
            }
            Ok(())
        };
        check_cancel()?;
        let (batch, image_len, channels) = hidden_states.dims3()?;
        let (text_batch, text_len, text_channels) = encoder_hidden_states.dims3()?;
        if batch != 1
            || text_batch != 1
            || image_len == 0
            || text_len == 0
            || image_len.checked_add(text_len).is_none_or(|n| n > MAX_TOKENS)
            || channels != self.cfg.in_channels
            || text_channels != self.cfg.joint_attention_dim
            || timestep.dims() != [1]
            || hidden_states.dtype() != encoder_hidden_states.dtype()
            || hidden_states.device().location() != encoder_hidden_states.device().location()
        {
            candle_core::bail!("invalid or excessive Klein transformer inputs");
        }
        let id_len = |ids: &Tensor| -> Result<usize> {
            match ids.dims() {
                [len, 4] | [1, len, 4] => Ok(*len),
                _ => candle_core::bail!("Klein position IDs must have shape [N,4] or [1,N,4]"),
            }
        };
        let (img_ids_len, txt_ids_len) = (id_len(img_ids)?, id_len(txt_ids)?);
        if img_ids_len != image_len || txt_ids_len != text_len {
            candle_core::bail!("Klein position IDs differ from token lengths");
        }
        let device = hidden_states.device();
        let timesteps = timestep_embedding(timestep, &self.cfg, hidden_states.dtype(), device)?;
        let temb = self.time_linear_2.forward(&silu(&self.time_linear_1.forward(&timesteps)?)?)?;
        let mod_input = silu(&temb)?;
        let img_mod = self.double_mod_img.forward(&mod_input)?;
        let txt_mod = self.double_mod_txt.forward(&mod_input)?;
        let single_mod = self.single_mod.forward(&mod_input)?;
        let image_rope = rope(img_ids, &self.cfg, device)?;
        let text_rope = rope(txt_ids, &self.cfg, device)?;
        let rope = Rope { cos: Tensor::cat(&[text_rope.cos, image_rope.cos], 1)?, sin: Tensor::cat(&[text_rope.sin, image_rope.sin], 1)? };
        let mut image = self.x_embedder.forward(hidden_states)?;
        let mut text = self.context_embedder.forward(encoder_hidden_states)?;
        for block in &self.double_blocks {
            check_cancel()?;
            (image, text) = block.forward(&image, &text, &img_mod, &txt_mod, &rope, &self.cfg, self.hidden)?;
        }
        let mut combined = Tensor::cat(&[text, image], 1)?;
        for block in &self.single_blocks {
            check_cancel()?;
            combined = block.forward(&combined, &single_mod, &rope, &self.cfg, self.hidden)?;
        }
        check_cancel()?;
        let image = combined.narrow(1, text_len, image_len)?;
        let out_mod = self.norm_out.forward(&silu(&temb)?)?;
        let scale = out_mod.narrow(1, 0, self.hidden)?.unsqueeze(1)?;
        let shift = out_mod.narrow(1, self.hidden, self.hidden)?.unsqueeze(1)?;
        let normalized = layer_norm(&image, self.cfg.eps)?.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
        self.proj_out.forward(&normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_tiling_matches_dense_attention() -> Result<()> {
        let device = Device::Cpu;
        let q = Tensor::from_vec((0..822).map(|i| (i as f32 * 0.13).sin()).collect(), (1, 137, 2, 3), &device)?;
        let k = Tensor::from_vec((0..822).map(|i| (i as f32 * 0.19).cos()).collect(), (1, 137, 2, 3), &device)?;
        let v = Tensor::from_vec((0..822).map(|i| (i as f32 * 0.11).sin()).collect(), (1, 137, 2, 3), &device)?;
        let dense_scores = (q.transpose(1, 2)?.matmul(&k.transpose(1, 2)?.transpose(2, 3)?)? * (1.0 / 3.0_f64.sqrt()))?;
        let dense = candle_nn::ops::softmax_last_dim(&dense_scores)?.matmul(&v.transpose(1, 2)?)?.transpose(1, 2)?;
        let tiled = tiled_attention(&q, &k, &v, 3)?;
        let error = dense.broadcast_sub(&tiled)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(error < 1e-6, "max error {error}");
        Ok(())
    }

    #[test]
    fn rope_rotates_interleaved_pairs_on_each_axis() -> Result<()> {
        let device = Device::Cpu;
        let cfg = KleinConfig {
            in_channels: 4,
            joint_attention_dim: 8,
            num_attention_heads: 1,
            attention_head_dim: 8,
            num_layers: 0,
            num_single_layers: 0,
            mlp_ratio: 1,
            axes_dims_rope: [2; 4],
            timestep_guidance_channels: 8,
            ..KleinConfig::default()
        };
        let ids = Tensor::new(&[[1f32, 2.0, 3.0, 4.0]], &device)?;
        let q = Tensor::new(&[[[[1f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]]], &device)?;
        let rotated = apply_rope(&q, &rope(&ids, &cfg, &device)?)?.flatten_all()?.to_vec1::<f32>()?;
        for (axis, pair) in rotated.as_chunks::<2>().0.iter().enumerate() {
            let angle = (axis + 1) as f32;
            assert!((pair[0] - angle.cos()).abs() < 1e-6);
            assert!((pair[1] - angle.sin()).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn malformed_inputs_are_rejected() -> Result<()> {
        let device = Device::Cpu;
        let cfg = KleinConfig {
            in_channels: 4,
            joint_attention_dim: 8,
            num_attention_heads: 1,
            attention_head_dim: 8,
            num_layers: 0,
            num_single_layers: 0,
            mlp_ratio: 1,
            axes_dims_rope: [2; 4],
            timestep_guidance_channels: 8,
            ..KleinConfig::default()
        };
        let model = KleinTransformer::new(&cfg, VarBuilder::zeros(DType::F32, &device))?;
        let image = Tensor::zeros((1, 2, 4), DType::F32, &device)?;
        let text = Tensor::zeros((1, 3, 8), DType::F32, &device)?;
        let t = Tensor::zeros((1,), DType::F32, &device)?;
        let img_ids = Tensor::zeros((1, 2, 4), DType::F32, &device)?;
        let txt_ids = Tensor::zeros((1, 3, 4), DType::F32, &device)?;
        assert_eq!(model.forward(&image, &text, &t, &img_ids, &txt_ids)?.dims(), &[1, 2, 4]);
        assert!(model.forward(&image, &text, &t, &img_ids, &img_ids).is_err());
        assert!(model.forward(&image, &text, &t, &Tensor::zeros((), DType::F32, &device)?, &txt_ids).is_err());
        let cancel = AtomicBool::new(true);
        assert!(model.forward_cancel(&image, &text, &t, &img_ids, &txt_ids, &cancel).is_err());
        Ok(())
    }

    /// Run manually with `KLEIN_WEIGHTS` and `KLEIN_FIXTURE` pointing to the
    /// pinned public transformer and captured Diffusers first-call fixture.
    /// `KLEIN_TEST_DEVICE=cuda` selects the packaged CUDA backend explicitly.
    #[test]
    #[ignore = "requires external public checkpoint and captured Diffusers tensors"]
    fn public_checkpoint_matches_diffusers_first_call() -> Result<()> {
        use crate::weights::{Backend, TensorLimit, Weights};
        use std::path::Path;

        let required = |name| std::env::var(name).map_err(|_| candle_core::Error::Msg(format!("set {name} to run the Klein parity test")));
        let model_path = required("KLEIN_WEIGHTS")?;
        let fixture_path = required("KLEIN_FIXTURE")?;
        let device = match std::env::var("KLEIN_TEST_DEVICE").as_deref() {
            Ok("cuda") => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(0)?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    candle_core::bail!("compile with the cuda feature for GPU parity")
                }
            }
            Ok("cpu") | Err(_) => Device::Cpu,
            Ok(other) => candle_core::bail!("unsupported KLEIN_TEST_DEVICE={other}"),
        };
        let open = |path: &str| Weights::open(Path::new(path)).map_err(|e| candle_core::Error::Msg(e.to_string()));
        let started = std::time::Instant::now();
        let model_weights =
            Weights::open_with_limit(Path::new(&model_path), TensorLimit::KleinTransformer).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let vb = VarBuilder::from_backend(Box::new(Backend(model_weights)), DType::BF16, device.clone());
        let model = KleinTransformer::new(&KleinConfig::default(), vb)?;
        let loaded = started.elapsed();
        let fixture = open(&fixture_path)?;
        let input = |name| fixture.read(name, DType::BF16, &device);
        let expected = fixture.read("output_0", DType::BF16, &device)?;
        let started = std::time::Instant::now();
        let actual = model.forward(
            &input("hidden_states")?,
            &input("encoder_hidden_states")?,
            &input("timestep")?,
            &fixture.read("img_ids", DType::I64, &device)?,
            &fixture.read("txt_ids", DType::I64, &device)?,
        )?;
        let inference = started.elapsed();
        let difference = actual.to_dtype(DType::F32)?.broadcast_sub(&expected.to_dtype(DType::F32)?)?.abs()?;
        let maximum = difference.max_all()?.to_scalar::<f32>()?;
        let mean = difference.mean_all()?.to_scalar::<f32>()?;
        eprintln!("Klein transformer parity device={:?} load={loaded:?} forward={inference:?} max={maximum} mean={mean}", device.location());
        assert!(maximum.is_finite() && mean.is_finite() && maximum < 0.5 && mean < 0.02, "native transformer diverged from Diffusers");
        Ok(())
    }
}
