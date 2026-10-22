//! Qwen3 features used by FLUX.2 Klein: hidden states after layers 9, 18 and 27.
//!
//! Adapted from Candle 0.9.2's Apache-2.0 Qwen3 implementation. This encoder
//! deliberately omits unused language-model layers and the vocabulary head.

use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm};
use candle_transformers::utils::repeat_kv;

use crate::{Error, Result};

#[derive(Clone, Copy)]
struct Config {
    vocab: usize,
    hidden: usize,
    intermediate: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    captures: [usize; 3],
}

impl Default for Config {
    fn default() -> Self {
        Self { vocab: 151936, hidden: 2560, intermediate: 9728, heads: 32, kv_heads: 8, head_dim: 128, captures: [9, 18, 27] }
    }
}

struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    pre_attn: RmsNorm,
    pre_mlp: RmsNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
    cfg: Config,
}

impl Layer {
    fn new(cfg: Config, vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        let attn = vb.pp("self_attn");
        let mlp = vb.pp("mlp");
        Ok(Self {
            q: linear_no_bias(cfg.hidden, cfg.heads * cfg.head_dim, attn.pp("q_proj"))?,
            k: linear_no_bias(cfg.hidden, cfg.kv_heads * cfg.head_dim, attn.pp("k_proj"))?,
            v: linear_no_bias(cfg.hidden, cfg.kv_heads * cfg.head_dim, attn.pp("v_proj"))?,
            out: linear_no_bias(cfg.heads * cfg.head_dim, cfg.hidden, attn.pp("o_proj"))?,
            q_norm: rms_norm(cfg.head_dim, 1e-6, attn.pp("q_norm"))?,
            k_norm: rms_norm(cfg.head_dim, 1e-6, attn.pp("k_norm"))?,
            pre_attn: rms_norm(cfg.hidden, 1e-6, vb.pp("input_layernorm"))?,
            pre_mlp: rms_norm(cfg.hidden, 1e-6, vb.pp("post_attention_layernorm"))?,
            gate: linear_no_bias(cfg.hidden, cfg.intermediate, mlp.pp("gate_proj"))?,
            up: linear_no_bias(cfg.hidden, cfg.intermediate, mlp.pp("up_proj"))?,
            down: linear_no_bias(cfg.intermediate, cfg.hidden, mlp.pp("down_proj"))?,
            cfg,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let (_, sequence, _) = x.dims3()?;
        let cfg = self.cfg;
        let norm = x.apply(&self.pre_attn)?;
        let q = norm.apply(&self.q)?.reshape((1, sequence, cfg.heads, cfg.head_dim))?.transpose(1, 2)?.contiguous()?.apply(&self.q_norm)?;
        let k = norm.apply(&self.k)?.reshape((1, sequence, cfg.kv_heads, cfg.head_dim))?.transpose(1, 2)?.contiguous()?.apply(&self.k_norm)?;
        let v = norm.apply(&self.v)?.reshape((1, sequence, cfg.kv_heads, cfg.head_dim))?.transpose(1, 2)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;
        let groups = cfg.heads / cfg.kv_heads;
        let k = repeat_kv(k, groups)?.contiguous()?;
        let v = repeat_kv(v, groups)?.contiguous()?;
        // Accumulate attention in F32, matching Torch's SDPA math backend.
        // Sequence is capped at 512, so the score allocation is bounded.
        let scores = (q.to_dtype(DType::F32)?.matmul(&k.to_dtype(DType::F32)?.transpose(2, 3)?)? / (cfg.head_dim as f64).sqrt())?;
        let scores = scores.broadcast_add(mask)?;
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
        let context =
            probabilities.matmul(&v.to_dtype(DType::F32)?)?.to_dtype(x.dtype())?.transpose(1, 2)?.reshape((1, sequence, cfg.heads * cfg.head_dim))?;
        let x = (x + context.apply(&self.out)?)?;
        let h = x.apply(&self.pre_mlp)?;
        let gated = (candle_nn::ops::silu(&h.apply(&self.gate)?)? * h.apply(&self.up)?)?;
        x + gated.apply(&self.down)?
    }
}

pub struct KleinTextEncoder {
    embedding: Embedding,
    layers: Vec<Layer>,
    cos: Tensor,
    sin: Tensor,
    cfg: Config,
    device: Device,
}

impl KleinTextEncoder {
    pub fn new(vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        Self::with_config(Config::default(), vb)
    }

    fn with_config(cfg: Config, vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        if cfg.heads == 0
            || cfg.kv_heads == 0
            || !cfg.heads.is_multiple_of(cfg.kv_heads)
            || cfg.head_dim == 0
            || !cfg.head_dim.is_multiple_of(2)
            || cfg.captures.first() == Some(&0)
            || !cfg.captures.windows(2).all(|pair| pair.first() < pair.get(1))
        {
            candle_core::bail!("invalid Klein text encoder configuration");
        }
        let embedding = embedding(cfg.vocab, cfg.hidden, vb.pp("model.embed_tokens"))?;
        let count = cfg.captures.last().copied().ok_or_else(|| candle_core::Error::Msg("missing feature layer".into()))?;
        let mut layers = Vec::with_capacity(count);
        for layer in 0..count {
            layers.push(Layer::new(cfg, vb.pp("model.layers").pp(layer))?);
        }
        let inv: Vec<f32> = (0..cfg.head_dim).step_by(2).map(|i| 1. / 1_000_000f64.powf(i as f64 / cfg.head_dim as f64) as f32).collect();
        let inv = Tensor::from_vec(inv, (1, cfg.head_dim / 2), vb.device())?;
        let positions = Tensor::arange(0u32, 512, vb.device())?.to_dtype(DType::F32)?.reshape((512, 1))?;
        let frequencies = positions.matmul(&inv)?;
        Ok(Self {
            embedding,
            layers,
            cos: frequencies.cos()?.to_dtype(vb.dtype())?,
            sin: frequencies.sin()?.to_dtype(vb.dtype())?,
            cfg,
            device: vb.device().clone(),
        })
    }

    /// Returns [1, sequence, 7680] for the pinned Klein Qwen3 checkpoint.
    pub fn encode(&self, ids: &[u32], attention_mask: &[u32], cancelled: &AtomicBool) -> Result<Tensor> {
        let length = ids.len();
        if length == 0
            || length > 512
            || attention_mask.len() != length
            || ids.iter().any(|id| *id as usize >= self.cfg.vocab)
            || attention_mask.iter().any(|value| *value > 1)
            || attention_mask.first() != Some(&1)
            || attention_mask.windows(2).any(|pair| pair.first() == Some(&0) && pair.get(1) == Some(&1))
        {
            return Err(Error::Model("invalid Klein text tokens or right-padding mask".into()));
        }
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let tokens = Tensor::from_slice(ids, (1, length), &self.device)?;
        let mut hidden = self.embedding.forward(&tokens)?;
        let mut mask = Vec::with_capacity(length * length);
        for query in 0..length {
            for (key, keep) in attention_mask.iter().enumerate() {
                mask.push(if key <= query && *keep == 1 { 0. } else { f32::NEG_INFINITY });
            }
        }
        let mask = Tensor::from_vec(mask, (1, 1, length, length), &self.device)?;
        let cos = self.cos.narrow(0, 0, length)?;
        let sin = self.sin.narrow(0, 0, length)?;
        let mut captured = Vec::with_capacity(3);
        for (index, layer) in self.layers.iter().enumerate() {
            if cancelled.load(Ordering::Relaxed) {
                return Err(Error::Cancelled);
            }
            hidden = layer.forward(&hidden, &mask, &cos, &sin)?;
            if self.cfg.captures.contains(&(index + 1)) {
                captured.push(hidden.clone());
            }
        }
        // Concatenate each token's layer features, without the LM final norm.
        Ok(Tensor::cat(&captured, 2)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_are_bounded_and_cancellable() {
        let cfg = Config { vocab: 16, hidden: 8, intermediate: 12, heads: 2, kv_heads: 1, head_dim: 4, captures: [1, 2, 3] };
        let encoder = KleinTextEncoder::with_config(cfg, VarBuilder::zeros(DType::F32, &Device::Cpu)).unwrap();
        let cancel = AtomicBool::new(false);
        let result = encoder.encode(&[1, 2, 0], &[1, 1, 0], &cancel).unwrap();
        assert_eq!(result.dims(), [1, 3, 24]);
        assert!(result.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|v| v.is_finite()));
        assert!(encoder.encode(&[16], &[1], &cancel).is_err());
        assert!(encoder.encode(&[1, 2, 3], &[1, 0, 1], &cancel).is_err());
        assert!(encoder.encode(&[1], &[0], &cancel).is_err());
        assert!(encoder.encode(&[1; 513], &[1; 513], &cancel).is_err());
        cancel.store(true, Ordering::Relaxed);
        assert!(matches!(encoder.encode(&[1], &[1], &cancel), Err(Error::Cancelled)));
    }
}
