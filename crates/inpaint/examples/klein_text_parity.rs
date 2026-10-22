//! Developer-only parity probe; all model and reference paths are local.
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, var_builder::SimpleBackend};
use lightcraft_inpaint::{
    klein_text::KleinTextEncoder,
    weights::{Backend, MultiBackend, TensorLimit, Weights},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let model = PathBuf::from(args.first().ok_or("expected model directory and reference safetensors path")?);
    let reference = PathBuf::from(args.get(1).ok_or("expected reference safetensors path")?);
    let device = if args.iter().any(|arg| arg == "--cuda") { Device::new_cuda(0)? } else { Device::Cpu };
    let dtype = if args.iter().any(|arg| arg == "--f32") { DType::F32 } else { DType::BF16 };
    let fixture = Backend(Weights::open(&reference)?);
    let read = |name: &str, dtype| fixture.get_unchecked(name, dtype, &Device::Cpu);
    let ids = read("input_ids", DType::I64)?.flatten_all()?.to_vec1::<i64>()?.into_iter().map(u32::try_from).collect::<Result<Vec<_>, _>>()?;
    let mask = read("attention_mask", DType::I64)?.flatten_all()?.to_vec1::<i64>()?.into_iter().map(u32::try_from).collect::<Result<Vec<_>, _>>()?;
    let mut shards = Vec::new();
    for name in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"] {
        shards.push(Weights::open_with_limit(&model.join("text_encoder").join(name), TensorLimit::KleinText)?);
    }
    let backend = MultiBackend::new(shards)?;
    let started = std::time::Instant::now();
    let encoder = KleinTextEncoder::new(VarBuilder::from_backend(Box::new(backend), dtype, device.clone()))?;
    eprintln!("Loaded partial Qwen3 in {:.2}s", started.elapsed().as_secs_f64());
    let result = encoder.encode(&ids, &mask, &AtomicBool::new(false))?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let mut metrics = Vec::new();
    for (slot, layer) in [9, 18, 27].into_iter().enumerate() {
        let expected = read(&format!("hidden_states_{layer}"), DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let actual = result.narrow(2, slot * 2560, 2560)?.flatten_all()?.to_vec1::<f32>()?;
        let mut sum = 0f64;
        let mut maximum = 0f64;
        let mut dot = 0f64;
        let mut aa = 0f64;
        let mut bb = 0f64;
        for (&a, &b) in actual.iter().zip(&expected) {
            let (a, b) = (f64::from(a), f64::from(b));
            let difference = (a - b).abs();
            sum += difference;
            maximum = maximum.max(difference);
            dot += a * b;
            aa += a * a;
            bb += b * b;
        }
        metrics.push(serde_json::json!({"layer":layer,"mae":sum / actual.len() as f64,"max_abs":maximum,"cosine":dot / (aa*bb).sqrt()}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"device":format!("{device:?}"),"dtype":format!("{dtype:?}"),"sequence":ids.len(),"elapsed_seconds":started.elapsed().as_secs_f64(),"metrics":metrics})
        )?
    );
    Ok(())
}
