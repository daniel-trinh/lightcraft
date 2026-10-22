//! Developer-only public-checkpoint parity against a captured Diffusers call.
//!
//! cargo run -p lightcraft-inpaint --example klein_transformer_parity --features cuda --
//!     <transformer.safetensors> <first-call.safetensors> cuda

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_nn::var_builder::SimpleBackend;
use lightcraft_inpaint::klein_transformer::{KleinConfig, KleinTransformer};
use lightcraft_inpaint::weights::{Backend, TensorLimit, Weights};

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let weights_path = PathBuf::from(args.next().ok_or("expected transformer checkpoint path")?);
    let fixture_path = PathBuf::from(args.next().ok_or("expected first-call fixture path")?);
    let device_choice = args.next().ok_or("expected device: cpu or cuda")?;
    if args.next().is_some() {
        return Err("too many arguments".into());
    }
    let device = match device_choice.to_str() {
        Some("cpu") => Device::Cpu,
        Some("cuda") => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err("build with --features cuda to use the NVIDIA GPU".into());
            }
        }
        _ => return Err("device must be cpu or cuda".into()),
    };
    let loaded_at = Instant::now();
    let checkpoint = Weights::open_with_limit(&weights_path, TensorLimit::KleinTransformer)?;
    let vb = VarBuilder::from_backend(Box::new(Backend(checkpoint)), DType::BF16, device.clone());
    let model = KleinTransformer::new(&KleinConfig::default(), vb)?;
    let load_time = loaded_at.elapsed();
    let fixture = Backend(Weights::open(&fixture_path)?);
    let input = |name, dtype| fixture.get_unchecked(name, dtype, &device);
    let image = input("hidden_states", DType::BF16)?;
    let text = input("encoder_hidden_states", DType::BF16)?;
    let timestep = input("timestep", DType::BF16)?;
    let image_ids = input("img_ids", DType::I64)?;
    let text_ids = input("txt_ids", DType::I64)?;
    let expected = input("output_0", DType::BF16)?;
    let forward_at = Instant::now();
    let actual = model.forward(&image, &text, &timestep, &image_ids, &text_ids)?;
    let forward_time = forward_at.elapsed();
    let difference = actual.to_dtype(DType::F32)?.broadcast_sub(&expected.to_dtype(DType::F32)?)?.abs()?;
    let max = difference.max_all()?.to_scalar::<f32>()?;
    let mean = difference.mean_all()?.to_scalar::<f32>()?;
    println!(
        "device={:?} image_tokens={} text_tokens={} load={load_time:?} forward={forward_time:?} max_abs={max} mean_abs={mean}",
        device.location(),
        image.dim(1)?,
        text.dim(1)?
    );
    if !max.is_finite() || !mean.is_finite() || max >= 0.5 || mean >= 0.02 {
        return Err("native transformer diverged from pinned Diffusers output".into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    run()
}
