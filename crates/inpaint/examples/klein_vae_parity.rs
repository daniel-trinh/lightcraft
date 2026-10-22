//! Developer QA for a public checkpoint and separately generated PyTorch reference.
//! Usage: cargo run -p lightcraft-inpaint --example klein_vae_parity --features cuda -- \
//!   <vae.safetensors> <vae-N.safetensors>
//! Reference fixtures are generated outside the product and are not distributed.

use std::error::Error;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use lightcraft_inpaint::klein_vae::{KleinVae, KleinVaeConfig};

fn compare(label: &str, actual: &Tensor, expected: &Tensor) -> Result<(), Box<dyn Error>> {
    let lhs = actual.to_dtype(DType::F32)?;
    let rhs = expected.to_dtype(DType::F32)?;
    let difference = (&lhs - &rhs)?.abs()?;
    let a = lhs.flatten_all()?.to_vec1::<f32>()?;
    let b = rhs.flatten_all()?.to_vec1::<f32>()?;
    println!(
        "{label}: shape={:?} MAE={:.7} max={:.7} actual_first={:?} expected_first={:?} actual_mean={:.6} expected_mean={:.6}",
        actual.dims(),
        difference.mean_all()?.to_scalar::<f32>()?,
        difference.max_all()?.to_scalar::<f32>()?,
        &a[..a.len().min(6)],
        &b[..b.len().min(6)],
        a.iter().sum::<f32>() / a.len() as f32,
        b.iter().sum::<f32>() / b.len() as f32
    );
    Ok(())
}

fn run(checkpoint: &Path, fixture: &Path) -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;
    let model_bytes = std::fs::read(checkpoint)?;
    let vb = VarBuilder::from_buffered_safetensors(model_bytes, DType::F32, &device)?;
    let vae = KleinVae::new(KleinVaeConfig::default(), vb)?;
    println!("loaded Klein VAE on {:?}", device);
    let fixture_bytes = std::fs::read(fixture)?;
    let fv = VarBuilder::from_buffered_safetensors(fixture_bytes, DType::F32, &device)?;
    let input_name = if fv.contains_tensor("input_nchw_minus1_plus1_f32") { "input_nchw_minus1_plus1_f32" } else { "input_nchw_minus1_plus1_bf16" };
    let input = fv.get_unchecked(input_name)?;
    let (_, _, side, _) = input.dims4()?;
    let mean = vae.posterior_mean(&input)?;
    compare("posterior mean", &mean, &fv.get_unchecked("posterior_mean")?)?;
    let encoded = vae.encode(&input)?;
    let expected_encoded = if fv.contains_tensor("encoded_spatial") {
        fv.get_unchecked("encoded_spatial")?
    } else {
        let packed = fv.get_unchecked("encoded_packed")?;
        packed.reshape((1, side / 16, side / 16, 128))?.permute((0, 3, 1, 2))?
    };
    compare("normalized packed", &encoded, &expected_encoded)?;
    let decoded = vae.decode(&encoded)?;
    compare("encode-decode", &decoded, &fv.get_unchecked("decoder_output_minus1_plus1")?)?;
    let decode_ref_input = vae.decode(&expected_encoded)?;
    compare("decode reference latents", &decode_ref_input, &fv.get_unchecked("decoder_output_minus1_plus1")?)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os();
    let _program = args.next();
    let checkpoint = args.next().ok_or("missing VAE checkpoint path")?;
    let fixture = args.next().ok_or("missing VAE parity fixture path")?;
    if args.next().is_some() {
        return Err("too many arguments".into());
    }
    run(Path::new(&checkpoint), Path::new(&fixture))
}
