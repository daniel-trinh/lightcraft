//! Developer parity check against work/klein-parity/tokenizer.json.
//! Usage: cargo run -p lightcraft-inpaint --example klein_tokenizer_parity -- <tokenizer-dir> <goldens.json>

use std::error::Error;
use std::fs;
use std::path::PathBuf;

use lightcraft_inpaint::klein_tokenizer::KleinTokenizer;
use serde_json::Value;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let tokenizer_dir = PathBuf::from(args.next().ok_or("missing tokenizer directory")?);
    let goldens_path = PathBuf::from(args.next().ok_or("missing tokenizer goldens JSON")?);
    if args.next().is_some() {
        return Err("expected exactly two arguments".into());
    }
    let tokenizer = KleinTokenizer::load(tokenizer_dir)?;
    let goldens: Value = serde_json::from_slice(&fs::read(goldens_path)?)?;
    let chat = goldens.get("chat_template_goldens").and_then(Value::as_object).ok_or("missing chat vectors")?;
    for (name, vector) in chat {
        let prompt = vector.get("prompt").and_then(Value::as_str).ok_or("missing chat prompt")?;
        let length =
            vector.get("tokenizer_args").and_then(|value| value.get("max_length")).and_then(Value::as_u64).ok_or("missing max_length")? as usize;
        let actual = tokenizer.encode(prompt, length)?;
        let ids: Vec<u32> = serde_json::from_value(vector.get("input_ids").cloned().ok_or("missing expected ids")?)?;
        let mask: Vec<u32> = serde_json::from_value(vector.get("attention_mask").cloned().ok_or("missing expected mask")?)?;
        if actual.input_ids != ids || actual.attention_mask != mask {
            return Err(format!("chat tokenizer vector {name} differs from Hugging Face output").into());
        }
    }
    println!("matched {} chat-template vectors", chat.len());
    Ok(())
}
