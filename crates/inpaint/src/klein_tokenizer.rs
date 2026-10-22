//! Pure-Rust Qwen2 BPE tokenizer for the public FLUX.2 Klein checkpoint.
//!
//! The on-disk Hugging Face `tokenizer.json` is the source of vocabulary and merges. This
//! implementation deliberately accepts only the tokenizer settings captured by the pinned
//! checkpoint; an unfamiliar tokenizer configuration fails closed instead of silently changing
//! prompt meaning.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use fancy_regex::{Regex, RegexBuilder};
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

use crate::{Error, Result};

const TOKENIZER_JSON_MAX: u64 = 32 * 1024 * 1024;
const PROMPT_MAX_BYTES: usize = 4096;
const MAX_CONTEXT: usize = 512;
const MAX_ADDED_TOKENS: usize = 1024;
const MAX_ADDED_TOKEN_BYTES: usize = 256;
const TOKEN_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KleinTokenizedInput {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
}

#[derive(Clone, Debug)]
struct AddedToken {
    text: String,
    id: u32,
}

/// Hugging Face Qwen2 BPE tokenizer with the Klein chat wrapper and right padding.
pub struct KleinTokenizer {
    vocab: HashMap<String, u32>,
    merges: HashMap<(String, String), usize>,
    added: Vec<AddedToken>,
    byte_chars: [char; 256],
    pretokenizer: Regex,
    pad_id: u32,
}

impl KleinTokenizer {
    /// Load a Hugging Face tokenizer directory containing `tokenizer.json`.
    pub fn load(tokenizer_dir: impl AsRef<Path>) -> Result<Self> {
        let path = tokenizer_dir.as_ref().join("tokenizer.json");
        let mut bytes = Vec::new();
        File::open(&path)?.take(TOKENIZER_JSON_MAX + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > TOKENIZER_JSON_MAX {
            return Err(Error::Model("tokenizer.json exceeded the 32 MiB limit while reading".into()));
        }
        let root: Value = serde_json::from_slice(&bytes)?;
        let config = root.get("backend_full_config").filter(|value| value.is_object()).unwrap_or(&root);
        validate_config(config)?;

        let model = object_field(config, "model")?;
        let vocab_value =
            model.get("vocab").and_then(Value::as_object).ok_or_else(|| Error::Model("tokenizer model is missing its vocabulary object".into()))?;
        if vocab_value.len() != 151_643 {
            return Err(Error::Model("Klein tokenizer vocabulary must contain exactly 151643 entries".into()));
        }
        let mut vocab = HashMap::with_capacity(151_643);
        for (token, id) in vocab_value {
            let id =
                id.as_u64().and_then(|v| u32::try_from(v).ok()).ok_or_else(|| Error::Model(format!("invalid vocabulary id for token {token:?}")))?;
            vocab.insert(token.clone(), id);
        }

        let merge_values =
            model.get("merges").and_then(Value::as_array).ok_or_else(|| Error::Model("tokenizer model is missing its merge list".into()))?;
        if merge_values.len() != 151_387 {
            return Err(Error::Model(format!("expected 151387 BPE merges, found {}", merge_values.len())));
        }
        let mut merges = HashMap::with_capacity(merge_values.len());
        for (rank, value) in merge_values.iter().enumerate() {
            let (left, right) = if let Some(pair) = value.as_str() {
                pair.split_once(' ').ok_or_else(|| Error::Model(format!("invalid BPE merge entry {pair:?}")))?
            } else if let Some(pair) = value.as_array().filter(|pair| pair.len() == 2) {
                (
                    pair.first().and_then(Value::as_str).ok_or_else(|| Error::Model("invalid left BPE token".into()))?,
                    pair.get(1).and_then(Value::as_str).ok_or_else(|| Error::Model("invalid right BPE token".into()))?,
                )
            } else {
                return Err(Error::Model("BPE merge entry must be a pair of strings".into()));
            };
            if left.is_empty() || right.is_empty() {
                return Err(Error::Model("BPE merge tokens must not be empty".into()));
            }
            merges.insert((left.to_owned(), right.to_owned()), rank);
        }

        let added_values =
            config.get("added_tokens").and_then(Value::as_array).ok_or_else(|| Error::Model("tokenizer is missing added_tokens".into()))?;
        validate_added_token_count(added_values.len())?;
        let mut added = Vec::with_capacity(added_values.len());
        for item in added_values {
            let text = item.get("content").and_then(Value::as_str).ok_or_else(|| Error::Model("added token is missing its content".into()))?;
            validate_added_token_text(text)?;
            let id = item
                .get("id")
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| Error::Model(format!("added token {text:?} has an invalid id")))?;
            if item.get("single_word").and_then(Value::as_bool) != Some(false)
                || item.get("lstrip").and_then(Value::as_bool) != Some(false)
                || item.get("rstrip").and_then(Value::as_bool) != Some(false)
                || item.get("normalized").and_then(Value::as_bool) != Some(false)
            {
                return Err(Error::Model(format!("unsupported added-token matching flags for {text:?}")));
            }
            added.push(AddedToken { text: text.to_owned(), id });
        }
        added.sort_by(|a, b| b.text.len().cmp(&a.text.len()).then_with(|| a.text.cmp(&b.text)));

        let mut regex_builder = RegexBuilder::new(TOKEN_PATTERN);
        regex_builder.backtrack_limit(100_000);
        let pretokenizer = regex_builder.build().map_err(|error| Error::Model(format!("could not compile Klein pre-tokenizer: {error}")))?;
        let token_id = |token: &str| -> Result<u32> {
            added
                .iter()
                .find(|item| item.text == token)
                .map(|item| item.id)
                .ok_or_else(|| Error::Model(format!("Klein tokenizer is missing required token {token}")))
        };

        let pad_id = token_id("<|endoftext|>")?;
        for token in ["<|im_start|>", "<|im_end|>", "<think>", "</think>"] {
            let _ = token_id(token)?;
        }
        Ok(Self { vocab, merges, added, byte_chars: bytes_to_unicode(), pretokenizer, pad_id })
    }

    /// Format the Klein user chat, tokenize it, truncate from the right, then pad to `max_length`.
    pub fn encode(&self, prompt: &str, max_length: usize) -> Result<KleinTokenizedInput> {
        if prompt.len() > PROMPT_MAX_BYTES {
            return Err(Error::Model(format!("prompt exceeds the {} byte limit", PROMPT_MAX_BYTES)));
        }
        if max_length == 0 || max_length > MAX_CONTEXT {
            return Err(Error::Model(format!("max_length must be between 1 and {MAX_CONTEXT}")));
        }
        let formatted = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        let mut input_ids = self.encode_unwrapped(&formatted)?;
        input_ids.truncate(max_length);
        let actual_len = input_ids.len();
        let mut attention_mask = vec![1; actual_len];
        input_ids.resize(max_length, self.pad_id);
        attention_mask.resize(max_length, 0);
        Ok(KleinTokenizedInput { input_ids, attention_mask })
    }

    fn encode_unwrapped(&self, text: &str) -> Result<Vec<u32>> {
        let mut output = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let mut next: Option<(usize, &AddedToken)> = None;
            for token in &self.added {
                if let Some(offset) = rest.find(&token.text)
                    && next.as_ref().is_none_or(|(best, previous)| offset < *best || (offset == *best && token.text.len() > previous.text.len()))
                {
                    next = Some((offset, token));
                }
            }
            match next {
                Some((offset, token)) => {
                    if offset > 0 {
                        self.encode_plain(&rest[..offset], &mut output)?;
                    }
                    output.push(token.id);
                    rest = &rest[offset + token.text.len()..];
                }
                None => {
                    self.encode_plain(rest, &mut output)?;
                    break;
                }
            }
        }
        Ok(output)
    }

    fn encode_plain(&self, text: &str, output: &mut Vec<u32>) -> Result<()> {
        let normalized: String = text.nfc().collect();
        let mut consumed = 0;
        for found in self.pretokenizer.find_iter(&normalized) {
            let found = found.map_err(|error| Error::Model(format!("Klein pre-tokenizer failed: {error}")))?;
            let start = found.start();
            let end = found.end();
            if start != consumed || end <= start {
                return Err(Error::Model("Klein pre-tokenizer did not cover the prompt contiguously".into()));
            }
            let piece = normalized.get(start..end).ok_or_else(|| Error::Model("pre-tokenizer split a UTF-8 character".into()))?;
            self.encode_piece(piece, output)?;
            consumed = end;
        }
        if consumed != normalized.len() {
            return Err(Error::Model("Klein pre-tokenizer left unprocessed prompt text".into()));
        }
        Ok(())
    }

    fn encode_piece(&self, piece: &str, output: &mut Vec<u32>) -> Result<()> {
        let mut symbols: Vec<String> = piece.as_bytes().iter().map(|byte| self.byte_chars[usize::from(*byte)].to_string()).collect();
        if symbols.len() > PROMPT_MAX_BYTES {
            return Err(Error::Model("pre-tokenized segment exceeds the prompt size limit".into()));
        }
        while symbols.len() > 1 {
            let best = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(index, pair)| self.merges.get(&(pair[0].clone(), pair[1].clone())).map(|rank| (index, *rank)))
                .min_by_key(|(_, rank)| *rank);
            let Some((index, _)) = best else { break };
            let left = symbols.get(index).cloned().ok_or_else(|| Error::Model("invalid BPE merge position".into()))?;
            let right = symbols.get(index + 1).cloned().ok_or_else(|| Error::Model("invalid BPE merge position".into()))?;
            symbols.splice(index..=index + 1, [format!("{left}{right}")]);
        }
        for symbol in symbols {
            let id = self
                .vocab
                .get(&symbol)
                .copied()
                .ok_or_else(|| Error::Model(format!("Klein tokenizer has no vocabulary entry for byte token {symbol:?}")))?;
            output.push(id);
        }
        Ok(())
    }
}

fn validate_config(config: &Value) -> Result<()> {
    let model = object_field(config, "model")?;
    if model.get("type").and_then(Value::as_str) != Some("BPE")
        || model.get("unk_token") != Some(&Value::Null)
        || model.get("continuing_subword_prefix").and_then(Value::as_str) != Some("")
        || model.get("end_of_word_suffix").and_then(Value::as_str) != Some("")
        || model.get("fuse_unk").and_then(Value::as_bool) != Some(false)
        || model.get("byte_fallback").and_then(Value::as_bool) != Some(false)
        || model.get("ignore_merges").and_then(Value::as_bool) != Some(false)
    {
        return Err(Error::Model("unsupported BPE tokenizer configuration".into()));
    }
    if config.get("normalizer").and_then(|v| v.get("type")).and_then(Value::as_str) != Some("NFC") {
        return Err(Error::Model("Klein tokenizer must use NFC normalization".into()));
    }
    let pre = object_field(config, "pre_tokenizer")?;
    if pre.get("type").and_then(Value::as_str) != Some("Sequence") {
        return Err(Error::Model("unsupported Klein pre-tokenizer sequence".into()));
    }
    let parts = pre.get("pretokenizers").and_then(Value::as_array).ok_or_else(|| Error::Model("missing pre-tokenizer steps".into()))?;
    let split = parts.first().ok_or_else(|| Error::Model("missing regex split pre-tokenizer".into()))?;
    if split.get("type").and_then(Value::as_str) != Some("Split")
        || split.get("behavior").and_then(Value::as_str) != Some("Isolated")
        || split.get("invert").and_then(Value::as_bool) != Some(false)
        || split.get("pattern").and_then(|v| v.get("Regex")).and_then(Value::as_str) != Some(TOKEN_PATTERN)
    {
        return Err(Error::Model("Klein regex pre-tokenizer settings do not match the pinned model".into()));
    }
    let byte_level = parts.get(1).ok_or_else(|| Error::Model("missing ByteLevel pre-tokenizer".into()))?;
    if parts.len() != 2
        || byte_level.get("type").and_then(Value::as_str) != Some("ByteLevel")
        || byte_level.get("add_prefix_space").and_then(Value::as_bool) != Some(false)
        || byte_level.get("trim_offsets").and_then(Value::as_bool) != Some(false)
        || byte_level.get("use_regex").and_then(Value::as_bool) != Some(false)
    {
        return Err(Error::Model("unsupported Klein ByteLevel pre-tokenizer settings".into()));
    }
    Ok(())
}

fn object_field<'a>(value: &'a Value, field: &str) -> Result<&'a serde_json::Map<String, Value>> {
    value.get(field).and_then(Value::as_object).ok_or_else(|| Error::Model(format!("tokenizer is missing object field {field:?}")))
}

fn validate_added_token_text(text: &str) -> Result<()> {
    if text.is_empty() {
        return Err(Error::Model("added token content must not be empty".into()));
    }
    if text.len() > MAX_ADDED_TOKEN_BYTES {
        return Err(Error::Model(format!("added token content exceeds the {MAX_ADDED_TOKEN_BYTES} byte limit")));
    }
    Ok(())
}

fn validate_added_token_count(count: usize) -> Result<()> {
    if count > MAX_ADDED_TOKENS {
        return Err(Error::Model(format!("added token count exceeds the {MAX_ADDED_TOKENS} entry limit")));
    }
    Ok(())
}

/// GPT-2 reversible mapping from bytes to printable Unicode code points.
fn bytes_to_unicode() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut bytes = Vec::new();
    let mut chars = Vec::new();
    for byte in b'!'..=b'~' {
        bytes.push(byte);
    }
    for byte in 0xA1..=0xAC {
        bytes.push(byte);
    }
    for byte in 0xAE..=0xFF {
        bytes.push(byte);
    }
    chars.extend(bytes.iter().map(|byte| u32::from(*byte)));
    let mut extra = 0u32;
    for byte in 0u16..=255 {
        if !bytes.contains(&(byte as u8)) {
            bytes.push(byte as u8);
            chars.push(256 + extra);
            extra += 1;
        }
    }
    for (byte, codepoint) in bytes.into_iter().zip(chars) {
        if let Some(character) = char::from_u32(codepoint) {
            map[usize::from(byte)] = character;
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_tokenizer() -> Result<KleinTokenizer> {
        let byte_chars = bytes_to_unicode();
        let mut vocab = HashMap::new();
        for (byte, symbol) in byte_chars.iter().enumerate() {
            vocab.insert(symbol.to_string(), byte as u32);
        }
        vocab.insert("ab".into(), 300);
        let added = [("<|endoftext|>", 500), ("<|im_start|>", 501), ("<|im_end|>", 502), ("<think>", 503), ("</think>", 504)]
            .into_iter()
            .map(|(text, id)| AddedToken { text: text.into(), id })
            .collect();
        let mut regex_builder = RegexBuilder::new(TOKEN_PATTERN);
        regex_builder.backtrack_limit(100_000);
        let pretokenizer = regex_builder.build().map_err(|error| Error::Model(format!("test regex should compile: {error}")))?;
        Ok(KleinTokenizer { vocab, merges: HashMap::from([(("a".into(), "b".into()), 0)]), added, byte_chars, pretokenizer, pad_id: 500 })
    }

    #[test]
    fn synthetic_tokenizer_covers_nfc_case_whitespace_bpe_and_added_tokens() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let tokenizer = synthetic_tokenizer()?;
        assert_eq!(tokenizer.encode_unwrapped("é")?, tokenizer.encode_unwrapped("e\u{301}")?);
        let lower = tokenizer.encode_unwrapped("a b")?;
        let uppercase = tokenizer.encode_unwrapped("A b")?;
        let repeated_space = tokenizer.encode_unwrapped("a  b")?;
        assert_ne!(lower, uppercase, "case is preserved");
        assert_ne!(lower, repeated_space, "whitespace is preserved");
        assert_eq!(tokenizer.encode_unwrapped("ab")?, vec![300], "the configured BPE merge is applied");
        assert_eq!(tokenizer.encode_unwrapped("<|im_end|>")?, vec![502], "added tokens bypass byte BPE");
        assert!(tokenizer.encode(&"x".repeat(PROMPT_MAX_BYTES + 1), 512).is_err());
        assert!(tokenizer.encode("x", MAX_CONTEXT + 1).is_err());
        Ok(())
    }

    #[test]
    fn rejects_empty_and_oversized_added_token_content() {
        assert!(validate_added_token_text("").is_err());
        assert!(validate_added_token_text(&"x".repeat(MAX_ADDED_TOKEN_BYTES + 1)).is_err());
        assert!(validate_added_token_text("<|valid|>").is_ok());
        assert!(validate_added_token_count(MAX_ADDED_TOKENS).is_ok());
        assert!(validate_added_token_count(MAX_ADDED_TOKENS + 1).is_err());
    }

    #[test]
    fn matches_captured_chat_goldens() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let Ok(goldens_path) = std::env::var("LIGHTCRAFT_KLEIN_TOKENIZER_GOLDENS") else { return Ok(()) };
        let goldens: Value = serde_json::from_slice(&std::fs::read(goldens_path)?)?;
        let tokenizer_dir = std::env::var("LIGHTCRAFT_KLEIN_TOKENIZER_DIR")?;
        let tokenizer = KleinTokenizer::load(tokenizer_dir)?;
        let raw_vectors = goldens.get("tokenization_vectors").and_then(Value::as_object).ok_or("missing raw tokenizer vectors")?;
        for (name, case) in raw_vectors {
            let input = case.get("input").and_then(Value::as_str).ok_or("missing raw input")?;
            let actual = tokenizer.encode_unwrapped(input)?;
            let expected: Vec<u32> = serde_json::from_value(case.get("ids_no_special_tokens").cloned().ok_or("missing raw ids")?)?;
            assert_eq!(actual, expected, "raw tokenizer vector {name}");
        }
        let cases = goldens.get("chat_template_goldens").and_then(Value::as_object).ok_or("missing chat goldens")?;
        for case in cases.values() {
            let prompt = case.get("prompt").and_then(Value::as_str).ok_or("missing golden prompt")?;
            let max_length =
                case.get("tokenizer_args").and_then(|v| v.get("max_length")).and_then(Value::as_u64).ok_or("missing max length")? as usize;
            let actual = tokenizer.encode(prompt, max_length)?;
            let expected_ids: Vec<u32> = serde_json::from_value(case.get("input_ids").cloned().ok_or("missing ids")?)?;
            let expected_mask: Vec<u32> = serde_json::from_value(case.get("attention_mask").cloned().ok_or("missing mask")?)?;
            assert_eq!(actual.input_ids, expected_ids);
            assert_eq!(actual.attention_mask, expected_mask);
        }
        Ok(())
    }

    #[test]
    fn rejects_oversized_prompt_and_context() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let tokenizer = synthetic_tokenizer()?;
        assert!(tokenizer.encode(&"x".repeat(PROMPT_MAX_BYTES + 1), 512).is_err());
        assert!(tokenizer.encode("x", 513).is_err());
        assert!(tokenizer.encode("x", 0).is_err());
        Ok(())
    }
}
