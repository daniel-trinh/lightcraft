//! Bounded, positional safetensors loading without memory mapping.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;

use crate::{Error, Result};

struct Entry {
    shape: Vec<usize>,
    dtype: DType,
    start: u64,
    len: usize,
}

pub struct Weights {
    file: Mutex<File>,
    entries: HashMap<String, Entry>,
    data_start: u64,
}

/// Allocation policies for the pinned model families. Existing models retain
/// their smaller bound; opting into Klein never removes the allocation limit.
#[derive(Clone, Copy)]
pub enum TensorLimit {
    Standard,
    KleinTransformer,
    KleinText,
}

impl TensorLimit {
    fn bytes(self) -> usize {
        match self {
            Self::Standard => 128 << 20,
            Self::KleinTransformer => 192 << 20,
            // Qwen3's BF16 token embedding is 777,912,320 bytes.
            Self::KleinText => 768 << 20,
        }
    }
}

impl Weights {
    pub(crate) fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn open(path: &Path) -> Result<Arc<Self>> {
        Self::open_with_limit(path, TensorLimit::Standard)
    }

    pub fn open_with_limit(path: &Path, limit: TensorLimit) -> Result<Arc<Self>> {
        let mut file = File::open(path).map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
        let file_len = file.metadata()?.len();
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len == 0 || header_len > (16 << 20) || header_len.checked_add(8).is_none_or(|n| n > file_len) {
            return Err(Error::Model("invalid safetensors header length".into()));
        }
        let mut bytes = vec![0u8; header_len as usize];
        file.read_exact(&mut bytes)?;
        let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&bytes)?;
        let mut entries = HashMap::new();
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            if entries.len() >= 8192 {
                return Err(Error::Model("too many tensors in model file".into()));
            }
            let dtype = match value.get("dtype").and_then(serde_json::Value::as_str) {
                Some("F16") => DType::F16,
                Some("BF16") => DType::BF16,
                Some("F32") => DType::F32,
                Some("I64") => DType::I64,
                _ => return Err(Error::Model(format!("unsupported dtype for {name}; expected F16, BF16, F32 or I64"))),
            };
            let shape: Vec<usize> = value
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| Error::Model(format!("missing shape for {name}")))?
                .iter()
                .map(|v| v.as_u64().and_then(|n| usize::try_from(n).ok()).ok_or_else(|| Error::Model(format!("bad shape for {name}"))))
                .collect::<Result<_>>()?;
            if shape.len() > 8 || shape.contains(&0) {
                return Err(Error::Model(format!("invalid tensor dimensions for {name}")));
            }
            let offsets =
                value.get("data_offsets").and_then(serde_json::Value::as_array).ok_or_else(|| Error::Model(format!("missing offsets for {name}")))?;
            let [start, end] = offsets.as_slice() else {
                return Err(Error::Model(format!("bad offsets for {name}")));
            };
            let start = start.as_u64().ok_or_else(|| Error::Model(format!("bad start for {name}")))?;
            let end = end.as_u64().ok_or_else(|| Error::Model(format!("bad end for {name}")))?;
            let size = shape
                .iter()
                .try_fold(dtype.size_in_bytes(), |a, n| a.checked_mul(*n))
                .ok_or_else(|| Error::Model(format!("oversized tensor {name}")))?;
            // A corrupt header cannot request the entire multi-GB model at once.
            if size > limit.bytes() {
                return Err(Error::Model(format!("tensor {name} exceeds the allocation limit")));
            }
            if end.checked_sub(start) != Some(size as u64) || header_len.checked_add(8).and_then(|n| n.checked_add(end)).is_none_or(|n| n > file_len)
            {
                return Err(Error::Model(format!("invalid size or extent for {name}")));
            }
            entries.insert(name, Entry { shape, dtype, start, len: size });
        }
        if entries.is_empty() {
            return Err(Error::Model("empty safetensors file".into()));
        }
        Ok(Arc::new(Self { file: Mutex::new(file), entries, data_start: header_len + 8 }))
    }

    pub(crate) fn read(&self, name: &str, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        let entry = self.entries.get(name).ok_or_else(|| candle_core::Error::Msg(format!("missing tensor {name}")))?;
        let mut data = vec![0u8; entry.len];
        let mut file = self.file.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        file.seek(SeekFrom::Start(self.data_start + entry.start)).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        file.read_exact(&mut data).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        drop(file);
        Tensor::from_raw_buffer(&data, entry.dtype, &entry.shape, device)?.to_dtype(dtype)
    }
}

pub struct Backend(pub Arc<Weights>);

/// A bounded set of safetensors shards with unambiguous tensor ownership.
pub struct MultiBackend {
    shards: Vec<Arc<Weights>>,
    owners: HashMap<String, usize>,
}

impl MultiBackend {
    pub fn new(shards: Vec<Arc<Weights>>) -> Result<Self> {
        if shards.is_empty() || shards.len() > 8 {
            return Err(Error::Model("model requires between one and eight weight shards".into()));
        }
        let mut owners = HashMap::new();
        for (index, shard) in shards.iter().enumerate() {
            for name in shard.entries.keys() {
                if owners.insert(name.clone(), index).is_some() {
                    return Err(Error::Model(format!("duplicate tensor across weight shards: {name}")));
                }
            }
        }
        Ok(Self { shards, owners })
    }

    fn shard(&self, name: &str) -> candle_core::Result<&Arc<Weights>> {
        self.owners.get(name).and_then(|index| self.shards.get(*index)).ok_or_else(|| candle_core::Error::Msg(format!("missing tensor {name}")))
    }
}

impl SimpleBackend for MultiBackend {
    fn get(&self, shape: Shape, name: &str, init: candle_nn::Init, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        Backend(Arc::clone(self.shard(name)?)).get(shape, name, init, dtype, device)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        self.shard(name)?.read(name, dtype, device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.owners.contains_key(name)
    }
}

impl SimpleBackend for Backend {
    fn get(&self, shape: Shape, name: &str, _init: candle_nn::Init, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        let entry = self.0.entries.get(name).ok_or_else(|| candle_core::Error::Msg(format!("missing tensor {name}")))?;
        if entry.shape.as_slice() != shape.dims() {
            candle_core::bail!("shape mismatch for {name}: expected {shape:?}, got {:?}", entry.shape);
        }
        let tensor = self.get_unchecked(name, dtype, device)?;
        if tensor.shape() != &shape {
            candle_core::bail!("shape mismatch for {name}: expected {shape:?}, got {:?}", tensor.shape());
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        self.0.read(name, dtype, device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.0.entries.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str, tensor: &str, dtype: &str, data: &[u8], elements: usize) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("lc-inpaint-{name}-{}.safetensors", std::process::id()));
        let header = serde_json::json!({tensor: {"dtype": dtype, "shape": [elements], "data_offsets": [0, data.len()]}}).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn bfloat16_tensor_loads_with_correct_byte_order() {
        let path = fixture("bf16", "x", "BF16", &[0x80, 0x3f, 0, 0xc0], 2);
        let weights = Weights::open(&path).unwrap();
        assert_eq!(weights.read("x", DType::F32, &Device::Cpu).unwrap().to_vec1::<f32>().unwrap(), [1.0, -2.0]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn shard_lookup_checks_names_shapes_and_duplicates() {
        let first = fixture("shard-first", "x", "BF16", &[0x80, 0x3f], 1);
        let second = fixture("shard-second", "y", "BF16", &[0, 0xc0], 1);
        let x = Weights::open(&first).unwrap();
        let y = Weights::open(&second).unwrap();
        assert!(MultiBackend::new(vec![]).is_err());
        assert!(MultiBackend::new(vec![Arc::clone(&x); 9]).is_err());
        assert!(MultiBackend::new(vec![Arc::clone(&x), Arc::clone(&x)]).is_err());
        let backend = MultiBackend::new(vec![x, y]).unwrap();
        assert!(backend.contains_tensor("y"));
        assert!(!backend.contains_tensor("absent"));
        assert_eq!(backend.get_unchecked("y", DType::F32, &Device::Cpu).unwrap().to_vec1::<f32>().unwrap(), [-2.0]);
        assert!(backend.get_unchecked("absent", DType::F32, &Device::Cpu).is_err());
        assert!(backend.get(Shape::from((2,)), "x", candle_nn::Init::Const(0.), DType::F32, &Device::Cpu).is_err());
        let _ = std::fs::remove_file(first);
        let _ = std::fs::remove_file(second);
    }

    #[test]
    fn larger_models_require_explicit_bounded_policy() {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!("lc-inpaint-policy-{}.safetensors", std::process::id()));
        for (elements, transformer_ok, text_ok) in [(84_934_656u64, true, true), (388_956_160, false, true), (402_653_185, false, false)] {
            let header = serde_json::json!({"x": {"dtype":"BF16", "shape":[elements], "data_offsets":[0,elements * 2]}}).to_string();
            let mut file = File::create(&path).unwrap();
            file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
            file.write_all(header.as_bytes()).unwrap();
            file.set_len(8 + header.len() as u64 + elements * 2).unwrap();
            drop(file);
            assert!(Weights::open(&path).is_err());
            assert_eq!(Weights::open_with_limit(&path, TensorLimit::KleinTransformer).is_ok(), transformer_ok);
            assert_eq!(Weights::open_with_limit(&path, TensorLimit::KleinText).is_ok(), text_ok);
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn oversized_tensor_is_rejected_before_allocation() {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!("lc-inpaint-oversized-{}.safetensors", std::process::id()));
        let header = br#"{"x":{"dtype":"F16","shape":[67108865],"data_offsets":[0,134217730]}}"#;
        let mut file = File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        file.write_all(header).unwrap();
        file.set_len(8 + header.len() as u64 + 134217730).unwrap();
        drop(file);
        let error = Weights::open(&path).err().unwrap().to_string();
        assert!(error.contains("allocation limit"), "{error}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn bounded_half_precision_tensor_loads_without_mapping() {
        let path = std::env::temp_dir().join(format!("lc-inpaint-weights-{}.safetensors", std::process::id()));
        let header = br#"{"x":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0, 0x3c, 0, 0x40]); // 1, 2 in IEEE half precision.
        std::fs::write(&path, &bytes).unwrap();
        let weights = Weights::open(&path).unwrap();
        assert_eq!(weights.read("x", DType::F32, &Device::Cpu).unwrap().to_vec1::<f32>().unwrap(), [1.0, 2.0]);
        bytes.truncate(bytes.len() - 1);
        std::fs::write(&path, &bytes).unwrap();
        assert!(Weights::open(&path).is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn mixed_half_and_integer_tensors_use_their_declared_byte_width() {
        let path = std::env::temp_dir().join(format!("lc-inpaint-mixed-{}.safetensors", std::process::id()));
        let header = br#"{"text_model.embeddings.position_ids":{"dtype":"I64","shape":[1,2],"data_offsets":[4,20]},"weight":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0, 0x3c, 0, 0x40]);
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&1i64.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let weights = Weights::open(&path).unwrap();
        assert_eq!(weights.read("weight", DType::F32, &Device::Cpu).unwrap().to_vec1::<f32>().unwrap(), [1.0, 2.0]);
        assert_eq!(weights.read("text_model.embeddings.position_ids", DType::I64, &Device::Cpu).unwrap().to_vec2::<i64>().unwrap(), [[0, 1]]);
        bytes.truncate(bytes.len() - 1);
        std::fs::write(&path, &bytes).unwrap();
        assert!(Weights::open(&path).is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn single_precision_tensor_loads_and_checks_extent() {
        let path = std::env::temp_dir().join(format!("lc-inpaint-f32-{}.safetensors", std::process::id()));
        let header = br#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.25f32.to_le_bytes());
        bytes.extend_from_slice(&(-2.5f32).to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let weights = Weights::open(&path).unwrap();
        assert_eq!(weights.read("weight", DType::F32, &Device::Cpu).unwrap().to_vec1::<f32>().unwrap(), [1.25, -2.5]);
        bytes.pop();
        std::fs::write(&path, &bytes).unwrap();
        assert!(Weights::open(&path).is_err());
        let _ = std::fs::remove_file(path);
    }
}
