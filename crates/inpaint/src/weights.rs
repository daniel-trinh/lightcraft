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

impl Weights {
    pub fn open(path: &Path) -> Result<Arc<Self>> {
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
                Some("I64") => DType::I64,
                _ => return Err(Error::Model(format!("unsupported dtype for {name}; expected F16 or I64"))),
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
            // The largest supported tensor is CLIP's 75 MiB token embedding. A corrupt
            // header cannot request allocation of the entire multi-GB model as one tensor.
            if size > (128 << 20) {
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

    fn read(&self, name: &str, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
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
}
