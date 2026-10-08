//! The user-selected, separately downloaded SD 1.5 inpainting checkpoint.

use std::path::Path;
use std::sync::atomic::AtomicBool;

use lightcraft_fetch::{FileSpec, Options, Progress};

use crate::{Error, Result};

pub const LICENSE_NAME: &str = "CreativeML OpenRAIL-M";
pub const LICENSE_URL: &str = "https://huggingface.co/spaces/CompVis/stable-diffusion-license";
pub const MODEL_REVISION: &str = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb";
pub const MODEL_BYTES: u64 = 2_134_218_891;
const MODEL_URL: &str = "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/resolve/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb";

pub struct ModelFile {
    pub folder: &'static str,
    pub file: FileSpec,
}

/// Download filenames are local to each component folder. All large files have pinned hashes.
pub const MODEL_FILES: &[ModelFile] = &[
    ModelFile {
        folder: "unet",
        file: FileSpec {
            name: "diffusion_pytorch_model.fp16.safetensors",
            size: Some(1_719_154_104),
            sha256: Some("24b788b4a777748377cc20364eea4ae113c8c42f4468c16bc8c02fdae5492af9"),
            max: 1_719_154_104,
        },
    },
    ModelFile {
        folder: "vae",
        file: FileSpec {
            name: "diffusion_pytorch_model.fp16.safetensors",
            size: Some(167_335_342),
            sha256: Some("4fbcf0ebe55a0984f5a5e00d8c4521d52359af7229bb4d81890039d2aa16dd7c"),
            max: 167_335_342,
        },
    },
    ModelFile {
        folder: "text_encoder",
        file: FileSpec {
            name: "model.fp16.safetensors",
            size: Some(246_144_864),
            sha256: Some("77795e2023adcf39bc29a884661950380bd093cf0750a966d473d1718dc9ef4e"),
            max: 246_144_864,
        },
    },
    ModelFile {
        folder: "tokenizer",
        file: FileSpec {
            name: "merges.txt",
            size: Some(524_619),
            sha256: Some("9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"),
            max: 524_619,
        },
    },
    ModelFile {
        folder: "tokenizer",
        file: FileSpec {
            name: "vocab.json",
            size: Some(1_059_962),
            sha256: Some("e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"),
            max: 1_059_962,
        },
    },
];

/// Fast metadata check suitable for UI frames. `Model::load` validates tensor headers.
pub fn installed(dir: &Path) -> bool {
    MODEL_FILES.iter().all(|f| std::fs::metadata(dir.join(f.folder).join(f.file.name)).is_ok_and(|m| m.is_file() && Some(m.len()) == f.file.size))
}

pub fn download(dir: &Path, cancel: &AtomicBool, progress: &mut dyn FnMut(&Progress)) -> Result<()> {
    let mut finished = 0u64;
    for model_file in MODEL_FILES {
        let base = format!("{MODEL_URL}/{}", model_file.folder);
        let offset = finished;
        lightcraft_fetch::download(
            std::slice::from_ref(&model_file.file),
            &[base],
            &dir.join(model_file.folder),
            &Options::default(),
            cancel,
            &mut |p| {
                progress(&Progress {
                    file: format!("{}/{}", model_file.folder, p.file),
                    done: offset.saturating_add(p.done),
                    total: MODEL_BYTES,
                    mirror: p.mirror.clone(),
                })
            },
        )
        .map_err(|e| Error::Model(format!("model download: {e}")))?;
        finished = finished.saturating_add(model_file.file.size.unwrap_or(0));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_manifest_is_pinned_and_bounded() {
        assert_eq!(MODEL_FILES.iter().filter_map(|f| f.file.size).sum::<u64>(), MODEL_BYTES);
        assert!(MODEL_FILES.iter().all(|f| f.file.max == f.file.size.unwrap_or(0)));
        assert!(MODEL_FILES.iter().filter(|f| f.file.name.ends_with("safetensors")).all(|f| f.file.sha256.is_some()));
    }
}
