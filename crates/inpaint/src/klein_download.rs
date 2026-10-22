//! Pinned public FLUX.2 Klein model files, downloaded only after an explicit user action.

use std::path::Path;
use std::sync::atomic::AtomicBool;

use lightcraft_fetch::{FileSpec, Options, Progress};

use crate::{Error, Result};

pub const LICENSE_NAME: &str = "Apache-2.0";
pub const LICENSE_URL: &str = "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/e7b7dc27f91deacad38e78976d1f2b499d76a294/LICENSE.md";
pub const MODEL_REVISION: &str = "e7b7dc27f91deacad38e78976d1f2b499d76a294";
pub const MODEL_BYTES: u64 = 15_975_644_852;
const MODEL_URL: &str = "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/e7b7dc27f91deacad38e78976d1f2b499d76a294";

pub struct ModelFile {
    /// Relative component directory; empty means the model root.
    pub folder: &'static str,
    pub file: FileSpec,
}

/// Runtime files from the public, ungated Hugging Face repository at the pinned revision.
///
/// SHA256 values for the LFS files are pinned from repository metadata. `LICENSE.md` is pinned
/// after its Git blob OID was matched. Hugging Face reports tokenizer.json as LFS too: its
/// repository `blobId` identifies the small LFS pointer, while the `lfs.sha256` identifies the
/// actual downloaded bytes.
pub const MODEL_FILES: &[ModelFile] = &[
    ModelFile {
        folder: "transformer",
        file: FileSpec {
            name: "diffusion_pytorch_model.safetensors",
            size: Some(7_751_109_744),
            sha256: Some("9f29f9edcfdae452a653ffb51a534ca4decd389952c225724ff3b94042612a6e"),
            max: 7_751_109_744,
        },
    },
    ModelFile {
        folder: "text_encoder",
        file: FileSpec {
            name: "model-00001-of-00002.safetensors",
            size: Some(4_967_215_360),
            sha256: Some("8c0506e7f4936fa7e26183a4fd8da4e2bdbc5990ba64ae441f965d51228f36ea"),
            max: 4_967_215_360,
        },
    },
    ModelFile {
        folder: "text_encoder",
        file: FileSpec {
            name: "model-00002-of-00002.safetensors",
            size: Some(3_077_766_632),
            sha256: Some("82f2bd839378541b0557bfabaf37c7d3d637071fdcb73302dedd7cf61162ce07"),
            max: 3_077_766_632,
        },
    },
    ModelFile {
        folder: "vae",
        file: FileSpec {
            name: "diffusion_pytorch_model.safetensors",
            size: Some(168_120_878),
            sha256: Some("ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04"),
            max: 168_120_878,
        },
    },
    ModelFile {
        folder: "tokenizer",
        file: FileSpec {
            name: "tokenizer.json",
            size: Some(11_422_654),
            sha256: Some("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
            max: 11_422_654,
        },
    },
    ModelFile {
        folder: "",
        file: FileSpec {
            name: "LICENSE.md",
            size: Some(9_584),
            sha256: Some("ca02bc51900ab07789d1b70283329e7137f5af98f5161c23a1c81fc38a4af1fe"),
            max: 9_584,
        },
    },
];

/// Fast startup/UI readiness check; safetensor headers are validated when weights are opened.
pub fn installed(dir: &Path) -> bool {
    MODEL_FILES.iter().all(|entry| {
        std::fs::metadata(dir.join(entry.folder).join(entry.file.name))
            .is_ok_and(|metadata| metadata.is_file() && Some(metadata.len()) == entry.file.size)
    })
}

/// Download files sequentially, preserving `lightcraft-fetch` resume, cancellation and checks.
pub fn download(dir: &Path, cancel: &AtomicBool, progress: &mut dyn FnMut(&Progress)) -> Result<()> {
    let mut finished = 0u64;
    for entry in MODEL_FILES {
        let base = if entry.folder.is_empty() { MODEL_URL.to_owned() } else { format!("{MODEL_URL}/{}", entry.folder) };
        let offset = finished;
        lightcraft_fetch::download(
            std::slice::from_ref(&entry.file),
            &[base],
            &dir.join(entry.folder),
            &Options::default(),
            cancel,
            &mut |current| {
                let file = if entry.folder.is_empty() { current.file.clone() } else { format!("{}/{}", entry.folder, current.file) };
                progress(&Progress {
                    file,
                    done: offset.saturating_add(current.done).min(MODEL_BYTES),
                    total: MODEL_BYTES,
                    mirror: current.mirror.clone(),
                });
            },
        )
        .map_err(|error| Error::Model(format!("Klein model download: {error}")))?;
        finished = finished.saturating_add(entry.file.size.unwrap_or(0));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_manifest_is_pinned_and_bounded() {
        assert_eq!(MODEL_FILES.len(), 6);
        assert_eq!(MODEL_FILES.iter().filter_map(|entry| entry.file.size).sum::<u64>(), MODEL_BYTES);
        assert!(MODEL_FILES.iter().all(|entry| entry.file.max == entry.file.size.unwrap_or(0)));
        assert!(MODEL_FILES.iter().all(|entry| entry.file.sha256.is_some()));
        assert_eq!(MODEL_REVISION.len(), 40);
        assert!(LICENSE_URL.contains(MODEL_REVISION));
        assert_eq!(MODEL_FILES.iter().filter(|entry| entry.file.name.ends_with(".safetensors")).count(), 4);
    }

    #[test]
    fn pre_cancelled_download_stops_before_network() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!("lightcraft-klein-cancel-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&root)?;
        let cancel = AtomicBool::new(true);
        let result = download(&root, &cancel, &mut |_| {});
        let _ = std::fs::remove_dir_all(&root);
        assert!(result.is_err());
        Ok(())
    }
}
