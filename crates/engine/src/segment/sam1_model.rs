//! Public, ungated SAM 1 checkpoint setup for click-based Object masks.

use std::path::PathBuf;

use super::{DownloadStatus, Segmenter};

pub const FILE_NAME: &str = "sam_vit_b_01ec64.safetensors";
pub const MODEL_BYTES: u64 = 374_979_376;
pub const MODEL_REVISION: &str = "b6e8348387e0296715f9e4a0a766b59c19a691fd";
pub const SHA256: &str = "fbf4c1ad844a2e9b867d89dc04a61f3ece19c57ebb7d8fcb6da92062a291d839";
pub const LICENSE_URL: &str = "https://huggingface.co/kornia/sam/blob/b6e8348387e0296715f9e4a0a766b59c19a691fd/LICENSE";
pub const MODEL_URL: &str = "https://huggingface.co/kornia/sam/resolve/b6e8348387e0296715f9e4a0a766b59c19a691fd";

/// Explicit developer path, saved preference, bundled checkpoint, then writable app storage.
pub fn default_model_path(saved: &str) -> Option<PathBuf> {
    std::env::var_os("LIGHTCRAFT_SAM1_MODEL")
        .map(PathBuf::from)
        .or_else(|| (!saved.trim().is_empty()).then(|| PathBuf::from(saved)))
        .or_else(|| {
            let path = std::env::current_exe().ok()?.parent()?.join("models/sam1").join(FILE_NAME);
            std::fs::metadata(&path).is_ok_and(|m| m.is_file() && m.len() == MODEL_BYTES).then_some(path)
        })
        .or_else(|| crate::camera_profiles::config_dir().map(|dir| dir.join("models/sam1").join(FILE_NAME)))
}

impl Segmenter {
    pub fn sam1_download_status(&self) -> DownloadStatus {
        #[cfg(feature = "sam")]
        {
            let s = self.sam1_download.status();
            DownloadStatus { running: s.running, done: s.done, total: s.total, file: s.file, error: s.error, finished: s.finished }
        }
        #[cfg(not(feature = "sam"))]
        DownloadStatus::default()
    }

    pub fn cancel_sam1_download(&self) -> bool {
        #[cfg(feature = "sam")]
        return self.sam1_download.cancel();
        #[cfg(not(feature = "sam"))]
        false
    }

    pub fn start_sam1_download(&self) -> Result<bool, String> {
        #[cfg(not(feature = "sam"))]
        return Err("AI masks are not available in this build".into());
        #[cfg(feature = "sam")]
        {
            use lightcraft_segment::fetch::{FileSpec, Options};
            const FILES: &[FileSpec] = &[FileSpec { name: FILE_NAME, size: Some(MODEL_BYTES), sha256: Some(SHA256), max: MODEL_BYTES }];
            // Verify the digest in the background even when metadata says installed;
            // otherwise a damaged same-size file would remain impossible to repair.
            let file = self.sam1_model.as_deref().ok_or("No local path is configured for the Object model")?;
            if file.file_name().and_then(|n| n.to_str()) != Some(FILE_NAME) {
                return Err(format!("The Object model download destination must be named {FILE_NAME}"));
            }
            let parent = file.parent().filter(|p| !p.as_os_str().is_empty()).ok_or("The Object model download needs a parent folder")?;
            self.sam1_download.start(FILES, vec![MODEL_URL.into()], parent.to_path_buf(), Options::default())
        }
    }
}
