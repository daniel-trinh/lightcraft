//! App-managed native inpainting model setup, progress and cancellation.
//! No service or external inference process is needed.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

pub const NOT_INSTALLED: &str = "The inpainting model is not installed";
pub const LICENSE_NAME: &str = "CreativeML OpenRAIL-M";
pub const LICENSE_URL: &str = "https://huggingface.co/spaces/CompVis/stable-diffusion-license";
pub const KLEIN_NOT_INSTALLED: &str = "The FLUX.2 Klein model is not installed";
pub const KLEIN_LICENSE_NAME: &str = "Apache-2.0";
pub const KLEIN_LICENSE_URL: &str =
    "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/e7b7dc27f91deacad38e78976d1f2b499d76a294/LICENSE.md";
pub const KLEIN_MODEL_BYTES: u64 = 15_975_644_852;

pub fn backend_description() -> &'static str {
    if std::env::var("LIGHTCRAFT_AI_DEVICE").as_deref() == Ok("cpu") {
        "CPU (explicit diagnostic override)"
    } else if cfg!(feature = "cuda") {
        "NVIDIA GPU (CUDA)"
    } else if cfg!(target_os = "macos") {
        "Metal when available, otherwise CPU"
    } else {
        "CPU"
    }
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct DownloadStatus {
    pub running: bool,
    pub done: u64,
    pub total: u64,
    pub file: String,
    pub error: Option<String>,
    pub finished: bool,
}

#[derive(Default)]
struct DownloadText {
    file: String,
    error: Option<String>,
}

#[derive(Default)]
struct DownloadShared {
    running: AtomicBool,
    cancel: AtomicBool,
    done: AtomicU64,
    total: AtomicU64,
    finished: AtomicBool,
    text: Mutex<DownloadText>,
}

#[cfg(feature = "inpaint")]
struct DownloadGuard(Arc<DownloadShared>);
#[cfg(feature = "inpaint")]
impl Drop for DownloadGuard {
    fn drop(&mut self) {
        self.0.running.store(false, Ordering::SeqCst);
    }
}

#[derive(Default)]
pub struct GenerationProgress {
    done: AtomicUsize,
    total: AtomicUsize,
    phase: AtomicUsize,
}

impl GenerationProgress {
    pub fn update(&self, done: usize, total: usize) {
        self.total.store(total, Ordering::Relaxed);
        self.done.store(done.min(total), Ordering::Relaxed);
    }

    pub fn get(&self) -> (usize, usize) {
        (self.done.load(Ordering::Relaxed), self.total.load(Ordering::Relaxed))
    }

    pub fn set_phase(&self, phase: &str) {
        let value = match phase {
            "Text" => 1,
            "Encode" => 2,
            "Denoise" => 3,
            "Decode" => 4,
            _ => 0,
        };
        self.phase.store(value, Ordering::Relaxed);
    }

    pub fn phase(&self) -> &'static str {
        match self.phase.load(Ordering::Relaxed) {
            1 => "Text",
            2 => "Encode",
            3 => "Denoise",
            4 => "Decode",
            _ => "",
        }
    }
}

/// Signals owned by a background generation job; cancellation is checked between steps.
pub struct GenerationSignals {
    pub cancel: Arc<AtomicBool>,
    pub progress: Arc<GenerationProgress>,
}

pub struct Generator {
    pub dir: Option<PathBuf>,
    pub klein_dir: Option<PathBuf>,
    pub removal: Option<PathBuf>,
    download: Arc<DownloadShared>,
    klein_download: Arc<DownloadShared>,
    generation_cancel: Arc<AtomicBool>,
    generation_progress: Arc<GenerationProgress>,
}

impl Default for Generator {
    fn default() -> Self {
        Self {
            dir: std::env::var_os("LIGHTCRAFT_INPAINT_DIR")
                .map(PathBuf::from)
                .or_else(|| crate::camera_profiles::config_dir().map(|d| d.join("models/sd15-inpaint"))),
            klein_dir: std::env::var_os("LIGHTCRAFT_KLEIN_DIR").map(PathBuf::from).or_else(|| {
                let portable = std::env::current_exe().ok().and_then(|exe| exe.parent().map(|dir| dir.join("models/flux2-klein-4b")));
                #[cfg(feature = "inpaint")]
                if let Some(ref dir) = portable
                    && lightcraft_inpaint::klein_download::installed(dir)
                {
                    return portable;
                }
                #[cfg(not(feature = "inpaint"))]
                let _ = portable;
                crate::camera_profiles::config_dir().map(|dir| dir.join("models/flux2-klein-4b"))
            }),
            removal: std::env::var_os("LIGHTCRAFT_MIGAN_MODEL")
                .map(PathBuf::from)
                .or_else(|| std::env::current_exe().ok()?.parent().map(|dir| dir.join("models/migan-f32.safetensors"))),
            download: Arc::default(),
            klein_download: Arc::default(),
            generation_cancel: Arc::default(),
            generation_progress: Arc::default(),
        }
    }
}

impl Generator {
    pub const AVAILABLE: bool = cfg!(feature = "inpaint");
    pub const KLEIN_AVAILABLE: bool = cfg!(all(feature = "inpaint", feature = "cuda"));

    pub fn klein_installed(&self) -> bool {
        #[cfg(feature = "inpaint")]
        return self.klein_dir.as_deref().is_some_and(lightcraft_inpaint::klein_download::installed);
        #[cfg(not(feature = "inpaint"))]
        false
    }

    pub fn klein_model_dir(&self) -> Result<PathBuf, String> {
        if !Self::KLEIN_AVAILABLE {
            return Err("Klein generation requires the NVIDIA CUDA build".into());
        }
        if std::env::var("LIGHTCRAFT_AI_DEVICE").as_deref() == Ok("cpu") {
            return Err("Klein generation requires an NVIDIA CUDA GPU; remove the CPU diagnostic override".into());
        }
        let dir = self.klein_dir.clone().ok_or("No folder is available for the Klein model")?;
        if self.klein_installed() {
            Ok(dir)
        } else {
            Err(format!("{KLEIN_NOT_INSTALLED}. Use the Klein model download button, or install the pinned files in {}.", dir.display()))
        }
    }

    pub fn klein_download_status(&self) -> DownloadStatus {
        let s = &self.klein_download;
        let (file, error) = s.text.try_lock().map(|t| (t.file.clone(), t.error.clone())).unwrap_or_default();
        DownloadStatus {
            running: s.running.load(Ordering::SeqCst),
            done: s.done.load(Ordering::Relaxed),
            total: s.total.load(Ordering::Relaxed),
            file,
            error,
            finished: s.finished.load(Ordering::SeqCst),
        }
    }

    pub fn cancel_klein_download(&self) -> bool {
        let running = self.klein_download.running.load(Ordering::SeqCst);
        if running {
            self.klein_download.cancel.store(true, Ordering::SeqCst);
        }
        running
    }

    /// Also verifies and repairs already present files when explicitly requested.
    pub fn start_klein_download(&self) -> Result<bool, String> {
        #[cfg(not(all(feature = "inpaint", feature = "cuda")))]
        return Err("Klein generation requires the NVIDIA CUDA build".into());
        #[cfg(all(feature = "inpaint", feature = "cuda"))]
        {
            let dir = self.klein_dir.clone().ok_or("No folder is available for the Klein model")?;
            let s = self.klein_download.clone();
            if s.running.swap(true, Ordering::SeqCst) {
                return Ok(false);
            }
            s.cancel.store(false, Ordering::SeqCst);
            s.done.store(0, Ordering::Relaxed);
            s.total.store(KLEIN_MODEL_BYTES, Ordering::Relaxed);
            s.finished.store(false, Ordering::SeqCst);
            *s.text.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = DownloadText::default();
            let guard = DownloadGuard(s.clone());
            std::thread::Builder::new()
                .name("klein-model-download".into())
                .spawn(move || {
                    let s = &guard.0;
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        lightcraft_inpaint::klein_download::download(&dir, &s.cancel, &mut |p| {
                            s.done.store(p.done, Ordering::Relaxed);
                            s.total.store(p.total, Ordering::Relaxed);
                            s.text.lock().unwrap_or_else(std::sync::PoisonError::into_inner).file.clone_from(&p.file);
                        })
                    }));
                    let error = match result {
                        Ok(Ok(())) => {
                            s.finished.store(true, Ordering::SeqCst);
                            None
                        }
                        Ok(Err(e)) => Some(e.to_string()),
                        Err(_) => Some("The Klein model download failed unexpectedly; try again".into()),
                    };
                    s.text.lock().unwrap_or_else(std::sync::PoisonError::into_inner).error = error;
                })
                .map_err(|e| format!("Could not start Klein model download: {e}"))?;
            Ok(true)
        }
    }

    /// Portable releases carry the small removal model beside the executable.
    /// An explicit path is available for developers; working-directory lookup is avoided.
    pub fn removal_model(&self) -> Result<PathBuf, String> {
        if !Self::AVAILABLE {
            return Err("Native removal is not available in this build".into());
        }
        let path = self.removal.clone().ok_or("Cannot locate the bundled removal model")?;
        let metadata = std::fs::metadata(&path).map_err(|_| {
            format!("Removal model is missing: {}. Extract the complete portable package, including its models folder.", path.display())
        })?;
        if !metadata.is_file() || metadata.len() != 29_528_324 {
            return Err("The bundled removal model is incomplete or invalid; extract the portable package again".into());
        }
        Ok(path)
    }

    pub fn installed(&self) -> bool {
        #[cfg(feature = "inpaint")]
        return self.dir.as_deref().is_some_and(lightcraft_inpaint::installed);
        #[cfg(not(feature = "inpaint"))]
        false
    }

    pub fn model_dir(&self) -> Result<PathBuf, String> {
        if !Self::AVAILABLE {
            return Err("Native generative fill is not available in this build".into());
        }
        let dir = self.dir.clone().ok_or("No folder is available for the inpainting model")?;
        if self.installed() {
            Ok(dir)
        } else {
            Err(format!("{NOT_INSTALLED}. Use the Fill model download button, or install the pinned files in {}.", dir.display()))
        }
    }

    pub fn download_status(&self) -> DownloadStatus {
        let s = &self.download;
        // Writers hold this lock only to replace a short status message.
        let (file, error) = s.text.try_lock().map(|t| (t.file.clone(), t.error.clone())).unwrap_or_default();
        DownloadStatus {
            running: s.running.load(Ordering::SeqCst),
            done: s.done.load(Ordering::Relaxed),
            total: s.total.load(Ordering::Relaxed),
            file,
            error,
            finished: s.finished.load(Ordering::SeqCst),
        }
    }

    pub fn cancel_download(&self) -> bool {
        let running = self.download.running.load(Ordering::SeqCst);
        if running {
            self.download.cancel.store(true, Ordering::SeqCst);
        }
        running
    }

    pub fn start_download(&self) -> Result<bool, String> {
        #[cfg(not(feature = "inpaint"))]
        return Err("Native generative fill is not available in this build".into());
        #[cfg(feature = "inpaint")]
        {
            if self.installed() {
                return Ok(false);
            }
            let dir = self.dir.clone().ok_or("No folder is available for the inpainting model")?;
            let s = self.download.clone();
            if s.running.swap(true, Ordering::SeqCst) {
                return Ok(false);
            }
            s.cancel.store(false, Ordering::SeqCst);
            s.done.store(0, Ordering::Relaxed);
            s.total.store(lightcraft_inpaint::MODEL_BYTES, Ordering::Relaxed);
            s.finished.store(false, Ordering::SeqCst);
            *s.text.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = DownloadText::default();
            let guard = DownloadGuard(s.clone());
            std::thread::Builder::new()
                .name("inpaint-model-download".into())
                .spawn(move || {
                    let s = &guard.0;
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        lightcraft_inpaint::download(&dir, &s.cancel, &mut |p| {
                            s.done.store(p.done, Ordering::Relaxed);
                            s.total.store(p.total, Ordering::Relaxed);
                            s.text.lock().unwrap_or_else(std::sync::PoisonError::into_inner).file.clone_from(&p.file);
                        })
                    }));
                    let error = match result {
                        Ok(Ok(())) => {
                            s.finished.store(true, Ordering::SeqCst);
                            None
                        }
                        Ok(Err(e)) => Some(e.to_string()),
                        Err(_) => Some("The model download failed unexpectedly; try again".into()),
                    };
                    s.text.lock().unwrap_or_else(std::sync::PoisonError::into_inner).error = error;
                })
                .map_err(|e| format!("Could not start model download: {e}"))?;
            Ok(true)
        }
    }

    pub fn generation_signals(&self) -> GenerationSignals {
        self.generation_cancel.store(false, Ordering::SeqCst);
        self.generation_progress.update(0, 0);
        self.generation_progress.set_phase("");
        GenerationSignals { cancel: self.generation_cancel.clone(), progress: self.generation_progress.clone() }
    }

    pub fn generation_progress(&self) -> (usize, usize) {
        self.generation_progress.get()
    }

    pub fn generation_phase(&self) -> &'static str {
        self.generation_progress.phase()
    }

    pub fn cancel_generation(&self) {
        self.generation_cancel.store(true, Ordering::SeqCst);
    }
}

#[cfg(all(test, feature = "inpaint"))]
mod klein_manifest_tests {
    use super::*;

    #[test]
    fn engine_model_metadata_matches_pinned_fetch_manifest() {
        assert_eq!(KLEIN_MODEL_BYTES, lightcraft_inpaint::klein_download::MODEL_BYTES);
        assert_eq!(KLEIN_LICENSE_NAME, lightcraft_inpaint::klein_download::LICENSE_NAME);
        assert_eq!(KLEIN_LICENSE_URL, lightcraft_inpaint::klein_download::LICENSE_URL);
    }
}
