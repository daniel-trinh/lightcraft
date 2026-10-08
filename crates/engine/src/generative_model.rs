//! App-managed native inpainting model setup, progress and cancellation.
//! No service or external inference process is needed.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

pub const NOT_INSTALLED: &str = "The inpainting model is not installed";
pub const LICENSE_NAME: &str = "CreativeML OpenRAIL-M";
pub const LICENSE_URL: &str = "https://huggingface.co/spaces/CompVis/stable-diffusion-license";

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
}

impl GenerationProgress {
    pub fn update(&self, done: usize, total: usize) {
        self.total.store(total, Ordering::Relaxed);
        self.done.store(done.min(total), Ordering::Relaxed);
    }

    pub fn get(&self) -> (usize, usize) {
        (self.done.load(Ordering::Relaxed), self.total.load(Ordering::Relaxed))
    }
}

/// Signals owned by a background generation job; cancellation is checked between steps.
pub struct GenerationSignals {
    pub cancel: Arc<AtomicBool>,
    pub progress: Arc<GenerationProgress>,
}

pub struct Generator {
    pub dir: Option<PathBuf>,
    download: Arc<DownloadShared>,
    generation_cancel: Arc<AtomicBool>,
    generation_progress: Arc<GenerationProgress>,
}

impl Default for Generator {
    fn default() -> Self {
        Self {
            dir: std::env::var_os("LIGHTCRAFT_INPAINT_DIR")
                .map(PathBuf::from)
                .or_else(|| crate::camera_profiles::config_dir().map(|d| d.join("models/sd15-inpaint"))),
            download: Arc::default(),
            generation_cancel: Arc::default(),
            generation_progress: Arc::default(),
        }
    }
}

impl Generator {
    pub const AVAILABLE: bool = cfg!(feature = "inpaint");

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
            Err(format!("{NOT_INSTALLED}. Use the Download model button in Remove, or install the pinned files in {}.", dir.display()))
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
        GenerationSignals { cancel: self.generation_cancel.clone(), progress: self.generation_progress.clone() }
    }

    pub fn generation_progress(&self) -> (usize, usize) {
        self.generation_progress.get()
    }

    pub fn cancel_generation(&self) {
        self.generation_cancel.store(true, Ordering::SeqCst);
    }
}
