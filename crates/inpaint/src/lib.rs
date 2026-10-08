//! Local Stable Diffusion 1.5 inpainting. The checkpoint is downloaded separately by the user.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

#[cfg(not(target_arch = "wasm32"))]
mod model;
#[cfg(not(target_arch = "wasm32"))]
pub use model::*;
#[cfg(not(target_arch = "wasm32"))]
mod download;
#[cfg(not(target_arch = "wasm32"))]
mod tokenizer;
#[cfg(not(target_arch = "wasm32"))]
mod weights;
#[cfg(not(target_arch = "wasm32"))]
pub use download::{LICENSE_NAME, LICENSE_URL, MODEL_BYTES, MODEL_FILES, MODEL_REVISION, ModelFile, download, installed};
