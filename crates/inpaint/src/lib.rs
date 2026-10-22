//! Local MI-GAN removal and Stable Diffusion 1.5 prompt-based inpainting.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

#[cfg(not(target_arch = "wasm32"))]
mod model;
#[cfg(not(target_arch = "wasm32"))]
pub use model::*;
#[cfg(not(target_arch = "wasm32"))]
mod download;
#[cfg(not(target_arch = "wasm32"))]
pub mod klein;
#[cfg(not(target_arch = "wasm32"))]
pub mod klein_download;
#[cfg(not(target_arch = "wasm32"))]
pub mod klein_sampling;
#[cfg(not(target_arch = "wasm32"))]
pub mod klein_text;
#[cfg(not(target_arch = "wasm32"))]
pub mod klein_tokenizer;
#[cfg(not(target_arch = "wasm32"))]
pub mod klein_transformer;
#[cfg(not(target_arch = "wasm32"))]
pub mod klein_vae;
#[cfg(not(target_arch = "wasm32"))]
pub mod migan;
#[cfg(not(target_arch = "wasm32"))]
mod tokenizer;
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub mod weights;
#[cfg(not(target_arch = "wasm32"))]
pub use download::{LICENSE_NAME, LICENSE_URL, MODEL_BYTES, MODEL_FILES, MODEL_REVISION, ModelFile, download, installed};
