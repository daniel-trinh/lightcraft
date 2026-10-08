# lightcraft-inpaint

Native SD 1.5 latent inpainting through Candle, without Python or a separate inference server. This L3 crate owns model-file installation, CLIP tokenization, bounded safetensors loading, and inference. The engine owns rendering, masks, compositing, output files, and catalog edits.

The public checkpoint is pinned by revision, size, and SHA-256 in `src/download.rs`. Weights remain outside the repository and have a separate CreativeML OpenRAIL-M licence. The caller must obtain licence acknowledgement before starting a download. Downloads support cancellation and resuming.

`Model::generate` accepts RGB bytes and a binary mask with white indicating replacement, at dimensions divisible by eight between 64 and 512. It reports diffusion-step progress and checks cancellation between steps. The caller must preserve unselected pixels when compositing the result. CPU inference uses F32; macOS uses Metal F16 when available. Windows GPUs are not currently used. Model loading itself cannot be interrupted.

Run `cargo test -p lightcraft-inpaint` for bounded loading, tokenization, conditioning, and validation tests. The engine's ignored `native_fill_and_remove_with_pinned_checkpoint` test exercises both editing commands with real weights; set `LIGHTCRAFT_INPAINT_TEST_MODEL` and `LIGHTCRAFT_INPAINT_TEST_OUTPUT` to explicit directories before running it. It uses procedural demo images and does not download models or upload images.

Desktop setup and limitations: [Local Generative Fill and Remove](../../docs/generative-fill.md).
