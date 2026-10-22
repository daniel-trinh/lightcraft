# lightcraft-inpaint

Native SD 1.5 latent inpainting through Candle, without Python or a separate inference server. This L3 crate owns model-file installation, CLIP tokenization, bounded safetensors loading, and inference. The engine owns rendering, masks, compositing, output files, and catalog edits.

The public checkpoint is pinned by revision, size, and SHA-256 in `src/download.rs`. Weights remain outside the repository and have a separate CreativeML OpenRAIL-M licence. The caller must obtain licence acknowledgement before starting a download. Downloads support cancellation and resuming.

`Model::generate` accepts RGB bytes and a binary mask with white indicating replacement, at dimensions divisible by eight between 64 and 512. It reports diffusion-step progress and checks cancellation between steps. The caller must preserve unselected pixels when compositing the result. CPU inference uses F32; macOS uses Metal F16 when available. The optional `cuda` feature selects NVIDIA CUDA F16 and reports initialization errors rather than silently falling back to CPU. Set `LIGHTCRAFT_AI_DEVICE=cpu` only to explicitly select CPU inference for diagnostics. Model loading itself cannot be interrupted.

Run `cargo test -p lightcraft-inpaint` for bounded loading, tokenization, conditioning, and validation tests. The engine's ignored `native_fill_and_remove_with_pinned_checkpoint` test exercises both editing commands with real weights; set `LIGHTCRAFT_INPAINT_TEST_MODEL` and `LIGHTCRAFT_INPAINT_TEST_OUTPUT` to explicit directories before running it. It uses procedural demo images and does not download models or upload images.

Desktop setup and limitations: [Local Generative Fill and Remove](../../docs/generative-fill.md).

The experimental `klein_tokenizer`, `klein_text`, `klein_transformer` and `klein_vae` modules are native building
blocks for FLUX.2 Klein 4B. They are not connected to the desktop Fill/Remove
commands yet. They read the pinned public Diffusers BF16 weights through bounded
positional loaders (including two text-encoder shards); no Python or service is
used by these Rust modules. The text encoder captures Qwen3 layers 9/18/27 and
omits unused language-model layers. Transformer attention uses bounded query
tiles, and both modules check cancellation between layers. Existing models retain
their 128 MiB per-tensor limit; explicit Klein policies allow the pinned fused
projection and token embedding, with hard limits of 192 MiB and 768 MiB.

The developer-only `klein_text_parity` example takes a local model directory and
reference safetensors file, with optional `--cuda`. It reports per-layer feature
errors; it does not download models. Real-checkpoint CUDA checks covered 64- and
512-token text features and a 1024-pixel image's captured first transformer call.
These component comparisons do not establish full native image-generation quality.

The tokenizer matches captured raw-token and chat-template vectors, with NFC,
case/whitespace preservation, Qwen BPE and right padding. The F32 VAE was compared
with Diffusers at 128 and 1024 pixels on CUDA. It rejects CPU inputs above 64 pixels:
the Candle CPU convolution path diverges at larger sizes. This restriction is
explicit; the intended Klein product path is CUDA. `klein_vae_parity` reproduces
the component comparison from local checkpoint and fixture files. No full Klein
generation path or desktop integration is provided yet. `klein_download` supplies
the managed-installation primitives: six required runtime files (15.98 GB) pinned
by size and SHA-256, with resume, progress and cancellation through
`lightcraft-fetch`. It is not connected to desktop model setup yet. No gated
Hugging Face account or token is required for these Apache-2.0 model files.

`klein_sampling` implements the shifted Euler schedule, latent packing, position
IDs, bilinear mask reduction and inpaint preservation blend. Captured reference
tests matched the schedule/IDs/masks and BF16 step intermediates exactly. Keeping
the reference dtype order matters: BF16 products round before the later F32
addition in the Euler update. These helpers are not yet a desktop generation job.
