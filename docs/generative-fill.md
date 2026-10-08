# Local Generative Fill and Remove

This experimental fork runs SD 1.5 inpainting inside Lightcraft through Candle in pure Rust. The app handles model setup and inference. No ComfyUI, Python runtime, separate server, or cloud image upload is required for the desktop workflow. The model is downloaded once; generation then reads local files and runs locally.

## Desktop setup

1. Create or select a mask in **Masking**, then open **Remove**.
2. Read the CreativeML OpenRAIL-M model licence, check the consent box, and click **Download model**. The pinned checkpoint and tokenizer files total 2.13 GB. Downloading does not start until you click the button; checking consent alone does nothing. Consent is not stored in saved UI settings.
3. The app shows download progress and supports cancellation and resuming partial files. Each file is checked against its pinned size and SHA-256 before installation.
4. Enter a replacement prompt and click **Fill**, or click **Remove** to use a background-matching prompt. For procedural demo photos, enter an output folder too.

Generation runs on a background thread, with progress and cancellation between steps. On Windows and Linux it currently uses the CPU; on macOS it uses Metal when available, otherwise CPU. Preparing/loading the model cannot currently be interrupted midway; a cancellation request is honored when that stage finishes. The model is released when the job finishes. Windows RTX GPUs are not yet used by this backend.

A result is a separate 8-bit sRGB PNG and library photo, stacked with the source. The source file and develop settings remain intact. Undo removes the generated catalog entry; the PNG remains recoverable on disk. The generated copy is at most 512 pixels on its long edge, with the source's crop and orientation applied. This is an experimental flattened-copy workflow, not full-resolution raw/HDR editing.

The model receives a binary version of the selected mask. Final blending uses the original soft mask, and pixels outside the selection remain identical to the rendered source. Model padding is cropped away before saving. Prompt/model quality determines the generated content; inpainting can hallucinate or fail to remove a distraction cleanly. The seed controls initial diffusion noise, but Candle's CPU VAE sampling means identical seeds do not currently guarantee identical results.

Developer QA exercised both native editing commands with the pinned checkpoint on a cropped 171 × 256 procedural demo image, 16 steps, and seed 42. Fill took 247 seconds while the workspace suite was also running; Remove took 82 seconds. These are observed CPU timings, not an isolated performance benchmark. The test verified dimensions, identical unselected pixels, unchanged source settings, catalog import, and one-step undo with the generated file retained. Visual inspection found rough content and visible blending artifacts; this is not evidence of production image quality or real-photo coverage. All seven `cargo xtask ci` checks also pass.

## Model files and licence

The pinned model revision is `8a4288a76071f7280aedbdb3253bdb9e9d5d84bb` of [stable-diffusion-v1-5/stable-diffusion-inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting). The checkpoint is open weights under [CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license), a separate licence from Lightcraft's source. Weights are never committed or distributed in this fork's releases.

The app installs into `models/sd15-inpaint/` under its settings folder, or the directory named by `LIGHTCRAFT_INPAINT_DIR`. Native inference is enabled in the desktop and CLI builds by the engine's `inpaint` feature. Web builds do not run diffusion inference, but imported PNG results remain usable.

## Commands

CLI, control channel and MCP use the same engine commands. First inspect setup with `generative.model.status`. `generative.model.download {"acknowledged":true}` requires explicit model-licence acknowledgement; scripts must only supply it after the user agreed. `generative.model.cancel` stops setup.

```json
{"id":"photo.generativeFill","params":{"maskId":1,"prompt":"yellow flowers growing in grass"}}
```

Use `photo.generativeRemove` with `maskId` for contextual removal. Optional native parameters are `negative`, `steps` (1-100, default 24), `guidance` (1-20, default 7.5), `seed`, `edge` (64-512, default 512), and `dir` for demo output. Native inference currently requires `denoise:1`. Desktop cancellation uses `photo.generativeCancel`.

## Advanced external backend

The earlier ComfyUI integration remains available to CLI/MCP callers with `backend:"comfy"`, an explicitly provided loopback `endpoint`, and an installed inpainting `checkpoint` filename. Explicit legacy endpoint/checkpoint parameters also select that backend. It requires separately installed ComfyUI and is not the normal desktop workflow. Only loopback IP origins are accepted; remote endpoints and redirects are rejected. This advanced backend sends the rendered photo, mask and prompt to that local process, which may retain inputs/outputs in its folders. Its output edge limit is 2048 pixels.
