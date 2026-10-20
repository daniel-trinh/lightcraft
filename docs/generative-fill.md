# Local Generative Fill and Remove

This fork runs SD 1.5 inpainting inside Lightcraft through Candle. The app handles model setup and inference. No ComfyUI, Python runtime, separate server, or cloud image upload is required for the desktop workflow. The model is downloaded once; generation then reads local files and runs locally. Optional NVIDIA builds use CUDA kernels and bundled runtime libraries; CPU builds use Rust inference.

## Desktop setup

1. Create or select a mask in **Masking**, then open **Remove**.
2. Read the CreativeML OpenRAIL-M model licence, check the consent box, and click **Download model**. The pinned checkpoint and tokenizer files total 2.13 GB. Downloading does not start until you click the button; checking consent alone does nothing. Consent is not stored in saved UI settings.
3. The app shows download progress and supports cancellation and resuming partial files. Each file is checked against its pinned size and SHA-256 before installation.
4. Enter a replacement prompt and click **Fill**, or click **Remove** to use a background-matching prompt. For procedural demo photos, enter an output folder too.

Generation runs on a background thread, with progress and cancellation between steps. The optional `cuda` build selects an NVIDIA GPU and reports initialization failures instead of silently falling back to CPU. Other Windows/Linux builds use CPU; macOS uses Metal when available, otherwise CPU. The panel and `generative.model.status` show the configured inference backend. Preparing/loading the model cannot currently be interrupted midway; a cancellation request is honored when that stage finishes. The model is released when the job finishes. See [Windows CUDA packaging](cuda-windows.md) for the portable runtime layout and GPU target.

A result is a separate 8-bit sRGB PNG and library photo, stacked with the source. The source file and develop settings remain intact. Undo removes the generated catalog entry; the PNG remains recoverable on disk. Native output retains the cropped, oriented photo's dimensions, up to 64 megapixels. The model processes a context patch around the mask at up to 512 pixels, then the engine blends it back into the full-resolution render. Large selections can therefore have less detail than the source. This produces a flattened rendered copy; it does not preserve raw or HDR pixel data in the generated PNG.

The model receives a binary version of the selected mask. Remove expands that internal mask by 32 model pixels so the model cannot see and recreate the object's edges; Fill keeps the supplied mask. Final blending always uses the original soft mask, and pixels outside the selection remain identical to the rendered source. Model padding is cropped away before saving. Prompt/model quality determines the generated content; inpainting can hallucinate or fail to remove a distraction cleanly. The seed controls initial diffusion noise and GPU VAE sampling; Candle's CPU VAE sampling means identical seeds do not currently guarantee identical CPU results.

Real-photo QA on an RTX 4090 exercised the native commands with 24 steps and seed 42.
The 512 × 512 NASA astronaut photo produced a plausible blue fabric patch with Fill;
Remove replaced its suit badge with fabric after adding the internal mask margin. Both
preserved all 256,220 unselected pixels. The 600 × 400 CC0 coffee photo preserved all
223,595 unselected pixels, but spoon removal invented an extra cup-like structure and
failed visual inspection. Generation took about 19–20 seconds per case while other workspace
checks were running, including model loading; these are observations, not isolated benchmarks.
The harness also checked source settings, catalog import, and one-step undo with PNG retention.
These mixed results do not establish reliable general-purpose removal quality. An offline
Diffusers reference using the same checkpoint also recreated a circular object with the tight
badge mask; its preprocessing was approximate, so this is supporting evidence rather than
tensor-level equivalence. Earlier procedural CPU checks remain useful for editing invariants,
not for judging photographic quality.

A separate 6000 × 4000 run used an enlarged version of the coffee fixture to exercise
24 MP allocation, rendering, saving, and compositing. It retained the full dimensions and
all 22,359,819 unselected pixels, with source settings and undo checks passing. This is a
resolution/conservation workload, not evidence of native 24 MP detail or better removal quality.

## Reproducing real-photo QA

The developer-only `ai_photo_qa` engine example runs the production native command on an
explicit photo. It copies that photo into the output folder before importing it; the source
path is never an editing destination. A JSON case (at most 1 MiB) names `source`, `modelDir`,
`outputDir`, `command` (`photo.generativeFill` or `photo.generativeRemove`), `mask` (ordinary
`mask.add` parameters), and optional `params` (prompt, negative, steps, guidance, seed, edge,
denoise, maskMargin). AI masks are deliberately excluded so this test does not require gated SAM weights.
For a brush mask, supply `brushStrokes`, an array of ordinary `mask.brushStroke` parameter
objects (at most 32 strokes and 4096 points in total). The harness supplies the mask ID.
Set `previewOnly` to true to save and inspect the source/mask before running inference;
preview reports are labelled `preview`, never `passed`.

```json
{
  "source": "C:/photos/test.png",
  "modelDir": "C:/models/sd15-inpaint",
  "outputDir": "C:/qa/fill",
  "command": "photo.generativeFill",
  "mask": {"kind":"radial", "center":[0.5,0.5], "rx":0.08, "ry":0.08, "feather":15},
  "params": {"prompt":"a red apple on a wooden table, photograph", "seed":42}
}
```

Run `cargo run -p lightcraft-engine --example ai_photo_qa --features inpaint -- case.json`
for the platform backend, or use `--features inpaint,cuda` in a configured CUDA build.
CUDA QA refuses the diagnostic CPU override. The example writes expected source/mask PNGs,
the generated PNG, and a JSON report checking native dimensions, identical pixels outside
the mask, changed selected pixels, unchanged source settings, and one-step undo retaining
the generated file. Passing these checks proves editing invariants, not that the model
removed the right object or produced convincing content; inspect the images as well.

## Model files and licence

The pinned model revision is `8a4288a76071f7280aedbdb3253bdb9e9d5d84bb` of [stable-diffusion-v1-5/stable-diffusion-inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting). The checkpoint is open weights under [CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license), a separate licence from Lightcraft's source. Weights are never committed or distributed in this fork's releases.

The app installs into `models/sd15-inpaint/` under its settings folder, or the directory named by `LIGHTCRAFT_INPAINT_DIR`. Native inference is enabled in the desktop and CLI builds by the engine's `inpaint` feature. Web builds do not run diffusion inference, but imported PNG results remain usable.

## Commands

CLI, control channel and MCP use the same engine commands. First inspect setup with `generative.model.status`. `generative.model.download {"acknowledged":true}` requires explicit model-licence acknowledgement; scripts must only supply it after the user agreed. `generative.model.cancel` stops setup.

```json
{"id":"photo.generativeFill","params":{"maskId":1,"prompt":"yellow flowers growing in grass"}}
```

Use `photo.generativeRemove` with `maskId` for contextual removal. Remove uses its background-matching prompt and ignores a supplied replacement prompt; Fill uses that prompt. Optional native parameters are `negative`, `steps` (1-100, default 24), `guidance` (1-20, default 7.5), `seed`, `edge` (64-512, default 512), `denoise` (greater than zero and at most one, default 0.99), `maskMargin` (0-64 model pixels, default 32 for Remove and 0 for Fill), and `dir` for demo output. The margin changes conditioning only, never the final selection. `floor(steps * denoise)` must select at least one step. Denoise below one starts from noised source latents; one starts from full noise. Desktop cancellation uses `photo.generativeCancel`.

## Advanced external backend

The earlier ComfyUI integration remains available to CLI/MCP callers with `backend:"comfy"`, an explicitly provided loopback `endpoint`, and an installed inpainting `checkpoint` filename. Explicit legacy endpoint/checkpoint parameters also select that backend. It requires separately installed ComfyUI and is not the normal desktop workflow. Only loopback IP origins are accepted; remote endpoints and redirects are rejected. This advanced backend sends the rendered photo, mask and prompt to that local process, which may retain inputs/outputs in its folders. Its output edge limit is 2048 pixels.
