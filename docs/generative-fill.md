# Local Generative Fill and Remove

The NVIDIA CUDA build adds native FLUX.2 Klein 4B for prompt-guided Fill and Remove. It runs inside LightCraft through Candle without Python, ComfyUI, a separate service or a cloud photo upload. Klein needs an NVIDIA CUDA GPU; there is no CPU fallback. The earlier MI-GAN Remove and SD 1.5 Fill remain available as legacy native paths, including CPU builds.

## Klein setup and behavior

Klein uses the public Apache-2.0 [FLUX.2-klein-4B checkpoint](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) pinned to revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`. Its five model/tokenizer files plus licence total 15,975,644,852 bytes (about 16 GB). The app-managed setup requires explicit model-licence acknowledgement, supports cancellation and resuming, and checks the pinned size and SHA-256 of every file, including same-sized corrupt files during repair. A CUDA portable package may instead include the same checked files under `models/flux2-klein-4b/` with `-KleinModelDir`; see [Windows CUDA packaging](cuda-windows.md). `LIGHTCRAFT_KLEIN_DIR` can name an existing model folder.

Choose a mask, then **Fill** with a description of the desired replacement or **Remove** to continue the surrounding background. Klein Remove uses a fixed generic background-continuation prompt and a white erased-mask condition; Fill uses the supplied prompt and full source image. Both use four diffusion steps. Klein processes an aligned, aspect-preserving context patch no larger than 1024 × 1024; the engine blends its output through the original soft mask into the full-resolution render (up to 64 MP), leaving unselected pixels unchanged. The result is a separate flattened 8-bit PNG and library photo; the source and develop settings remain intact. Undo removes the generated catalog entry while retaining the PNG on disk for recovery.

The native model components have passed focused checkpoint and 1024-pixel CUDA parity checks. NASA badge and coffee-spoon examples were inspected with specific and generic Remove prompts, and a Fill example was inspected. Those are limited real-photo checks, not a guarantee across masks, subjects or cameras. Production-command checks also covered a 6000 × 4000 output, unchanged unselected pixels and undo. Normal desktop Fill and Remove button checks preserved source files/settings and imported separate results; one-step undo retained both PNGs. These checks establish editing behavior on these fixtures, not general model accuracy.

The commands accept `backend:"klein"` on `photo.generativeFill` and `photo.generativeRemove`; Klein setup uses `generative.model.download {"model":"klein","acknowledged":true}` and `generative.model.cancel {"model":"klein"}`. The `generative.model.status` response reports Klein availability and installation separately. Web builds do not run Klein inference. The older local options below use `backend:"native"` (the legacy command default).

## Legacy native setup

1. Create or select a mask in **Masking**, then open **Remove**.
2. Select the legacy native/MI-GAN option for **Remove**. It is ready when the portable package's `models` folder is present. Select the whole distraction, including its shadow. No Fill model download is needed for this removal path.
3. For legacy SD 1.5 **Fill**, read the CreativeML OpenRAIL-M model licence, check the consent box, and click **Download model**. The pinned checkpoint and tokenizer files total 2.13 GB. Downloading does not start until you click the button; checking consent alone does nothing. Consent is not stored in saved UI settings.
4. The app shows download progress and supports cancellation and resuming partial files. Each file is checked against its pinned size and SHA-256 before installation.
5. Enter a replacement prompt and click **Fill**. For procedural demo photos, enter an output folder too.

Legacy native generation runs on a background thread, with progress and cancellation between steps. The optional `cuda` build selects an NVIDIA GPU and reports initialization failures instead of silently falling back to CPU. Other Windows/Linux builds use CPU; macOS uses Metal when available, otherwise CPU. The panel and `generative.model.status` show the configured inference backend. Preparing/loading the model cannot currently be interrupted midway; a cancellation request is honored when that stage finishes. The model is released when the job finishes. See [Windows CUDA packaging](cuda-windows.md) for the portable runtime layout and GPU target.

A result is a separate 8-bit sRGB PNG and library photo, stacked with the source. The source file and develop settings remain intact. Undo removes the generated catalog entry; the PNG remains recoverable on disk. Native output retains the cropped, oriented photo's dimensions, up to 64 megapixels. The model processes a context patch around the mask at up to 512 pixels, then the engine blends it back into the full-resolution render. Large selections can therefore have less detail than the source. This produces a flattened rendered copy; it does not preserve raw or HDR pixel data in the generated PNG.

Models receive a binary version of the selected mask. Both operations keep the supplied mask by default; `maskMargin` optionally enlarges conditioning by up to 64 model pixels. Final blending always uses the original soft mask, and pixels outside the selection remain identical to the rendered source. MI-GAN uses a centered 512 × 512 canvas with edge padding; the output is cropped back before compositing. Remove is deterministic and does not use diffusion prompts, seeds, steps, guidance or denoise settings (these legacy command parameters are accepted and validated for compatibility). For Fill, the seed controls initial diffusion noise and GPU VAE sampling; Candle's CPU VAE sampling means identical seeds do not currently guarantee identical CPU results. Both models can fail on difficult textures or objects. Source/weight provenance and the developer conversion process are in [migan-model.md](migan-model.md).

Earlier SD 1.5 removal QA (before the MI-GAN replacement) on an RTX 4090 exercised the native commands with 24 steps and seed 42.
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

The replacement MI-GAN core was checked against the author's raw ONNX float output on
gray, coffee and astronaut fixtures. CUDA's tiled depthwise path matched the coffee
reference to maximum absolute error 0.000009. On the RTX 4090, an idle-host test measured
about 7.9 seconds for first inference and 76 milliseconds for repeated network runs in
the same context; these exclude photo rendering, PNG encoding and catalog import.
The app retains that context between edits while releasing job-owned model tensors.
Actual production-command QA preserved all unselected pixels and undo invariants in
640 × 427, 512 × 512, 600 × 400 and 6000 × 4000 cases. The badge removal looked useful;
the launch-tower case left a base artifact, and even a wider spoon mask left smeared
structure. These results are still mixed, so the new removal model does not establish
reliable removal of every distraction.

A portable preflight launched with only Windows system directories on PATH, no external
inference process, the bundled removal model, and an absent Fill model directory. Two
normal Remove button actions completed on CUDA; both preserved the source file/settings,
cleared stale mask selection on the result, and supported one-step undo retaining the PNG.
Observed click-to-catalog times were 12.6 seconds initially and 0.54 seconds on the second
background job (500 ms polling, shared host load, not an isolated benchmark). Only the CUDA
context/kernels are cached between jobs, not the checkpoint tensors.

A separate 6000 × 4000 run used an enlarged version of the coffee fixture to exercise
24 MP allocation, rendering, saving, and compositing. It retained the full dimensions and
all 22,359,819 unselected pixels, with source settings and undo checks passing. This is a
resolution/conservation workload, not evidence of native 24 MP detail or better removal quality.

## Reproducing real-photo QA

The developer-only `ai_photo_qa` engine example runs the production native command on an
explicit photo. It copies that photo into the output folder before importing it; the source
path is never an editing destination. A JSON case (at most 1 MiB) names `source`, optional `backend` (`native` by default or `klein`), `modelDir` (legacy Fill or both Klein operations) or `removalModel` (legacy Remove),
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

Use `photo.generativeRemove` with `maskId` for contextual MI-GAN removal. It uses a fixed 512-pixel canvas and ignores replacement/negative prompts and diffusion settings. Fill uses the replacement prompt. Both support `maskMargin` (0–64 model pixels, default 0) and `dir` for demo output. The margin changes conditioning only, never the final selection. Developers can use `LIGHTCRAFT_MIGAN_MODEL` to provide the pinned converted checkpoint; portable releases resolve it relative to the executable.

Fill's optional native parameters are `negative`, `steps` (1–100, default 24), `guidance` (1–20, default 7.5), `seed`, `edge` (64–512, default 512), and `denoise` (greater than zero and at most one, default 0.99). `floor(steps * denoise)` must select at least one step. Denoise below one starts from noised source latents; one starts from full noise. Remove still validates supplied legacy diffusion parameters for compatibility, though it does not use them. Desktop cancellation uses `photo.generativeCancel`.

## Advanced external backend

The earlier ComfyUI integration remains available to CLI/MCP callers with `backend:"comfy"`, an explicitly provided loopback `endpoint`, and an installed inpainting `checkpoint` filename. Explicit legacy endpoint/checkpoint parameters also select that backend. It requires separately installed ComfyUI and is not the normal desktop workflow. Only loopback IP origins are accepted; remote endpoints and redirects are rejected. This advanced backend sends the rendered photo, mask and prompt to that local process, which may retain inputs/outputs in its folders. Its output edge limit is 2048 pixels.
