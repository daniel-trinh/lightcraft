# Experimental Generative Fill and Remove

The current implementation is a development prototype with an external ComfyUI backend. It is not the intended end-user setup: the app does not yet install models, launch inference, or manage the backend lifecycle. App-managed inference is still required before this feature can be considered ready for ordinary use. The repository's pure-Rust requirement must be resolved before bundling a Python/PyTorch runtime.

LightCraft can send a selected mask and a rendered copy of the photo to a **local ComfyUI** instance. The model's result is blended over the mask and imported as a separate PNG next to the original. The original file and its develop settings remain intact. Undo removes the generated photo from the catalog; the generated PNG remains on disk so it can be recovered.

This feature needs a ComfyUI installation and an inpainting checkpoint already installed in its `models/checkpoints` folder. LightCraft does not download a diffusion model or call a cloud service. The endpoint must be a loopback IP such as `http://127.0.0.1:8188`; remote hosts are rejected. Photo pixels, mask, and prompt are sent to that local process. ComfyUI may retain uploaded input files and saved output files in its own folders; manage those through ComfyUI.

In the desktop app, create or select a mask in **Masking**, open **Remove**, enter the local endpoint and exact checkpoint filename, then click **Fill** or **Remove**. For a demo photo, also enter an output folder. The fields are saved in the UI settings. Fill uses the replacement prompt; Remove uses a background-matching prompt and adds removal terms to the negative prompt. Generation runs in the background. The result is a new library photo stacked with the source. Desktop-generated copies have a long edge of at most 1024 pixels.

For CLI, control channel, or MCP, use the same engine command:

```json
{"id":"photo.generativeFill","params":{"maskId":1,"endpoint":"http://127.0.0.1:8188","checkpoint":"your-inpainting-model.safetensors","prompt":"clean natural background"}}
```

Use `photo.generativeRemove` with the same parameters for contextual removal; it supplies a background-matching prompt. `negative`, `steps` (1–100), `denoise` (0.01–1), `seed`, and `edge` (64–2048, default 1024) are optional. A generated demo photo needs `dir` for the output destination. The command uses stock ComfyUI nodes: `LoadImage`, `LoadImageMask`, `CheckpointLoaderSimple`, `CLIPTextEncode`, `InpaintModelConditioning`, `KSampler`, `VAEDecode`, and `SaveImage`. Checkpoint quality and prompt choice determine removal quality; an ordinary checkpoint may produce poor inpainting. The workflow uses 8-bit sRGB at at most 2048 pixels on the long edge. It pads model input to an eight-pixel multiple, crops the result back to the rendered dimensions, and changes pixels only inside the mask. It does not preserve original raw resolution or HDR depth in the generated copy.
