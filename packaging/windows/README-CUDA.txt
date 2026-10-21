LightCraft for Windows — local NVIDIA AI editing

Extract the entire ZIP, keep its files together, and launch lightcraft.exe.
You do not need to start a server or install Python, ComfyUI, or the CUDA toolkit.
An NVIDIA graphics driver is required. This package targets the GPU architecture
recorded in SHA256-MANIFEST.json; the default build targets the RTX 4090 (8.9).

Fill and Remove
1. Import a photo and create a mask around the area to change.
2. Open Remove and choose Remove to reconstruct the background. Its small
   MIT-licensed model is included in the models folder; no download is needed.
3. For prompt-based Fill, read the model licence, then choose Download model.
   The one-time Fill download is approximately 2.13 GB and can be resumed.
   Enter a replacement prompt and choose Fill. Each result appears as a new photo
   beside its source. Neither operation uploads your photo.

AI masks
Object selection can use the Apache-2.0 SAM 1 model with clicks to include and
Alt-clicks to exclude. If models/sam1 is included, it is ready on a fresh launch;
otherwise select SAM 1 in the Masking panel and choose Download Object model.
The one-time download is 375 MB. No Hugging Face account is required for SAM 1.
Existing preferences retain the chosen Object model.

Describe and the AI presets (Sky, People, Subject, Background) use SAM 3.
Subject and Background ask what to select (for example, person, rocket or cup).
Background selects everything outside the described subject.
SAM 3 model access is managed by Meta at https://huggingface.co/facebook/sam3.
Once your access is approved, download model.safetensors, vocab.json, and
merges.txt into one folder. In the AI model dialog choose Choose Model Folder.
The app remembers that location. Meta approval is separate from installing this app.

Once model setup is complete, inference runs on your computer. Photos are not
uploaded to an AI service. Generative results can contain mistakes: inspect each
result before exporting. Your source photo is retained. Undo removes a generated
photo from the catalog while keeping its PNG on disk for recovery.

Keep the bundled licence files with this package. The included MI-GAN removal
weights have an MIT licence in models/licenses. Bundled SAM 1 weights have an
Apache-2.0 licence there too. Fill and SAM 3 have their own licences and are
obtained separately during setup.
