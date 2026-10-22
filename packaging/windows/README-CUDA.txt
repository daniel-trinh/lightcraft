LightCraft for Windows — local NVIDIA AI editing

Extract the entire ZIP, keep its files together, and launch lightcraft.exe.
You do not need to start a server or install Python, ComfyUI, or the CUDA toolkit.
An NVIDIA graphics driver is required. This package targets the GPU architecture
recorded in SHA256-MANIFEST.json; the default build targets the RTX 4090 (8.9).

Fill and Remove
1. Import a photo and create a mask around the area to change.
2. Open Remove. The NVIDIA CUDA build uses Klein by default. If model setup is
   needed, review the Apache-2.0 licence and choose Download model (about 16 GB).
   Downloads resume and verify pinned files. If models/flux2-klein-4b is already
   included in this package, Fill and Remove are ready without a download.
3. Fill takes your replacement description. Remove uses a fixed background
   continuation prompt; select the whole distraction, including its shadow.
   Each result appears as a new photo beside its source. The original is kept.
4. The earlier bundled MIT-licensed MI-GAN Remove and separately downloaded
   SD 1.5 Fill remain available under Compatibility options.

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

Keep the bundled licence files with this package. MI-GAN's MIT licence and any
bundled SAM 1 or Klein Apache-2.0 licences are under models/licenses. The
legacy SD 1.5 Fill and separately obtained SAM 3 have their own licences.
