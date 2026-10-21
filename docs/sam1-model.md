# SAM 1 Object model provenance

The optional click-based Object backend runs Meta's SAM 1 ViT-B in Rust using
[Candle 0.9.2](https://github.com/huggingface/candle/tree/0.9.2/candle-transformers/src/models/segment_anything).
It does not provide text prompts or SAM 3 semantic presets.

The model is the F32 safetensors checkpoint published by
[Kornia](https://huggingface.co/kornia/sam/tree/b6e8348387e0296715f9e4a0a766b59c19a691fd),
pinned at revision `b6e8348387e0296715f9e4a0a766b59c19a691fd`:

| Item | Value |
|---|---|
| File | `sam_vit_b_01ec64.safetensors` |
| Bytes | 374,979,376 |
| SHA-256 | `fbf4c1ad844a2e9b867d89dc04a61f3ece19c57ebb7d8fcb6da92062a291d839` |
| Original Meta checkpoint | `sam_vit_b_01ec64.pth` |
| Original SHA-256 | `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` |
| Licence | Apache-2.0, [bundled text](licenses/SAM1-LICENSE.txt) |

Kornia's pinned model card records unchanged tensor keys, shapes, dtypes and bit values
from Meta's original checkpoint, and bitwise-identical image-encoder results in its own
conversion verification. This statement describes upstream conversion evidence; it does not
establish LightCraft inference parity. The shared upstream token storage is duplicated in
safetensors without changing its values. LightCraft does not convert or unpickle model files.
The separately licensed SA-1B dataset is not included.

The app's explicit **Download Object model (375 MB)** action downloads this pinned file,
checks size and SHA-256, and publishes it only after verification. Cancellation retains a
resumable partial. No account or Meta access-request form is needed for this SAM 1 checkpoint.
Model download transfers weights to the app; photos stay on the computer during inference.

Maintainers may supply the exact file as `-Sam1Model` to
`packaging/windows/package-cuda.ps1`. The packager verifies size and SHA-256 before copying it
to `models/sam1/sam_vit_b_01ec64.safetensors`, includes this provenance and the Apache licence,
and records its checksum in the portable manifest. On a first launch with no saved UI
preferences and an installed SAM 1 model, Object selection starts on SAM 1. Existing settings
and previously saved Object masks retain their backend. Describe and AI presets use SAM 3.

Real-checkpoint inference passed on an RTX 4090 with square, portrait and landscape photos:
positive and negative clicks, detail refinement, stored-mask rendering, source preservation
and undo. A normal desktop session also verified repair of a same-sized corrupt checkpoint
and mask-based native Remove. These checks establish working inference, not reference-model
numerical parity or general segmentation accuracy.
See [AI masks](ai-masks.md) for setup and the distinction between the two backends.
