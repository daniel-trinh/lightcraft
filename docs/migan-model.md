# MI-GAN reference model provenance

This page records the provenance and maintainer conversion process for the MI-GAN checkpoint used by native Remove. The portable CUDA packager includes the converted safetensors file and both upstream MIT licences; the product runs the network in Rust/Candle, without an ONNX runtime.

The upstream implementation is [Picsart AI Research's MI-GAN repository](https://github.com/Picsart-AI-Research/MI-GAN), pinned at code revision [`2b793c5ece43f4253e32d4afc257120a5deed6f5`](https://github.com/Picsart-AI-Research/MI-GAN/tree/2b793c5ece43f4253e32d4afc257120a5deed6f5). The model source is the `migan.onnx` file from [Hugging Face revision `406830d0fa60666da0071c342ad2fbc8f30c5c64`](https://huggingface.co/andraniksargsyan/migan/tree/406830d0fa60666da0071c342ad2fbc8f30c5c64). The pinned input is 29,546,882 bytes with SHA-256 `593eba0b7e04730f1b61c0a3cbca68d97d8d6a7ff5c6a44a7b9d7fcd880fc5ae`.

The upstream code and model weights have separate upstream license files. Both are MIT; retain them independently as [MI-GAN code license](licenses/MI-GAN-LICENSE.txt) and [MI-GAN weights license](licenses/MI-GAN-LICENSE-WEIGHTS.txt). The weights license applies to the checkpoint and derived tensor file; the code license applies to copied or adapted upstream code.

## Reproduce the conversion

The converter is a maintainer tool, not an end-user runtime dependency. Use Python with NumPy and ONNX 1.19.x (the recorded conversion environment used ONNX 1.19.0). It reads only the supplied local ONNX file; it does not download, execute, or contact a model service. It does not use PyTorch or ONNX Runtime.

```powershell
python -m pip install numpy onnx==1.19.0
python tools/convert-migan.py work/migan-reference/migan.onnx work/migan-reference/migan-canonical-a.safetensors
python tools/convert-migan.py work/migan-reference/migan.onnx work/migan-reference/migan-canonical-b.safetensors
```

The source is size- and SHA-256-verified before ONNX parsing. Conversion selects only nonempty F32 and I64 graph initializers and tensor-valued `Constant` nodes, retaining each ONNX value name. Tensor names are serialized in lexical order, with fixed field order, compact JSON, an eight-byte little-endian header length, and space padding to an eight-byte boundary. Tensor payloads are written as contiguous little-endian arrays. The metadata records the pinned ONNX digest and model revision. Output files are never overwritten; the converter writes to a unique sibling staging directory, validates the expected payload size, and publishes the completed file atomically. A failed conversion leaves no final output and does not remove the supplied input or existing files.

The canonical conversion is 29,528,324 bytes with SHA-256 `365c3680e7e33b140761d11a0e62d7ed596572d036bbc3e670540c2baaf9c845`. It was produced twice with identical bytes, and all 525 tensor names, shapes, dtypes, and values match the earlier reference conversion. The earlier reference file is 29,528,092 bytes with SHA-256 `67a51e31d06eb5cee323dcfe86daf1f39f745a345b0d30fe4962137584706878`; its traversal order differs from the lexical order required by this canonical writer.

The source ONNX container is a maintainer input. User packages contain only the converted tensors, licences and provenance; users do not run the converter or install Python.
