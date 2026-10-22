# Standalone CUDA build for Windows

`packaging/windows/package-cuda.ps1` builds the x64 desktop app with the Rust `cuda` feature and
creates a portable folder plus ZIP. The target computer does not need Python, ComfyUI, a CUDA
toolkit, or a separately launched service. The CUDA runtime DLLs are shipped beside
`lightcraft.exe`; this is a portable folder, not a physically single-file executable. Users
launch `lightcraft.exe` from the extracted folder. The MIT-licensed MI-GAN removal model is
included under `models/`; the legacy SD 1.5 Fill model is downloaded by the app when requested.
The CUDA app also supports native FLUX.2 Klein 4B Fill and Remove. Its pinned Apache-2.0
public model is approximately 16 GB and is either downloaded with explicit licence
acknowledgement in the app or bundled with the optional `-KleinModelDir` argument.
The optional `-Sam1Model` argument also bundles the ungated click-selection checkpoint and
its Apache-2.0 licence. Describe and semantic presets continue to use separately obtained SAM 3.

## Build

Install Rust with the `x86_64-pc-windows-msvc` target and Visual Studio C++ x64 Build Tools with
the Windows SDK. The script imports the Visual Studio compiler
environment for its own process when needed; it does not install tools or alter the global
`PATH`. It scopes `CUDA_ROOT`, `CUDA_PATH`, the CUDA `bin` directory, `NVCC_CCBIN`, and
`CUDA_COMPUTE_CAP` (default `89`) to the build process, and caps Cargo/Rayon parallelism at four
workers.

The default GPU target is compute capability 8.9, matching the RTX 4090 used for
this fork's validation. It does not support older RTX 20/30 hardware. Maintainers
building for another GPU must set `CUDA_COMPUTE_CAP` to a supported target before
building and validate inference on that hardware. The chosen target is recorded
in the package manifest; a successful build alone does not prove GPU compatibility.

Provide a complete CUDA 12.8.1 SDK/runtime tree through `-CudaRoot`. It must include the six
runtime DLLs (`cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`, `curand64_10.dll`,
`nvrtc64_120_0.dll`, and `nvrtc-builtins64_128.dll`) and the NVIDIA CUDA Toolkit EULA in
`LICENSE`. Additional component licenses beneath the SDK's `licenses/` directory are copied
into the package with their relative paths. Missing runtime files stop packaging with the
specific dependency named; dependency discovery does not use `PATH`. Runtime files and their
SHA-256 hashes are recorded in `SHA256-MANIFEST.json`, along with inspected DLL dependencies.
The application links its C runtime statically. Packaging checks every executable and runtime
DLL with `dumpbin`: dependencies must be bundled, Windows system libraries, or the installed
NVIDIA driver's `nvcuda.dll`. An unexpected dependency stops packaging even if it happens to
be installed on the build machine. LightCraft source license files are included.

Release builds also require `CRAFT_FONTS_DIR` to point to a local checkout of
[`storytold/craft-fonts`](https://github.com/storytold/craft-fonts); the font files are embedded
at build time and license texts found in the craft-fonts checkout are copied under
`FONT-LICENSES/`. This is a maintainer build input, not a user dependency.

Example, from the LightCraft repository root:

```powershell
$env:CRAFT_FONTS_DIR = 'C:\src\craft-fonts'
pwsh packaging/windows/package-cuda.ps1 `
  -CudaRoot 'C:\src\cuda-sdk' `
  -MiganModel 'C:\src\models\migan-f32.safetensors' `
  -Sam1Model 'C:\src\models\sam_vit_b_01ec64.safetensors' `
  -KleinModelDir 'C:\src\models\flux2-klein-4b' `
  -OutDir 'C:\releases\lightcraft-cuda-1.0.0'
```

`OutDir` must not already exist. The script builds first, stages into a unique temporary
directory next to the destination, then moves the completed folder into place. It does not
delete or replace an existing destination. The output contains `LightCraft-CUDA-portable/` and
`LightCraft-CUDA-portable.zip`.

Pass `-MiganModel` with the pinned, converted removal checkpoint described in
[migan-model.md](migan-model.md). Packaging verifies its exact size and SHA-256 before copying
it and its code/weight licences. The legacy SD 1.5 Fill checkpoint remains a separate app-managed download.
`-Sam1Model` is optional; when provided, it must match the size and SHA-256 in
[sam1-model.md](sam1-model.md). A mismatched file stops packaging before a release is built.
`-KleinModelDir` is also optional. It verifies five model/tokenizer files and the licence against the pinned
revision's sizes and SHA-256 hashes before staging them under
`models/flux2-klein-4b/`, with the model's Apache-2.0 licence under
`models/licenses/`. The archive writer supports files larger than 2 GB and ZIP64.
Omit the argument to keep the smaller package and use app-managed Klein setup.
Release artifacts include their own manifest and runtime validation reports; the earlier
MI-GAN/SAM 1 portable checks alone do not establish Klein runtime behavior.
