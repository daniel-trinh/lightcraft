# Standalone CUDA build for Windows

`packaging/windows/package-cuda.ps1` builds the x64 desktop app with the Rust `cuda` feature and
creates a portable folder plus ZIP. The target computer does not need Python, ComfyUI, a CUDA
toolkit, or a separately launched service. The CUDA runtime DLLs are shipped beside
`lightcraft.exe`; this is a portable folder, not a physically single-file executable. Users
launch `lightcraft.exe` from the extracted folder. Model files are downloaded by the app when
needed and are not included in the package.

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
  -OutDir 'C:\releases\lightcraft-cuda-1.0.0'
```

`OutDir` must not already exist. The script builds first, stages into a unique temporary
directory next to the destination, then moves the completed folder into place. It does not
delete or replace an existing destination. The output contains `LightCraft-CUDA-portable/` and
`LightCraft-CUDA-portable.zip`.

CUDA model weights are separate downloads. They are not bundled because their size and licensing
terms are model-specific; the app-managed downloader obtains them when requested.
