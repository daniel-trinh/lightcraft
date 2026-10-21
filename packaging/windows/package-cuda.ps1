<#
.SYNOPSIS
  Build and package the standalone Windows CUDA release.

.DESCRIPTION
  Builds LightCraft for x64 with the cuda feature and packages the app plus the CUDA runtime
  DLLs required by its cudarc CUDA backend. The output path must not already exist. CUDA_ROOT
  must point at the extracted CUDA SDK/runtime root. CUDA build environment changes are scoped
  to this PowerShell process; the script never installs CUDA or changes the global PATH.
#>
param(
  [Parameter(Mandatory = $true)][string] $CudaRoot,
  [Parameter(Mandatory = $true)][string] $MiganModel,
  [string] $Sam1Model,
  [Parameter(Mandatory = $true)][string] $OutDir
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$MiganModel = (Resolve-Path -LiteralPath $MiganModel -ErrorAction Stop).Path
$miganHash = '365c3680e7e33b140761d11a0e62d7ed596572d036bbc3e670540c2baaf9c845'
if ((Get-Item -LiteralPath $MiganModel).Length -ne 29528324 -or
    (Get-FileHash -LiteralPath $MiganModel -Algorithm SHA256).Hash.ToLowerInvariant() -ne $miganHash) {
  throw 'Removal model does not match the pinned converted MI-GAN checkpoint. See docs/migan-model.md.'
}
if ($Sam1Model) {
  $Sam1Model = (Resolve-Path -LiteralPath $Sam1Model -ErrorAction Stop).Path
  if ((Get-Item -LiteralPath $Sam1Model).Length -ne 374979376 -or
      (Get-FileHash -LiteralPath $Sam1Model -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'fbf4c1ad844a2e9b867d89dc04a61f3ece19c57ebb7d8fcb6da92062a291d839') {
    throw 'Object model does not match the pinned SAM 1 ViT-B safetensors checkpoint. See docs/sam1-model.md.'
  }
}
$CudaRoot = (Resolve-Path -LiteralPath $CudaRoot -ErrorAction Stop).Path
$env:CUDA_ROOT = $CudaRoot
$env:CUDA_PATH = $CudaRoot
$cudaBin = Join-Path $CudaRoot 'bin'
if (-not (Test-Path -LiteralPath $cudaBin -PathType Container)) { throw "CUDA SDK bin directory is missing: $cudaBin" }
$env:Path = "$cudaBin;$env:Path"
$env:CUDA_COMPUTE_CAP = if ($env:CUDA_COMPUTE_CAP) { $env:CUDA_COMPUTE_CAP } else { '89' }
$env:CARGO_BUILD_JOBS = '4'
$env:RAYON_NUM_THREADS = '4'
$OutDir = [IO.Path]::GetFullPath($OutDir)
if (Test-Path -LiteralPath $OutDir) { throw "Output already exists; choose a new, empty destination: $OutDir" }
$parent = Split-Path -Parent $OutDir
if (-not (Test-Path -LiteralPath $parent)) { throw "Output parent directory does not exist: $parent" }

function Invoke-Native([string] $What, [scriptblock] $Block) {
  Write-Output "==> $What"
  & $Block
  if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

# Cargo's MSVC target needs the compiler and Windows SDK environment. Import it into this
# PowerShell process only; no machine/user environment variables are changed.
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
  $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (Test-Path $vswhere) {
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $vcvars = if ($vs) { Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat' }
    if ($vcvars -and (Test-Path $vcvars)) {
      # A batch file avoids PowerShell 7's native argument quoting changing the
      # nested quotes around a Visual Studio path containing spaces.
      $envBatch = Join-Path ([IO.Path]::GetTempPath()) ('lightcraft-vcvars-' + [Guid]::NewGuid().ToString('N') + '.cmd')
      try {
        [IO.File]::WriteAllLines($envBatch, @('@echo off', ('call "' + $vcvars + '" >nul'), 'if errorlevel 1 exit /b 1', 'set'))
        $envLines = & $env:ComSpec /d /c $envBatch
        if ($LASTEXITCODE -ne 0) { throw 'Visual Studio compiler environment setup failed.' }
        foreach ($line in $envLines) {
          if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
        }
      } finally { Remove-Item -LiteralPath $envBatch -Force -ErrorAction SilentlyContinue }
    }
  }
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
  throw 'MSVC x64 build tools were not found. Install Visual Studio Build Tools with the C++ x64 tools and Windows SDK, or run this script from an x64 Native Tools developer terminal.'
}
$clPath = (Get-Command cl.exe).Source
$env:NVCC_CCBIN = $clPath

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not $vs -and (Test-Path $vswhere)) {
  $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}

$requiredDlls = @(
  'cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll',
  'curand64_10.dll', 'nvrtc64_120_0.dll', 'nvrtc-builtins64_128.dll'
)
$runtime = @{}
foreach ($dll in $requiredDlls) {
  $found = Get-ChildItem -LiteralPath $CudaRoot -Filter $dll -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $found) { throw "CUDA runtime dependency '$dll' is missing beneath '$CudaRoot'. Supply the complete CUDA 12.8.1 runtime SDK; PATH is not searched." }
  $runtime[$dll] = $found.FullName
}
$cudaLicense = Join-Path $CudaRoot 'LICENSE'
if (-not (Test-Path -LiteralPath $cudaLicense -PathType Leaf)) {
  throw "CUDA SDK license file is missing: $cudaLicense. Include the NVIDIA CUDA Toolkit EULA with the runtime SDK."
}

$buildSha = (& git -C $Root rev-parse HEAD 2>$null).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Could not identify the source revision.' }
$sourceDirty = [bool](& git -C $Root status --porcelain --untracked-files=normal)
$env:LIGHTCRAFT_BUILD_SHA = $buildSha
$env:LIGHTCRAFT_REQUIRE_WINRES = '1'
$env:CRAFT_FONTS_REQUIRED = '1'
if (-not $env:CRAFT_FONTS_DIR -or -not (Test-Path -LiteralPath $env:CRAFT_FONTS_DIR -PathType Container)) {
  throw 'Release builds require CRAFT_FONTS_DIR pointing at a checked-out storytold/craft-fonts directory.'
}
$target = 'x86_64-pc-windows-msvc'
$flagVar = 'CARGO_TARGET_' + ($target.ToUpperInvariant() -replace '-', '_') + '_RUSTFLAGS'
[Environment]::SetEnvironmentVariable($flagVar, '-C target-feature=+crt-static', 'Process')
Push-Location $Root
try {
  Invoke-Native 'cargo build --release --features cuda' { cargo build --release --locked -p lightcraft -p lightcraft-cli --target $target --features cuda }
} finally { Pop-Location }

$targetDir = if ($env:CARGO_TARGET_DIR) {
  if ([IO.Path]::IsPathRooted($env:CARGO_TARGET_DIR)) { $env:CARGO_TARGET_DIR } else { Join-Path $Root $env:CARGO_TARGET_DIR }
} else { Join-Path $Root 'target' }
$bin = Join-Path $targetDir "$target\release"
$exe = Join-Path $bin 'lightcraft.exe'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Release executable was not produced: $exe" }
$guid = [Guid]::NewGuid().ToString('N')
$stage = Join-Path $parent ".lightcraft-cuda-stage-$guid"
$portable = Join-Path $stage 'LightCraft-CUDA-portable'
try {
  New-Item -ItemType Directory -Path $portable -Force | Out-Null
  Copy-Item -LiteralPath $exe -Destination $portable
  $modelFolder = Join-Path $portable 'models'
  New-Item -ItemType Directory -Path $modelFolder | Out-Null
  Copy-Item -LiteralPath $MiganModel -Destination (Join-Path $modelFolder 'migan-f32.safetensors')
  $modelLicenses = Join-Path $modelFolder 'licenses'
  New-Item -ItemType Directory -Path $modelLicenses | Out-Null
  foreach ($license in @('MI-GAN-LICENSE.txt', 'MI-GAN-LICENSE-WEIGHTS.txt')) {
    Copy-Item -LiteralPath (Join-Path $Root "docs/licenses/$license") -Destination $modelLicenses
  }
  Copy-Item -LiteralPath (Join-Path $Root 'docs/migan-model.md') -Destination (Join-Path $modelFolder 'PROVENANCE.md')
  if ($Sam1Model) {
    $sam1Folder = Join-Path $modelFolder 'sam1'
    New-Item -ItemType Directory -Path $sam1Folder | Out-Null
    Copy-Item -LiteralPath $Sam1Model -Destination (Join-Path $sam1Folder 'sam_vit_b_01ec64.safetensors')
    Copy-Item -LiteralPath (Join-Path $Root 'docs/licenses/SAM1-LICENSE.txt') -Destination $modelLicenses
    Copy-Item -LiteralPath (Join-Path $Root 'docs/sam1-model.md') -Destination (Join-Path $sam1Folder 'PROVENANCE.md')
  }
  Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README-CUDA.txt') -Destination (Join-Path $portable 'START-HERE.txt')
  foreach ($dll in $requiredDlls) { Copy-Item -LiteralPath $runtime[$dll] -Destination (Join-Path $portable $dll) }
  foreach ($file in @('README.md', 'LICENSE', 'LICENSE-MIT', 'LICENSE-APACHE', 'NOTICE')) {
    $path = Join-Path $Root $file
    if (Test-Path -LiteralPath $path -PathType Leaf) { Copy-Item -LiteralPath $path -Destination $portable }
  }
  Copy-Item -LiteralPath $cudaLicense -Destination (Join-Path $portable 'NVIDIA-CUDA-Toolkit-LICENSE.txt')
  $cudaLicenseDir = Join-Path $portable 'NVIDIA-LICENSES'
  $sdkLicenses = Join-Path $CudaRoot 'licenses'
  if (Test-Path -LiteralPath $sdkLicenses -PathType Container) {
    Get-ChildItem -LiteralPath $sdkLicenses -Recurse -File | Where-Object Name -Match '^(LICENSE|NOTICE|EULA)$|^(LICENSE|NOTICE|EULA|.*license.*|.*notice.*)\.(txt|rtf|html?)$' | ForEach-Object {
      $relative = [IO.Path]::GetRelativePath($sdkLicenses, $_.FullName)
      $destination = Join-Path $cudaLicenseDir $relative
      New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
      Copy-Item -LiteralPath $_.FullName -Destination $destination
    }
  }
  # The application and NVIDIA libraries use a static CRT. Verify the dependency
  # closure rather than relying on redistributables installed on the build PC.
  $dumpbin = Join-Path (Split-Path -Parent $clPath) 'dumpbin.exe'
  if (-not (Test-Path -LiteralPath $dumpbin -PathType Leaf)) { throw "Dependency inspector missing: $dumpbin" }
  $systemDlls = @(
    'kernel32.dll', 'ole32.dll', 'combase.dll', 'shell32.dll', 'user32.dll',
    'gdi32.dll', 'advapi32.dll', 'uiautomationcore.dll', 'oleaut32.dll',
    'opengl32.dll', 'dxgi.dll', 'setupapi.dll', 'bcryptprimitives.dll',
    'ws2_32.dll', 'imm32.dll', 'dwmapi.dll', 'uxtheme.dll', 'bcrypt.dll',
    'ntdll.dll', 'nvcuda.dll'
  )
  $packagedDlls = @(Get-ChildItem -LiteralPath $portable -Filter '*.dll' -File | ForEach-Object { $_.Name.ToLowerInvariant() })
  $dependencies = [ordered]@{}
  foreach ($binary in Get-ChildItem -LiteralPath $portable -File | Where-Object Extension -In @('.exe', '.dll')) {
    $inspection = & $dumpbin /nologo /dependents $binary.FullName
    if ($LASTEXITCODE -ne 0) { throw "Could not inspect dependencies of $($binary.Name)." }
    $imports = @($inspection | ForEach-Object {
      if ($_ -match '^\s+([A-Za-z0-9_.-]+\.dll)\s*$') { $Matches[1].ToLowerInvariant() }
    } | Sort-Object -Unique)
    if ($imports.Count -eq 0) { throw "No dependencies found for $($binary.Name); refusing an unverified package." }
    foreach ($dependency in $imports) {
      if ($dependency -notin $packagedDlls -and $dependency -notin $systemDlls -and $dependency -notmatch '^(api-ms-win-|ext-ms-win-)') {
        throw "Unbundled dependency '$dependency' required by '$($binary.Name)'. Package its runtime and license before releasing."
      }
    }
    $dependencies[$binary.Name] = $imports
  }

  $fontLicenses = Get-ChildItem -LiteralPath $env:CRAFT_FONTS_DIR -Recurse -File | Where-Object Name -Match '^(OFL|LICENSE|COPYING).*\.txt$'
  if (-not $fontLicenses) { throw "Could not find craft-fonts license texts beneath '$env:CRAFT_FONTS_DIR'." }
  $fontLicenseDir = Join-Path $portable 'FONT-LICENSES'
  foreach ($license in $fontLicenses) {
    $relative = [IO.Path]::GetRelativePath($env:CRAFT_FONTS_DIR, $license.FullName)
    $destination = Join-Path $fontLicenseDir $relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $license.FullName -Destination $destination
  }

  $files = Get-ChildItem -LiteralPath $portable -File -Recurse | Sort-Object FullName | ForEach-Object {
    [ordered]@{
      name = [IO.Path]::GetRelativePath($portable, $_.FullName).Replace('\', '/')
      sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
      bytes = $_.Length
    }
  }
  Push-Location $Root
  try { $version = ((& cargo metadata --no-deps --format-version 1 | ConvertFrom-Json).packages | Where-Object name -eq 'lightcraft' | Select-Object -ExpandProperty version -First 1) }
  finally { Pop-Location }
  $manifest = [ordered]@{ product = 'LightCraft CUDA portable'; version = $version; architecture = 'x86_64'; cuda_toolkit = '12.8.1'; cuda_compute_capability = $env:CUDA_COMPUTE_CAP; source_revision = $buildSha; source_worktree_dirty = $sourceDirty; dependencies = $dependencies; files = @($files) }
  $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $portable 'SHA256-MANIFEST.json') -Encoding utf8
  $zip = Join-Path $stage 'LightCraft-CUDA-portable.zip'
  Compress-Archive -LiteralPath $portable -DestinationPath $zip
  Move-Item -LiteralPath $stage -Destination $OutDir
} catch {
  $resolvedParent = [IO.Path]::GetFullPath($parent).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
  $resolvedStage = [IO.Path]::GetFullPath($stage)
  if ($resolvedStage.StartsWith($resolvedParent, [StringComparison]::OrdinalIgnoreCase) -and $resolvedStage -match '\\.lightcraft-cuda-stage-[0-9a-f]{32}$' -and (Test-Path -LiteralPath $resolvedStage)) {
    Remove-Item -LiteralPath $resolvedStage -Recurse -Force
  }
  throw
}
Write-Output "Portable folder and ZIP created at: $OutDir"
