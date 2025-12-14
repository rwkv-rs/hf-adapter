[CmdletBinding()]
param(
    [ValidateSet("all", "infer", "train")]
    [string]$Mode = "all",
    [int]$Steps = 12,
    [int]$Threads = 4,
    [string]$OutputDir = "",
    [string]$Python = "python",
    [switch]$Install
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $Root
try {
    $PythonExe = $Python
    if ($Install) {
        $Venv = Join-Path $Root ".venv-cpu-demo"
        if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
            & $Python -m venv $Venv
            if ($LASTEXITCODE -ne 0) { throw "Failed to create $Venv" }
        }
        $PythonExe = Join-Path $Venv "Scripts\python.exe"
        & $PythonExe -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip" }
        & $PythonExe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
        if ($LASTEXITCODE -ne 0) { throw "Failed to install CPU PyTorch" }
        & $PythonExe -m pip install -e .
        if ($LASTEXITCODE -ne 0) { throw "Failed to install rwkv7-hf-adapter" }
    }

    & $PythonExe -c "import torch, transformers, rwkv7_hf"
    if ($LASTEXITCODE -ne 0) {
        throw "Python dependencies are missing. Run this script again with -Install."
    }

    $DemoArgs = @(
        "examples/cpu_tiny_demo.py",
        "--mode", $Mode,
        "--steps", $Steps,
        "--threads", $Threads
    )
    if ($OutputDir) {
        $DemoArgs += @("--output-dir", $OutputDir)
    }
    & $PythonExe @DemoArgs
    if ($LASTEXITCODE -ne 0) { throw "RWKV-7 CPU demo failed" }
}
finally {
    Pop-Location
}
