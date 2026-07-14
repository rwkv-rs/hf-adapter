[CmdletBinding()]
param(
    [string]$Root = "",
    [string]$OutDir = "D:\bench\qwen35-5070-fla-20260714\matrix-2b-native-prefill",
    [string]$RwkvModel = "D:\models\rwkv7\rwkv7-g1g-1.5b-hf",
    [string]$QwenModel = "D:\models\qwen\Qwen3.5-2B",
    [string]$Python = "",
    [int]$Warmup = 1,
    [int]$Runs = 1,
    [switch]$SmokeOnly
)

$ErrorActionPreference = "Stop"
if (-not $Root) {
    $Root = Split-Path -Parent $PSScriptRoot
}
if (-not $Python) {
    $Python = (Get-Command python).Source
}
$matrixName = "qwen35_5070_laptop_fla_hf"
$modelPair = "rwkv-1.5b__qwen3.5-2b"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$env:CUDA_VISIBLE_DEVICES = "0"
$env:RWKV7_NATIVE_MODEL = "0"
$env:RWKV7_FAST_TOKEN_QUANT = "1"
$env:RWKV7_FAST_PREFILL = "1"
$env:RWKV7_FAST_PREFILL_QUANT = "1"
$env:RWKV7_NATIVE_PREFILL_GRAPH = "1"
$env:RWKV7_NATIVE_PREFILL_FUSED_SCAN = "1"
$env:RWKV7_NATIVE_PREFILL_FUSED_SHIFT_MIX = "1"
$env:RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP = "1"
$env:RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN = "0"
$env:RWKV7_NATIVE_PREFILL_FUSED_OUTPUT = "1"
$env:RWKV7_NATIVE_PREFILL_FUSED_SCAN_OUTPUT = "0"
$env:RWKV7_NATIVE_PREFILL_FUSED_CLAMPW_SCAN = "0"
$env:RWKV7_NATIVE_PREFILL_DPLR_SCAN = "0"
# The sm_120 policy chooses M8/M16/M32/M64 for bsz 1/2/4/8. Do not let a
# developer shell's diagnostic override leak into the acceptance matrix.
$env:RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M = $null
$env:RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS = $null
$env:PYTORCH_CUDA_ALLOC_CONF = $null
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$Root;$env:PYTHONPATH" } else { $Root }
Set-Location $Root

function Invoke-CheckedPython {
    param([string[]]$Arguments, [string]$ExitCodePath)

    & $Python @Arguments
    $code = $LASTEXITCODE
    Set-Content -LiteralPath $ExitCodePath -Value $code -Encoding ascii
    if ($code -ne 0) {
        throw "Python command failed with exit code $code"
    }
}

$commonSmoke = @(
    "bench/bench_cross_model_speed.py",
    "--model", $QwenModel,
    "--model-kind", "qwen35",
    "--model-role", "reference",
    "--model-pair", $modelPair,
    "--model-size-label", "2b",
    "--benchmark-matrix", $matrixName,
    "--dtype", "fp16",
    "--quantization", "none",
    "--device", "cuda",
    "--batch-size", "1",
    "--prompt-tokens", "128",
    "--decode-tokens", "8",
    "--warmup", "1",
    "--runs", "1"
)

$flaSmoke = Join-Path $OutDir "fla-smoke.jsonl"
Invoke-CheckedPython `
    -Arguments ($commonSmoke + @(
        "--qwen-backend", "fla",
        "--probe-output", (Join-Path $OutDir "fla-probe.pt"),
        "--probe-tokens", "8",
        "--results", $flaSmoke
    )) `
    -ExitCodePath (Join-Path $OutDir "fla-smoke-exit-code.txt")

$flaRow = Get-Content -LiteralPath $flaSmoke | Select-Object -Last 1 | ConvertFrom-Json
if ($flaRow.status -ne "pass" -or -not $flaRow.qwen_fla_core_contract_pass) {
    throw "Qwen3.5 did not bind the required FLA core operators"
}
if ($flaRow.device -ne "NVIDIA GeForce RTX 5070 Laptop GPU" -or $flaRow.gpu_arch -ne "sm_120") {
    throw "Unexpected exact-card route: $($flaRow.device) / $($flaRow.gpu_arch)"
}

Invoke-CheckedPython `
    -Arguments ($commonSmoke + @(
        "--qwen-backend", "torch",
        "--probe-output", (Join-Path $OutDir "torch-probe.pt"),
        "--probe-tokens", "8",
        "--results", (Join-Path $OutDir "torch-smoke.jsonl")
    )) `
    -ExitCodePath (Join-Path $OutDir "torch-smoke-exit-code.txt")

Invoke-CheckedPython `
    -Arguments @(
        "bench/compare_qwen35_backend_probe.py",
        "--fla-probe", (Join-Path $OutDir "fla-probe.pt"),
        "--torch-probe", (Join-Path $OutDir "torch-probe.pt"),
        "--min-cosine", "0.999",
        "--output", (Join-Path $OutDir "fla-vs-torch-probe.json"),
        "--fail-on-gate"
    ) `
    -ExitCodePath (Join-Path $OutDir "probe-compare-exit-code.txt")

$rwkvProbeCommon = @(
    "bench/bench_cross_model_speed.py",
    "--model", $RwkvModel,
    "--model-kind", "rwkv",
    "--model-role", "candidate",
    "--model-pair", $modelPair,
    "--model-size-label", "1.5b",
    "--benchmark-matrix", $matrixName,
    "--dtype", "fp16",
    "--device", "cuda",
    "--batch-size", "1",
    "--prompt-tokens", "128",
    "--decode-tokens", "8",
    "--warmup", "1",
    "--runs", "1",
    "--rwkv-code-source", "repo",
    "--probe-tokens", "8"
)
foreach ($quantization in @("none", "bnb8", "bnb4")) {
    $referenceProbe = Join-Path $OutDir "rwkv-prefill-reference-$quantization.pt"
    $nativeProbe = Join-Path $OutDir "rwkv-prefill-native-$quantization.pt"
    $env:RWKV7_BNB_SKIP_POLICY = if ($quantization -eq "bnb8") { "decode_rk" } else { "memory" }
    $env:RWKV7_FAST_PREFILL = "0"
    $env:RWKV7_FAST_TOKEN_QUANT = "0"
    Invoke-CheckedPython `
        -Arguments ($rwkvProbeCommon + @(
            "--quantization", $quantization,
            "--probe-output", $referenceProbe,
            "--results", (Join-Path $OutDir "rwkv-prefill-reference.jsonl")
        )) `
        -ExitCodePath (Join-Path $OutDir "rwkv-prefill-reference-$quantization-exit-code.txt")
    $env:RWKV7_FAST_PREFILL = "1"
    $env:RWKV7_FAST_TOKEN_QUANT = "1"
    Invoke-CheckedPython `
        -Arguments ($rwkvProbeCommon + @(
            "--quantization", $quantization,
            "--probe-output", $nativeProbe,
            "--results", (Join-Path $OutDir "rwkv-prefill-native.jsonl")
        )) `
        -ExitCodePath (Join-Path $OutDir "rwkv-prefill-native-$quantization-exit-code.txt")
    Invoke-CheckedPython `
        -Arguments @(
            "bench/compare_rwkv_prefill_probe.py",
            "--reference-probe", $referenceProbe,
            "--native-probe", $nativeProbe,
            "--min-cosine", "0.9999",
            "--output", (Join-Path $OutDir "rwkv-prefill-correctness-$quantization.json"),
            "--fail-on-gate"
        ) `
        -ExitCodePath (Join-Path $OutDir "rwkv-prefill-compare-$quantization-exit-code.txt")
}
$env:RWKV7_FAST_PREFILL = "1"
$env:RWKV7_FAST_TOKEN_QUANT = "1"
$env:RWKV7_BNB_SKIP_POLICY = "memory"

if ($SmokeOnly) {
    exit 0
}

$results = Join-Path $OutDir "results.jsonl"
$matrixArgs = @(
    "bench/run_qwen35_speed_matrix.py",
    "--pair", "$modelPair=$RwkvModel::$QwenModel",
    "--prompt-tokens", "128", "512", "2048",
    "--decode-tokens", "128", "512",
    "--batch-sizes", "1", "2", "4", "8",
    "--quantizations", "none", "bnb8", "bnb4",
    "--dtype", "fp16",
    "--benchmark-matrix", $matrixName,
    "--qwen-backend", "fla",
    "--rwkv-fast-token-backend", "native_graph",
    "--rwkv-bnb8-skip-policy", "decode_rk",
    "--warmup", $Warmup.ToString(),
    "--runs", $Runs.ToString(),
    "--results", $results,
    "--skip-existing"
)
& $Python @matrixArgs
$matrixCode = $LASTEXITCODE
Set-Content -LiteralPath (Join-Path $OutDir "matrix-exit-code.txt") -Value $matrixCode -Encoding ascii
if ($matrixCode -ne 0) {
    Write-Warning "Matrix recorded one or more failed rows; continuing to the coverage summary"
}

$compareArgs = @(
    "bench/compare_qwen35_speed_matrix.py",
    "--results", $results,
    "--expected-cells", "72",
    "--min-prefill-speedup", "1.05",
    "--min-decode-speedup", "1.05",
    "--require-memory-not-larger",
    "--required-reference-backend", "fla",
    "--json-output", (Join-Path $OutDir "summary.json"),
    "--markdown-output", (Join-Path $OutDir "summary.md")
)
Invoke-CheckedPython `
    -Arguments $compareArgs `
    -ExitCodePath (Join-Path $OutDir "compare-exit-code.txt")

exit $matrixCode
