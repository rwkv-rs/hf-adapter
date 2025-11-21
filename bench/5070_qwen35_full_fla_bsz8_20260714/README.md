# RTX 5070 Laptop RWKV-7 vs Qwen3.5 Full-FLA bsz8

Date: 2026-07-14

Hardware: one `NVIDIA GeForce RTX 5070 Laptop GPU`, `sm_120`, 8151 MiB,
driver `582.05`.

Software: Windows, PyTorch `2.11.0+cu128`, CUDA `12.8`, Triton Windows
`3.7.1.post27`, Transformers `5.12.1`, flash-linear-attention `0.5.1`, and
bitsandbytes `0.49.2`.

## Scope

This artifact compares RWKV-7 1.5B with official Qwen3.5 2B at `bsz=8` on
identical prompt, decode, dtype, and quantization shapes. The 18 cells cover
prompt 128/512/2048, decode 128/512, and fp16/BNB8/BNB4. All 36 raw rows pass.

Every Qwen performance row binds all 18 Gated DeltaNet layers to FLA chunk
prefill, FLA fused-recurrent decode, FLA fused gated normalization, and FLA
Triton causal-convolution prefill/update. The effective backend is
`qwen_fla_gated_delta_rule_fla_triton_conv`; no Qwen performance row uses the
Transformers Torch convolution fallback. The separate Transformers-conv run
is only a numerical oracle and is never included in `results.jsonl`.

RWKV uses opt-in native fused prefill and native-graph decode. BNB4 additionally
uses the opt-in external-quant prefill graph. These paths remain card-local and
default-off.

## Strict result

The gate requires at least `1.05x` raw prefill and decode throughput, no-larger
physical model footprint and peak allocated VRAM, full Qwen FLA bindings, and
at least `1.0x` token throughput per active billion parameters.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | `1.082707x` | `1.375135x` | `1.688725x` | 18/18 |
| Decode RWKV/Qwen | `1.795119x` | `2.544989x` | `3.456505x` | 18/18 |
| Model footprint RWKV/Qwen | `0.729146x` | `0.811662x` | `0.856635x` | 18/18 |
| Peak VRAM RWKV/Qwen | `0.605574x` | `0.845321x` | `0.955585x` | 18/18 |
| Prefill tok/s per active-B | `1.333940x` | `1.694224x` | `2.080579x` | 18/18 |
| Decode tok/s per active-B | `2.211641x` | `3.135530x` | `4.258556x` | 18/18 |

Per precision family, minimum prefill/decode speedups are `1.107277x/2.490276x`
for fp16, `1.286450x/1.795119x` for W8, and `1.082707x/2.507579x` for W4.
There are no red or missing cells.

RWKV has 1,527,404,544 logical and active parameters; Qwen has 1,881,825,088.
Both checkpoints are dense, so active equals total and the RWKV/Qwen active
ratio is `0.811661x`. Rows also record exact active-parameter applications.
Model efficiency is `tok/s / active-B` and is an acceptance gate. Hardware
logical work rate (`tok/s * active_parameters`) is reported separately and is
not a model-efficiency gate; its minimum prefill ratio is `0.878791x` because
RWKV executes fewer active parameters per token.

Runtime working set, defined as peak allocated VRAM minus physical model
footprint, is lower in 8/18 cells (`0.546805x-1.915321x`). It is disclosed as
activation/temporary-allocation telemetry rather than a capacity gate. Total
peak VRAM, the actual fit constraint, is lower in all 18 cells.

## Correctness

The full-FLA Qwen bridge matches the Transformers-conv numerical oracle on
identical input and eight greedy steps: greedy 8/8, prompt cosine
`0.99999022`, and final cosine `0.99999237`.

RWKV reference/native-prefill probes run at `bsz=8` with identical inputs and
eight greedy steps:

| Precision | Prompt cosine | Final cosine | Greedy |
|---|---:|---:|---:|
| fp16 | `0.99999487` | `0.99999511` | 8/8 |
| BNB8 | `0.99997848` | `0.99998188` | 8/8 |
| BNB4 external graph | `0.99999601` | `0.99999559` | 8/8 |

All probes pass the `0.9999` cosine gate. This is exact-card inference speed,
memory, and backend evidence. It does not establish Qwen3.5 model-quality
superiority or inferiority.

## Reproduction

Run from the repository root in Windows PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File bench/run_5070_qwen35_full_fla_bsz8.ps1
```

The runner fails closed on any Qwen Torch fallback, runs Qwen and RWKV
correctness probes, executes the fresh-process matrix, and applies the strict
speed, parameter-efficiency, footprint, and peak-VRAM gates.

Canonical files:

- `results.jsonl`: 36 valid raw performance rows;
- `summary.json` / `summary.md`: 18-cell strict reports;
- `full-fla-vs-transformers-conv-oracle.json`: Qwen numerical oracle only;
- `rwkv-prefill-correctness-*.json`: bsz8 fp16/W8/W4 native-prefill probes;
- `environment.json` and `exit-codes.json`: provenance and process status.
