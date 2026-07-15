# RTX 5070 Laptop Qwen3.5 FLA and RWKV Native Matrix

Date: 2026-07-14

Hardware: one `NVIDIA GeForce RTX 5070 Laptop GPU`, `sm_120`, 8151 MiB,
driver `582.05`.

Software: Windows, PyTorch `2.11.0+cu128`, Triton Windows
`3.7.1.post27`, Transformers `5.12.1`, flash-linear-attention `0.5.1`, and
bitsandbytes `0.49.2`. `causal-conv1d` is not installed in this environment.

## Scope

This artifact compares RWKV-7 1.5B with official Qwen3.5 2B on exact matched
batch, prompt, decode, and quantization shapes: prompt 128/512/2048, decode
128/512, bsz 1/2/4/8, and fp16/bnb8/bnb4. It contains 144 passing raw rows
forming all 72 comparison cells.

Every Qwen row binds all 18 linear-attention layers to FLA chunk prefill, FLA
fused-recurrent decode, and FLA fused gated normalization. Convolution uses the
Transformers Torch fallback, so the effective reference backend is honestly
reported as `qwen_fla_gated_delta_rule_torch_conv`, not fully fused Qwen.

RWKV uses opt-in native fused prefill for all three precisions. Fp16 and BNB4
decode use the native graph route. BNB8 decode uses FLA with the `decode_rk`
hybrid policy because external BNB8 graph capture is not valid on this stack.
These routes remain opt-in and card-local.

## Strict result

The gate requires RWKV to reach at least `1.05x` Qwen in both prefill and
decode, while model footprint and peak allocated VRAM must not exceed Qwen in
the same cell.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | `1.109682x` | `1.473473x` | `4.298400x` | 72/72 at >=1.05x |
| Decode RWKV/Qwen | `1.466175x` | `2.720961x` | `6.882948x` | 72/72 at >=1.05x |
| Model footprint RWKV/Qwen | `0.729146x` | `0.811662x` | `0.856635x` | 72/72 no larger |
| Peak VRAM RWKV/Qwen | `0.648549x` | `0.823298x` | `0.967497x` | 72/72 no larger |

Coverage, reference backend, speed, and memory gates all pass. There are no
red or missing cells.

## Correctness

The isolated Qwen FLA/Torch probe uses identical input and eight greedy steps.
It passes 8/8 greedy equality with prompt/final logits cosine
`0.99999076`/`0.99999213`.

The RWKV native-prefill probes compare the reference and native routes using
identical input and eight greedy steps:

| Quantization | Prompt cosine | Final cosine | Greedy |
|---|---:|---:|---:|
| fp16 | `0.99999499` | `0.99999499` | 8/8 |
| BNB8 | `0.99998081` | `0.99998426` | 8/8 |
| BNB4 | `0.99999619` | `0.99999619` | 8/8 |

All probes pass the `0.9999` cosine gate. The evidence establishes an
exact-card inference performance result; it does not establish that RWKV has
better instruction, reasoning, code, math, multilingual, or long-context
quality than Qwen3.5.

## Reproduction

Run from the repository root in Windows PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File bench/run_5070_qwen35_fla_matrix.ps1
```

The runner fails closed on missing FLA core operators, runs Qwen and RWKV
correctness probes, executes the fresh-process matrix, and applies the strict
speed plus memory comparator.

Canonical files:

- `results.jsonl`: 144 final raw rows;
- `summary.json` / `summary.md`: machine-readable and concise 72-cell reports;
- `fla-vs-torch-probe.json`: Qwen backend correctness probe;
- `rwkv-prefill-correctness-*.json`: fp16/BNB8/BNB4 native-prefill probes;
- `environment.json` and `exit-codes.json`: provenance and process status.
