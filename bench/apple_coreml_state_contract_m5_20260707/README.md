# Apple CoreML stateful contract evidence

This evidence directory records the next CoreML/ANE step for the
RWKV-7-over-Qwen3.5 Apple/mobile goal.

It does **not** claim stateful CoreML runtime performance yet.  Instead it makes
the missing CoreML stateful decode/prefill lane concrete and auditable: the
CoreML export manifest now contains the planned RWKV recurrent state tensor
contract, and the runtime plan row reports whether stateful `decode` and
`prefill` functions are actually implemented.

## Device and target

- Device: Mac17,3 / Apple M5 / 16GB unified memory
- OS: macOS 26.5 arm64
- Model: `/Users/wangyue/Documents/vllmsp/models/rwkv7-g1d-0.4b-hf`
- CoreML target: `iOS18`, `cpu-and-ne`
- Quantization plan: `lut4`
- State mode: `wkv-coreml`
- Planned shape: `4096 chars / 128 tokens`

## Commands

```bash
PY=/Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python
OUTDIR=bench/apple_coreml_state_contract_m5_20260707
EXPORT_DIR="$OUTDIR/coreml-rwkv7-g1d-0.4b"
MODEL=/Users/wangyue/Documents/vllmsp/models/rwkv7-g1d-0.4b-hf

PYTHONPATH=. "$PY" scripts/export_rwkv7_coreml.py \
  "$MODEL" \
  "$EXPORT_DIR" \
  --dry-run \
  --chunks 4 \
  --prefill-seq-length 4096 \
  --sample-seq-length 128 \
  --state-mode wkv-coreml \
  --quantization lut4 \
  --deployment-target iOS18 \
  --compute-units cpu-and-ne \
  --results "$OUTDIR/results_coreml_export_plan.jsonl"

PYTHONPATH=. "$PY" bench/run_coreml_apple_baseline.py \
  --manifest "$EXPORT_DIR/coreml_export_manifest.json" \
  --dry-run \
  --prompt-target-chars 4096 \
  --decode-lengths 128 \
  --repeat 1 \
  --compute-units cpu-and-ne \
  --results "$OUTDIR/results_coreml_runtime_plan.jsonl"

PYTHONPATH=. "$PY" bench/audit_qwen35_apple_goal.py \
  --results bench/apple_qwen35_08b_longctx_m5_20260707/results_qwen35_08b_4096_128_token_only.jsonl \
  --results bench/apple_qwen35_08b_longctx_m5_20260707/results_rwkv04_mm4_4096_128_eval8_fused.jsonl \
  --results bench/apple_qwen35_08b_longctx_m5_20260707/results_compare_4096_128.jsonl \
  --results "$OUTDIR/results_coreml_export_plan.jsonl" \
  --results "$OUTDIR/results_coreml_runtime_plan.jsonl" \
  --tier 'qwen3.5:0.8b-mlx|mlx-community/Qwen3.5-0.8B-MLX-4bit|qwen35-0.8b-mlx-4bit=rwkv7-g1d-0.4b-hf' \
  --required-shape chars4096:128 \
  --require-coreml \
  --append "$OUTDIR/results_coreml_goal_audit_with_longctx.jsonl"
```

## Manifest contract

`coreml-rwkv7-g1d-0.4b/coreml_export_manifest.json` now includes
`state_contract.version=rwkv7_coreml_state_contract_v1`.

For this 0.4B model the planned state tensors are:

| Tensor | Shape | Dtype |
|---|---:|---|
| per-layer `wkv_state` | `[1, 16, 64, 64]` | `float32` |
| per-layer `attn_x_prev` | `[1, 1024]` | `float16` |
| per-layer `ffn_x_prev` | `[1, 1024]` | `float16` |
| global `v_first` | `[1, 1024]` | `float16` |
| global `seen_tokens` | `[1]` | `int32` |

The runtime plan row records:

- `stateful_contract_present=true`
- `state_contract_version=rwkv7_coreml_state_contract_v1`
- `decode_implemented=false`
- `prefill_implemented=false`
- `pass_status_requires_stateful_decode=true`

## Audit status

`results_coreml_goal_audit_with_longctx.jsonl` combines the existing 0.8B
long-context Qwen/RWKV MLX evidence with this CoreML contract evidence.  It now
reports:

- long-context coverage: pass
- quant/state-cache fields on the MLX row: pass
- comparison gate: fail, because RWKV is still slower than Qwen3.5
- CoreML stateful runtime: `prototype`
- CoreML reason: `CoreML stateful decode/prefill contract exists, but runtime pass row is missing`

## Engineering implication

The next CoreML/ANE implementation step is no longer ambiguous.  To turn this
prototype into a pass row, a future PR must produce a real `.mlpackage` with
implemented stateful `decode` and `prefill` functions that round-trip the above
state tensors and emit real `qwen35_apple_baseline` rows with TTFT, prefill
throughput, decode throughput, memory, quantization mode, and correctness fields.
