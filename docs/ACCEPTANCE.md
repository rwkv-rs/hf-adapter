# Acceptance contract

Last audited: **2026-08-17**.

## How to report completion

The current HF milestone is complete. Report completion by named scope, never
as a repository-wide percentage.

| Scope | Acceptance |
|---|---|
| HF adapter release | **COMPLETE** |
| Dense HF inference PP/TP | **PASS for declared scope** |
| Native production inference | **PASS for retained exact-card lines** |
| FLA reference compatibility | **PASS** |
| Quantized speed | **Only the exact retained W8/W4 lines may be claimed** |

## Functional gates

The release must preserve:

- `AutoConfig`, `AutoTokenizer`, `AutoModel`, and `AutoModelForCausalLM`;
- standard forward and `generate(..., use_cache=True)`;
- recurrent-state allocation, select, reorder, drop, compact, offload and
  restore;
- dynamic batching and chunked prefill;
- save/reload and remote-code loading;
- PEFT LoRA, Trainer, TRL SFT/DPO/GRPO, and checkpoint resume;
- dense, W8 and W4 loading where supported;
- initial HF-compatible speculative decoding.

## Correctness gates

- Outputs and timing samples must be finite.
- Optimized routes must record requested, selected and effective telemetry and
  fail closed when an explicitly required route does not activate.
- Cross-backend probes compare complete greedy traces and prompt/final logits
  at the threshold declared by the artifact.
- Cache length, handoff, select/reorder/drop and state pointer contracts must
  remain correct.
- Quantized speed claims require footprint reduction and logits/greedy parity;
  a memory-only route is labeled as such.

### Official `train_temp` parity gate

Training promotion remains pinned to `demo-training-prepare.sh` and
`demo-training-run.sh`: preparation uses `micro_bsz=1`; the measured shell
shape uses `micro_bsz=16`, `FFN3072`, `ctx_len=512`, `lr_init=6e-4`,
`lr_final=6e-5`, `adam_eps=1e-18`, `weight_decay=0.001`, `grad_cp=1`,
`deepspeed_stage_2`, and `magic_prime=2926181`. The machine-readable contract
is [`train_temp_official_x070_12x768_b16.json`](../configs/train_temp_official_x070_12x768_b16.json).

## Performance gates

- Exact GPU, model hash, dtype, batch, prompt, decode, warmup and sample count
  are part of the result identity.
- B8 tok/s is aggregate throughput.
- Paired Qwen ratios use unrounded raw throughput.
- Parameter adjustment is
  `(RWKV/Qwen throughput) * (RWKV/Qwen active parameters)`.
- Microbenchmarks cannot promote a production route without end-to-end rows.

Current cross-card evidence is indexed in
[`RESULTS_INDEX.md`](RESULTS_INDEX.md). The retained RTX 4080 7.2B state line
records **344.39 tok/s** and greedy **12,288/12,288** in
[`4080_7p2b_fp16_state_20260809`](../bench/4080_7p2b_fp16_state_20260809/README.md).

## Backend boundary

Native is the retained RWKV performance backend. FLA remains an explicit
compatibility/reference implementation and a correctness oracle. Historical
RWKV FLA speed bundles are not current acceptance evidence.
