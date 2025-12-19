# RTX 4080 Native HF production validation

Date: 2026-07-19

Hardware: one `NVIDIA GeForce RTX 4080` 16GB, `sm_89`, driver `595.71.05`.

Software: Linux, Python `3.11.15`, PyTorch `2.6.0+cu124`, CUDA runtime
`12.4`, Transformers `5.12.1`, Triton `3.2.0`, bitsandbytes `0.49.2`,
TorchAO `0.16.0`, flash-linear-attention `0.5.1`, fla-core `0.5.1`, and
causal-conv1d `1.5.0.post8`.

The implementation and runners start from upstream `main` commit
`8945395a165c497c2e3eb5f1b6e9284176b48872`.

## Scope

This artifact validates the repository-native HF backend on the exact RTX
4080. It covers:

- 0.4B Native HF API, recurrent cache, dynamic batch helpers, chunked
  prefill, save/reload, PEFT backward, and speculative generation;
- 0.4B BF16 Trainer, checkpoint resume, SFT, DPO, GRPO, and PEFT
  save/load/merge smoke;
- 0.4B and 1.5B fp16 native-prefill graph correctness and performance at
  B1/B2/B4/B8 and prompt 128/512/2048;
- 1.5B B1/B2/B4/B8 cached-decode execution;
- 1.5B versus official Qwen3.5-2B at B8, prompt 128/512/2048, decode
  128/512, full-prompt prefill, and fp16;
- full-model BNB8/BNB4 memory routes plus output-head A8W8/TorchAO-W4
  paired speed routes.

This is exact-card inference, memory, backend, and training-compatibility
evidence. It does not compare task quality, prove multi-GPU training, or claim
that full-model W8/W4 is universally no slower than fp16.

## Native prefill and decode policy

The exact RTX 4080 policy enables native prefill graph/scan, shift mix, state
prep, and output prep only for the measured 24 shapes:

- hidden/layers: `1024/24` and `2048/24`;
- batch: `1`, `2`, `4`, `8`;
- prompt: `128`, `512`, `2048`.

Outside this allowlist the native model keeps its compatible fallback. The
graph cache holds four shapes. Decode uses four-warp fused norm/mix; its B8
probe is `1.0840x` the unfused graph path with `1024/1024` greedy tokens equal.
The Ada linear and sparse-FFN experiments remain disabled because their A/B
rows are neutral or negative.

| Model | Shapes | Native/reference prefill | Greedy/cache handoff |
|---|---:|---:|---:|
| 0.4B | 12/12 | `1.0560x-2.7803x` | 12/12 |
| 1.5B | 12/12 | `1.0391x-1.5416x` | 12/12 |

The exact-policy row uses `native_prefill_graph`; a B8/prompt256 row outside
the allowlist records the fallback without silently inheriting a 4090 tile.

## Full-FLA Qwen3.5 comparison

The final matrix compares RWKV-7 1.5B with official Qwen3.5-2B at B8. All six
Qwen rows require live FLA bindings and reject Torch fallback. The separate
operator probe verifies all 18 Gated DeltaNet layers on chunk-FLA prefill and
fused-recurrent decode, all 18 causal-conv1d prefill/update bindings, and fused
gated normalization.

| Metric | Minimum | Maximum | Passing cells |
|---|---:|---:|---:|
| Dense prefill RWKV/Qwen | `1.024180x` | `1.122658x` | 6/6 |
| Dense decode RWKV/Qwen | `1.435296x` | `1.472913x` | 6/6 |
| Decode active-work ratio | `1.768344x` | `1.814687x` | 6/6 |

RWKV/Qwen physical model footprints are `2913.3/3589.3 MB`. These are raw
throughput and active-parameter-work comparisons on the same card and shape;
they are not a Qwen3.5 model-quality result.

## Quantized routes

BNB8 and BNB4 are full-model memory routes. Their six shapes all execute with
finite logits and reduce model footprint to `0.604572x` and `0.406858x` dense.
They are not used for a speed claim.

The paired speed routes replace one output-head module and measure fp16 then
quantized execution in the same process. In line with the promoted RTX
3090/4090 contract, acceptance requires cached decode and complete-cell
`prefill + decode` latency to be no slower than fp16. Prefill remains explicit
telemetry and is not independently gated.

| Route | Prefill minimum | Decode minimum | Total minimum | Footprint | Min cosine | Greedy |
|---|---:|---:|---:|---:|---:|---:|
| A8W8 head | `0.9988x` | `1.0076x` | `1.005056x` | `0.9561x` | `0.999951` | 6/6 |
| TorchAO-W4 head | `0.9996x` | `1.0458x` | `1.026122x` | `0.9355x` | `0.999520` | 6/6 |

The P512/D128 prefill rows were repeated with five warmups and eleven timing
samples. A8W8 confirms at `1.0005x`; W4 remains `0.9995x` prefill while its
decode remains `1.0465x`. The original rows and confirmations are retained so
the total-latency interpretation is auditable. Direct 8-row A8W8 GEMV and
group64 W4 probes were not promoted.

## Training compatibility

The 0.4B BF16 Native HF smoke records six finite Trainer updates with 144/144
nonzero LoRA gradients, checkpoint resume to global step 3, and two-step SFT,
DPO, and GRPO runs with changed trainable weights. PEFT adapter
save/load/merge and the separate two-step roundtrip pass. These bounded smokes
prove interface and update behavior, not long-run convergence or capacity.

## Repository tests

The final Linux run on the RTX 4080 host completes with `548 passed`,
`9 skipped`, and exit code 0. The run includes the CUDA/Triton sequence-mix
kernel test as well as the platform-independent adapter suite. The complete
output is retained in `full_tests_linux.log`.

## Reproduction

Install the CUDA dependencies and place local converted model directories on a
machine whose reported device contains `RTX 4080`:

```bash
python -m pip install -e ".[train,quant]"

CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN=python \
bash bench/run_4080_qwen35_pair_acceptance.sh \
  rwkv-1.5b__qwen3.5-2b \
  /path/to/rwkv7-g1g-1.5b-hf \
  /path/to/Qwen3.5-2B \
  /tmp/rtx4080-acceptance
```

Observable success is exit code 0, `matrix_failures.txt=0`,
`pipeline_exit_code.txt=0`, `summary.json.status=pass`, exact coverage
`6/6/12/12`, and no entry in `summary.json.errors`. The runner rejects another
GPU or model-pair label before starting.

If execution stops, retain the completed JSONL and logs for diagnosis, then
rerun the failed exact cell with `bench/bench_native_quant_e2e_decode.py`; the
top-level runner intentionally starts a fresh final matrix. Check the Qwen
operator probe first if a dependency update changes FLA or causal-conv1d
bindings.

## Files

- `qwen35_dense_final.jsonl`, `qwen35_memory_final.jsonl`, and
  `qwen35_paired_quant_final.jsonl`: final raw rows;
- `qwen35_summary.json` / `qwen35_summary.md`: fail-closed accepted report;
- `qwen_fla_operator_probe.pt`: saved input, logits and greedy-token tensors;
- `qwen_fla_smoke.jsonl`: live optimized Qwen operator bindings;
- `prefill_0p4b.jsonl`, `prefill_1p5b.jsonl`, and policy rows: native-prefill
  evidence;
- `decode_norm_mix_0p4b.jsonl` and `decode_fusions_0p4b.jsonl`: positive and
  rejected decode fusions;
- `training_bf16_0p4b.log` and `training_results_0p4b.jsonl`: training smokes;
- `full_tests_linux.log` / `full_tests_linux_exit_code.txt`: final Linux suite;
- `environment.json` and `SHA256SUMS`: provenance and integrity.
