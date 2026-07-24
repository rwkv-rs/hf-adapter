# Ascend 910B3 HF compatibility evidence — 2026-07-24

This directory records a minimal, random tiny-model compatibility gate for the
canonical FLA-free RWKV-7 Hugging Face backend. It is intentionally short and
is not a throughput or model-quality benchmark.

## Exact environment

See `environment.json`. The measured device is one Ascend 910B3 with CANN
8.5.0, Python 3.11.14, torch 2.9.0+cpu, torch_npu 2.9.0, Transformers 4.57.6,
and BF16 tensors.

## Commands

```bash
cd /data/work/hf-adapter
export PYTHONPATH=$PWD
PYTHON_BIN=/data/venvs/hf/bin/python DTYPE=bf16 BACKEND=eager \
  RESULTS=$PWD/bench/ascend_910b3_20260724/results.jsonl \
  bash scripts/run_huawei_ascend_smoke.sh

PYTHON_BIN=/data/venvs/hf/bin/python DTYPE=bf16 BACKEND=native_jit \
  RESULTS=$PWD/bench/ascend_910b3_20260724/results_native_jit.jsonl \
  bash scripts/run_huawei_ascend_smoke.sh
```

Both commands exited 0. Both JSON rows report pass for AutoConfig,
AutoTokenizer, AutoModel, AutoModelForCausalLM, forward, HF generate,
safe save/reload, dynamic recurrent cache selection and chunked-prefill parity.

## Boundary

The fixture has two random layers with hidden size 16. No official checkpoint,
quality comparison, training, quantization, performance acceptance, NPU Graph,
long-running load or multi-NPU claim is made by this artifact.

## Exact-shape quant operator evidence

`operator_w8a16_big.py/.jsonl` and `operator_w4_group_probe.py/.jsonl` are
preserved copies of the same-host raw operator runs. W8A16 per-output-channel
quantization at 4096->16384 records 1.30x/1.17x/1.04x fp16 for M=1/8/64; at
16384->4096 it records 2.05x/1.98x/1.27x. Cosine is at least 0.999955 and
packed projection payload is about half fp16. Small shapes are slightly slower
and therefore excluded by the HF speed policy.

W4 group-128 has selective large-FFN wins, but random-weight projection cosine
is around 0.993. W4 remains negative/experimental evidence; no default HF W4
path is promoted without calibrated whole-model quality and paired speed gates.

`w8_layer_integration.jsonl` is generated through the repository
`AscendW8A16Linear` wrapper. Raw operator speed and HF wrapper speed are reported
separately so Python dispatch overhead cannot be hidden.

The raw JSONL rows use status `measured`, carry separate quality/speed/memory
operator gates where applicable, and always set `production_gate_pass=false`.
They are not PASS evidence for an HF model. Production `should_quantize` in the
W4 module returns false for every card/shape; the raw-candidate matcher requires
the exact Ascend910B3, CANN 8.5.0, torch 2.9.0+cpu, torch_npu 2.9.0 and FP16
stack. Every unmeasured Huawei device/version/dtype fails closed. BF16 W4
conversion is rejected before any model layer is replaced.
`RWKV7_ALLOW_UNVALIDATED_ASCEND=1` is an experiment-only escape hatch; runtime
and quant decisions explicitly report that the stack is unvalidated.


## Real 7.2B HF evidence

The source `fla-hub/rwkv7-7.2B-g0a` config declared 32 heads, but the checkpoint
has `model.layers.0.attn.r_k=[64,64]` and 4096-wide projections. The first native
load therefore failed closed on a 4096-versus-2048 shape mismatch. The derived
view recorded an inferred `num_heads=64` repair; source shards were unchanged.
Use `scripts/create_native_hf_view.py SOURCE OUTPUT` to reproduce the view.

`results_7p2_real_bf16.json` passes standard native HF BF16 load/forward,
chunked-prefill cosine 1.0, cache continuation and generation with 14.71 GB
resident and 14.80 GB peak NPU allocation.

All whole-model quant candidates are rejected. `results_7p2_real_w8bf16.json`
records W8 value-only payload/HBM 0.851x/0.852x, decode 0.996x, cosine 0.988 and
greedy divergence. `results_7p2_real_w4fp16.json` records W4 key+value
payload/HBM 0.562x/0.575x, decode 0.978x and cosine 0.986. The stricter five-group
interleaved `results_7p2_real_w4fp16_value_only.json` records payload/HBM
0.781x/0.787x, median decode 0.988x, cosine 0.987, max/mean KL 2.905/0.400 and
min/mean top-20 overlap 0.40/0.85. No quantized whole-model default is promoted.

## Independent real-checkpoint oracle

`reference_7p2_real_b1_bf16.json` pins FLA commit
`d1ce07369d581813553f30a750af3b6b5f9af6a9`, relevant FLA file hashes, all
three official checkpoint shard hashes and tokenizer hashes. Its CPU BF16
oracle directly reads safetensors and runs a naive PyTorch recurrence; it never
invokes the adapter candidate forward. For `Hello` / input `[33155]` /
greedy-3, it emits `[45,308,459]`, matching vLLM and HF NPU.

`compare_7p2_real_b1_bf16_npu_vs_oracle.json` is a complete 208-tensor gate
over prefill and three decode logits plus prefill/final recurrent states. It
passes with logit cosine/NRMSE 0.999966/0.007904 and state cosine/NRMSE
0.999844/0.017715. Canonical capture hashes are:

- reference: `35f24f1e0116fcbee548c69ee38652b897f14e6a09dedfad79753a0380e324f9`
- HF NPU candidate: `8b83cb5af96a250565c713a5dd9f9a3ad0dff7d123659f00fc59fb86d6fe4cbf`

The ignored 66 MiB artifacts remain on the validation server at
`/data/work/hf-adapter/bench/ascend_910b3_20260724/reference_7p2_real_b1_bf16.safetensors`
and
`/data/work/hf-adapter/bench/ascend_910b3_20260724/candidate_7p2_real_b1_bf16_npu.safetensors`.
Their file SHA256 values are respectively
`56abd02361c496211648b66962929c1a2feca69155c725f7bf410ddcd90c86db` and
`ed29249405e00d47376d567edf0ac3a6478dcc6126e4575cd3733155ae7c31b4`.
They can be rebuilt with `build_ascend_hf_reference.py` and
`capture_ascend_hf_candidate.py`; the committed JSON keeps individual tensor
hashes so a changed or incomplete rebuild fails closed.

`results_7p2_real_b2_ragged.json` additionally compares two different valid
lengths in a padded B2 batch with their compact B1 forwards and cache
continuations, then checks `chunk_size=1` split parity. All gates pass; global
logit/state minimum cosine is 0.99999988/0.99999970, both maximum normalized
RMSE values are zero, and the evidence hash is
`54a7ac69520349e99136c144e1f9a4fc850d813b2b8cd4668b8b99b0215f160e`.
