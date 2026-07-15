# RTX 5090 g1h 13.3B acceptance

Status: **PASS**.

This artifact validates the latest official g1h 13.3B checkpoint at the
32 GB RTX 5090 fit boundary.

## Source and conversion

- Source: `rwkv7-g1h-13.3b-20260710-ctx10240.pth`
- Size: `26,540,868,485` bytes
- SHA256: `5bd705d13497d23530e544d5afb45bdf542b5f67dffee31e3e2b35e4042cfcfb`
- Conversion: fp16, `fused_recurrent`, no fused norm, low-memory path,
  six safetensors shards with a 5 GB shard limit
- Converted structure: 2,016 indexed tensor keys, 61 layers, hidden size
  4,096, head dimension 64, and an `RWKV7Tokenizer`

The safetensors index, every shard, `AutoConfig`, and `AutoTokenizer` were
opened locally on the benchmark host before acceptance. See
[`conversion_validation.json`](conversion_validation.json) and
[`checkpoint-sha256.txt`](checkpoint-sha256.txt).

## Environment

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, `sm_120`
- PyTorch: `2.11.0+cu128`
- CUDA runtime: `12.8`
- Triton: `3.6.0`
- Transformers: `5.12.1`
- FLA: `0.5.1`
- Dtype/attention: fp16 / `fused_recurrent`

## Results

HF load, forward, and four-token generation pass. The smoke row reports a
25,309.1 MiB model footprint and 25,448.3 MiB peak VRAM.

The speed-policy boundary uses B8, prompt 128, decode 128, paired fp16
baselines, 16 warmups, and three timing repeats:

| Quant | Decode tok/s | Speed ratio | Footprint MiB | Footprint ratio | Peak MiB | Prompt cosine | Final cosine | Same next |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fp16 | `225.1` | `1.0000x` | `25,309.1` | `1.0000x` | `26,583.7` | `1.000000` | `1.000000` | yes |
| MM8 | `224.8` | `1.0013x` | `25,053.3` | `0.9899x` | `26,327.9` | `0.999976` | `0.999971` | yes |
| MM4 | `222.9` | `0.9845x` | `24,925.3` | `0.9848x` | `26,199.9` | `0.999851` | `0.999854` | yes |

All smoke, matrix, and `>=0.98x` gate exit codes are zero. MM8/MM4 each
replace one selected speed-policy module (`lm_head`); these rows establish the
large-model fit and near-fp16 speed boundary, not full-memory quantization.

Raw rows are in [`13p3_smoke.jsonl`](13p3_smoke.jsonl) and
[`quant_13p3_boundary.jsonl`](quant_13p3_boundary.jsonl). The fail-closed gate
report is [`quant_13p3_summary.md`](quant_13p3_summary.md).
