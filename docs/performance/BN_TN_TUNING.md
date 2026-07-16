# BN/TN kernel tuning

`BN` and `TN` are separate CUDA tiling terms in this repository:

- **BN / block-N:** output columns owned by one CUDA thread block;
- **TN / thread-N:** output columns accumulated by one CUDA thread;
- the explicit probe launches `BN / TN` threads, and rejects configurations
  that do not form complete 32-thread warps.

Do not confuse either term with bitsandbytes. Do not infer `TN` from a Triton
`num_warps` value: Triton's physical lane/MMA layout is compiler controlled.

## Measurement lane

[`../../bench/bench_quant_bn_tn.py`](../../bench/bench_quant_bn_tn.py) compiles
a small handwritten-CUDA W8/W4 decode probe and compares each legal `BN × TN`
pair with:

1. the current native MM8/MM4 dispatch;
2. same-shape dense fp16;
3. the current quantized output for cosine and maximum-absolute-error gates.

The minimum required matrix is:

- true batch `1` and `8`;
- square, FFN-up and FFN-down projections;
- `BN={64,128,256}` and `TN={1,2,4,8}`, filtered by legal warp count;
- exact GPU name, compute capability, Torch/CUDA versions and raw JSONL.

Example:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
python bench/bench_quant_bn_tn.py \
  --batch-sizes 1 8 \
  --shapes 2048x2048 2048x8192 8192x2048 \
  --output bench/<artifact>/bn_tn.jsonl
```

## Promotion rule

A microkernel winner is not automatically a production winner. Promote a
card/shape dispatch only when it passes quantized-output correctness, beats the
current native kernel, and improves or preserves paired end-to-end prefill or
decode throughput. Keep unmeasured cards on their existing route.

## RTX 5090 result

The 2026-07-16 `sm_120` sweep covers 288/288 correct rows over 32 W8/W4,
B1/B8 and 1.5B/7.2B projection cases. Some internal W4 B8 shapes beat the
current quant kernel, but every handwritten BN/TN winner remains slower than
same-shape fp16 and the FFN-down/lm-head routes regress. No production dispatch
is promoted from this experiment. Evidence:
[`../../bench/5090_bn_tn_20260716/`](../../bench/5090_bn_tn_20260716/README.md).
