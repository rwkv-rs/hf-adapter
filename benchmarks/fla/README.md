# FLA benchmark diagnostics

FLA is an optional optimized-backend reference. These scripts and artifacts are non-blocking diagnostics; official RWKV checkpoints and the repository HF tests define correctness.

Install FLA in a separate benchmark environment, then run:

```bash
python benchmarks/fla/compare.py \
  --model /models/rwkv7-g1d-0.4b-hf \
  --dtype fp16 \
  --device cuda \
  --fla-source /src/flash-linear-attention \
  --output-dir /results/fla/4080
```

Missing the recorded numerical thresholds is written to the bundle but returns
zero because this comparison is not a release gate. Add `--require-thresholds`
only when developing an FLA-compatible optimized backend.

The reproducible speed harness is `speed.py`. The first pinned RTX 4080
throughput bundle is archived under `results/4080-speed-20260826/`; it reports
operator, prefill and cached-decode latency without CUDA graphs or
`torch.compile`.
