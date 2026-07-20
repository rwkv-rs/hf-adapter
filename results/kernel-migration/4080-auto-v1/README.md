# RTX 4080 promoted optional-kernel evidence

This compact bundle records the first production `auto` policy for the clean
RWKV7 Hugging Face adapter. The model/config/cache remain package-free and the
separate `rwkv7-kernels` wheel selects the actual recurrent implementation:

- `T=1`, FP16 inference: `native-triton-rank1-scan-v1`
- `T>1`, FP16 inference: `torch-cuda-graph-reference-v1`
- unsupported dtype/device, autograd, or missing wheel: clean PyTorch reference

## Correctness

`validation/{validation-fp16,validation-bf16,validation-fp32}.json` all report
`passed: true`. FP16 covers 12 operator cases, three model sizes, B=1/4/8,
T=17/128, cached teacher decode, 64-token greedy, left/right padding, cache
state equality, regrouping, masked-state behavior, and training fallback.
Multi-token logits are exact through the Graph route; decode uses Triton and
keeps identical greedy tokens within the fixed FP16 threshold.

`hf-smoke/hf-smoke.json` reports `passed: true` for AutoConfig, AutoTokenizer,
AutoModel, AutoModelForCausalLM, greedy, beam, save/reload, and finite training
gradients. The recorded training route is the differentiable reference, not an
inference kernel mislabeled as training acceleration.

## Speed versus the readable reference

The two files under `speed/` use warmup=2 and repeats=5. Across 0.4B and 1.5B:

- exact Graph prefill: 2.28x–2.78x faster;
- Triton cached decode: 1.05x–1.32x faster.

`operator-speed/auto-final.json` is the final eager/operator matrix for
B=1/4/8 and T=1/17/128/512/2048. It records the route on every case. The
T=1 Triton route is 1.26x–1.33x faster than pinned FLA fused recurrent; the
exact multi-token Graph route is intentionally slower than FLA fused/chunk.

The end-to-end `production-speed/` table uses 0.4B and 1.5B, prefill
B=1/4/8 × T=128/512/2048, and 256-step cached decode B=1/4/8. It measures
reference, production `auto`, and pinned FLA in the same process and records
warmup separately:

- production `auto` remains 1.05x–1.70x faster than the readable reference on
  cached decode, but FLA is 1.36x–1.49x faster than `auto` end to end;
- production `auto` remains 2.26x–2.82x faster than the readable reference on
  prefill, but FLA is roughly 3.9x–13.5x faster than the exact Graph route.

These are the current measurements, not a claim that the correctness-promoted
route already beats FLA. Faster long-sequence prefill is deferred to the
versioned `prefill_v1` protocol after its numerical gates are met.

## lm_eval smoke

`lm-eval-smoke/{reference,optimized,fla}/manifest.jsonl` records a package-free
HF reference, production `auto`, and pinned-FLA PIQA smoke. All three exited
zero and selected the same four answers. The optimized manifest records 96
actual `torch-cuda-graph-reference-v1` calls. Raw samples and logs stay on the
validation host; the formal 144-unit matrix is not marked complete by this
smoke.

## Explicit Triton caveat

The multi-token Triton scan remains an experimental benchmark lane. Numerical
experiments made the FP32 recurrent state exact and the FP16 readout over
99.98% elementwise equal, but the remaining reduction-order differences either
left a few full-model logits above 0.15 or removed the speed advantage. Those
failed runs are intentionally not presented as release evidence. Production
`auto` uses the exact Graph route for prefill instead of hiding that result.
