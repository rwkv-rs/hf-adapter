# RTX 4080 reference-line checkpoint — 2026-08-25

This is an immutable pre-release gate bundle, not a passing release bundle.

## Environment and provenance

- GPU: NVIDIA GeForce RTX 4080, 16 GiB
- driver: 595.84
- Python: 3.12.2
- PyTorch: 2.11.0+cu130
- CUDA runtime: 13.0
- Triton: 3.6.0
- Transformers: 5.8.0
- reference/evaluation code: `f657c67`
- conversion/package-free smoke code: `0f729bb`
- FLA commit: `80e494f6c588e091fc8316b612870df29375c5b8`
- FLA archive SHA256: `e6954fb1b670ef9c580bcebcf68b1bab63922b74fbcdfb3c0f5ba992736bf900`

## Conversion and package-free smoke

The 0.4B round-tripped official checkpoint was converted with the new
`reference` low-memory CLI. The output loaded outside the source checkout via
`AutoConfig`, `AutoModel`, `AutoModelForCausalLM`, and `AutoTokenizer` and
completed cached generation. It included all reference code and the chat
template. The generated safetensors SHA256 exactly matched the existing 0.4B
HF weight file: `8aa0fb580a0c5d442b28f63b9b08c2c60a81821f7c60e79307c11a1a3a2693e0`.

## Pinned FLA gates

All three dtypes were rerun after making cache-state parity part of the gate,
using FP64 accumulation for the reported cosine, applying FP32
`rtol=1e-4, atol=1e-5` via `torch.allclose`, and applying the 0.15 max-absolute
limit only to FP16 logits as specified by the release plan.

| dtype | logits | cache state | operator | greedy 64 | result |
|---|---|---|---|---|---|
| FP16 | failed | passed | 6/6 passed | 64/64 equal | **failed** |
| BF16 | failed | passed | 6/6 passed | different | **failed** |
| FP32 | failed | passed | 0/6 passed | 64/64 equal | **failed** |

### FP16

- operator output/state/backward matrix: 6/6 passed for B=1/4 and T=1/17/128;
- cached teacher decode: passed, max abs 0.09375, identical argmax;
- 64-token greedy: 64/64 identical;
- full-model B4/T1 max abs: 0.15625 (limit 0.15);
- full-model B4/T128 max abs: 0.28125 (limit 0.15).

The strict FP16 release gate therefore remains **failed**. No tolerance was
relaxed. The complete raw measurements and command are in
`clean-vs-fla-model-fp16.json`.

### BF16

The lowest full-model logit cosine was 0.99968572, below 0.9999, and greedy
generation diverged. Cache-state parity and all six operator cases passed. Raw
results are in `bf16/`.

### FP32

FLA emitted its own warning that `ChunkDeltaRuleFunction` does not support
FP32 on some platforms. T=128 full-model comparisons failed, several strict
`torch.allclose` checks failed even at shorter shapes, and all six operator
cases failed. Cache-state parity and greedy generation passed. Raw results are
in `fp32/`.

These results do not identify which model is correct. Pinned FLA warns that its
RWKV7 implementation may be buggy, so the independent official-checkpoint
oracle remains mandatory before changing the reference implementation.
