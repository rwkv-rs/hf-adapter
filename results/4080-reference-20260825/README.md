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
- reference code: `0f729bb`
- FLA commit: `80e494f6c588e091fc8316b612870df29375c5b8`
- FLA archive SHA256: `e6954fb1b670ef9c580bcebcf68b1bab63922b74fbcdfb3c0f5ba992736bf900`

## Conversion and package-free smoke

The 0.4B round-tripped official checkpoint was converted with the new
`reference` low-memory CLI. The output loaded outside the source checkout via
`AutoConfig`, `AutoModel`, `AutoModelForCausalLM`, and `AutoTokenizer` and
completed cached generation. It included all reference code and the chat
template. The generated safetensors SHA256 exactly matched the existing 0.4B
HF weight file: `8aa0fb580a0c5d442b28f63b9b08c2c60a81821f7c60e79307c11a1a3a2693e0`.

## Pinned FLA FP16 gate

- operator output/state/backward matrix: 6/6 passed for B=1/4 and T=1/17/128;
- cached teacher decode: passed, max abs 0.09375, identical argmax;
- 64-token greedy: 64/64 identical;
- full-model B4/T1 max abs: 0.15625 (limit 0.15);
- full-model B4/T128 max abs: 0.28125 (limit 0.15).

The strict FP16 release gate therefore remains **failed**. No tolerance was
relaxed. The complete raw measurements and command are in
`clean-vs-fla-model-fp16.json`.
