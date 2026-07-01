# rwkv7-hf 0.9.0 wheel verification

- Source code: `559a07f2faa458be11fa67a8b36f636ec5b626ba`
- `python -m build`: passed
- `python -m twine check dist/*`: passed
- Wheel installed into a new Python 3.12 virtual environment with no source checkout.
- The installed `rwkv7-hf convert` converted the official 0.1B checkpoint with the default complete `reference` layout and `--low-memory`.
- The installed `rwkv7-hf smoke --model ...` loaded that package-free directory on RTX 4080, ran prefill and cached decode, and passed.
- `smoke.json` records package/runtime versions, generated tokens, timing and memory. The checksum files record both distributions and the converted output.
