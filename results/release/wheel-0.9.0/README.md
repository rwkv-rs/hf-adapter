# rwkv7-hf 0.9.0 final-wheel verification

- Runtime/documentation source: `1b5632c2cbd1ecb975d8fa071604827a69038eb8`
- `python -m build`: passed
- `python -m twine check --strict dist/*`: passed
- The final wheel was installed in a new Python 3.12 virtual environment with no source checkout.
- PyTorch `2.13.0+cu126` was installed first so the wheel included `sm_70` support for V100; `clean-venv.json` records the environment and CUDA architectures.
- Exact-revision package-free Hub loading, forward, `RWKV7Cache`, and generation passed (`clean-hub-smoke.json`).
- The installed `rwkv7-hf convert` converted the official 0.1B checkpoint using the default complete `reference` layout and `--low-memory`.
- The installed `rwkv7-hf smoke` loaded that package-free output on Tesla V100 32GB, ran prefill and cached decode, and passed (`smoke.json`).
- The upstream trusted-publishing workflow published 0.9.0 to PyPI. The wheel downloaded back from `https://pypi.org/simple` has SHA256 `7ade2db79e06328f61712c7aa4af97415f2cefaa6a82386c5f86c4c66851acb5`; its CLI and exact-revision Hub CUDA smoke passed (`pypi-install.json`, `pypi-hub-smoke.json`).

The checksum files record the locally validated release distributions and converted model files. GitHub Actions rebuilds version 0.9.0 from the immutable release tag for trusted PyPI publishing.
