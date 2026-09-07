# Public Hugging Face release acceptance — 2026-08-19

This retained bundle verifies the public `rwkv7-hf==0.7.0` thin-model release
after the six model repositories and their Collection were finalized.

## Gates

| Gate | Result | Artifact |
|---|---|---|
| Clean PyPI install plus real 0.1B Hub download, load, forward, and generation | PASS | [`full_0p1b.json`](full_0p1b.json) |
| Six-model config, tokenizer, manifest, revision, and remote LFS SHA/size matrix | PASS (6/6) | [`metadata_matrix.json`](metadata_matrix.json) |
| 7.2B sharded load and one-token forward | PASS | model repository `conversion_manifest.json` |
| 13.3B sharded load and one-token forward | PASS | model repository `conversion_manifest.json` |

The full clean-install gate ran in a new Python 3.10 virtual environment with
`rwkv7-hf 0.7.0` installed from PyPI, PyTorch `2.12.1+cu130`, Transformers
`5.13.0`, no local adapter source on `PYTHONPATH`, and CUDA hidden so the model
executed on CPU. It resolved the public model tag to commit
`a5d124a4697978f4461d84d849b0aae5937da522` and generated `Hello, I'm a`.

## Reproduction

```bash
python -m venv .venv-public-verify
source .venv-public-verify/bin/activate
python -m pip install "rwkv7-hf==0.7.0"
python scripts/verify_hf_release.py \
  --model wangyue114514/rwkv7-g1d-0.1b-hf \
  --revision v0.7.0
```

For each larger model, the retained metadata gate used:

```bash
python scripts/verify_hf_release.py \
  --model MODEL_ID \
  --revision v0.7.0 \
  --metadata-only
```

Success is accepted only when the command exits zero and ends with
`RESULT: PASS`.
