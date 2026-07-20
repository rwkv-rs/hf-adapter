# Hub release preflight

`hub-baseline-20260828.json` is the immutable **pre-release** snapshot used to
prove that the six model weight payloads do not change during the v1.0 update.
Its overall status is intentionally `failed`: all six repositories still carry
the v0.9 canonical source files and therefore differ from the v1.0
source-of-truth in `rwkv7_hf/`.

The final audit must use this file as `--weight-baseline`, require the release
tag, and exit zero:

```bash
python evaluation/audit_hub_release.py \
  --code-sha "$RELEASE_SHA" \
  --source-dir rwkv7_hf \
  --weight-baseline results/release-preflight/hub-baseline-20260828.json \
  --require-tag v1.0.0 \
  --output results/release-preflight/hub-v1.0.0-final.json \
  --repo wangyue114514/rwkv7-g1d-0.1b-hf \
  --repo wangyue114514/rwkv7-g1d-0.4b-hf \
  --repo wangyue114514/rwkv7-g1g-1.5b-hf \
  --repo wangyue114514/rwkv7-g1g-2.9b-hf \
  --repo wangyue114514/rwkv7-g1g-7.2b-hf \
  --repo wangyue114514/rwkv7-g1g-13.3b-hf
```

The validator compares all six canonical remote-code files byte-for-byte,
checks the standard `auto_map`, rejects legacy/duplicate model modules, records
the Hub revision and tag target, and compares every safetensors LFS SHA256 and
size with the pre-release snapshot without downloading the weights.
