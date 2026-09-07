# Published model repositories

**Published reference tag: `v0.9.0`. Planned candidate tag: `v1.0.0` (not
declared published by this document).** Pin `revision="v0.9.0"` for both the
tokenizer and model when reproducing the stable release. The table lists
existing repositories, not proof that their `main` branches contain the 1.0
candidate. Current installation and acceptance status: [HF_STATUS.md](../HF_STATUS.md).

| Model | Parameters | Repository |
|---|---:|---|
| G1d 0.1B | 191M | [wangyue114514/rwkv7-g1d-0.1b-hf](https://huggingface.co/wangyue114514/rwkv7-g1d-0.1b-hf) |
| G1d 0.4B | 451M | [wangyue114514/rwkv7-g1d-0.4b-hf](https://huggingface.co/wangyue114514/rwkv7-g1d-0.4b-hf) |
| G1g 1.5B | 1.53B | [wangyue114514/rwkv7-g1g-1.5b-hf](https://huggingface.co/wangyue114514/rwkv7-g1g-1.5b-hf) |
| G1g 2.9B | 2.95B | [wangyue114514/rwkv7-g1g-2.9b-hf](https://huggingface.co/wangyue114514/rwkv7-g1g-2.9b-hf) |
| G1g 7.2B | 7.20B | [wangyue114514/rwkv7-g1g-7.2b-hf](https://huggingface.co/wangyue114514/rwkv7-g1g-7.2b-hf) |
| G1g 13.3B | 13.27B | [wangyue114514/rwkv7-g1g-13.3b-hf](https://huggingface.co/wangyue114514/rwkv7-g1g-13.3b-hf) |

Each size is an independent HF repository. The planned release `v1.0.0` will update the
self-contained reference code, model card and architecture-only configuration
in place; old tags and weight revisions stay immutable. Safetensors are not
uploaded again when SHA256 matches the frozen pre-release baseline.

The published reference repositories are package-free for normal Transformers
inference. For the candidate, the optional performance path is installed
separately, but the loaded model must already contain the candidate API-v4
boundary. Installing `rwkv7-kernels` does not rewrite a model pinned to
`v0.9.0`. Use candidate-converted local directories for development; do not
advertise `==1.0.0` PyPI installation before its artifacts are published.

## Maintainer-only candidate publication procedure

The commands below are a future release procedure, **not completed actions**.
Do not run `--publish` or create Hub tags while the required gates remain open.

The final release process stages all six repositories from the same tagged
source SHA, commits code/config/model-card changes, creates Hub tag `v1.0.0`,
then redownloads every repository through a new empty cache. Weight hashes,
resolved Hub revisions, finite forward/cache-generation results and the exact
reference class names are retained in the release audit. Hub blob caches and
Transformers remote-code module caches are distinct and empty for every model.
Compare the code actually loaded by `from_pretrained`, not only installed-wheel
metadata; stale `kernel_bridge.py` or other old adapter files cannot establish
the candidate's canonical package-free layout.

The stage manifest is also the publication transaction record: it binds every
small file to SHA256, every existing safetensors shard to its Hub LFS
SHA256/size, all six parent commits, the release tag, and the tagged source
commit. Publishing aborts if a parent, weight, or staged byte changes. The
post-release audit compares Hub `main` and `v1.0.0` against that manifest and
the frozen pre-release weight baseline:

```bash
python scripts/prepare_hf_release.py \
  --output-dir /results/hub-stage-v1.0.0 \
  --source-sha "$FINAL_SOURCE_SHA" \
  --tag v1.0.0

python scripts/publish_hf_release.py \
  --stage-dir /results/hub-stage-v1.0.0 \
  --tag v1.0.0 \
  --publish \
  --output /results/hub-publish-v1.0.0.json

python evaluation/audit_hub_release.py \
  --repo wangyue114514/rwkv7-g1d-0.1b-hf \
  --repo wangyue114514/rwkv7-g1d-0.4b-hf \
  --repo wangyue114514/rwkv7-g1g-1.5b-hf \
  --repo wangyue114514/rwkv7-g1g-2.9b-hf \
  --repo wangyue114514/rwkv7-g1g-7.2b-hf \
  --repo wangyue114514/rwkv7-g1g-13.3b-hf \
  --source-dir rwkv7_hf \
  --revision main \
  --require-tag v1.0.0 \
  --code-sha "$FINAL_SOURCE_SHA" \
  --weight-baseline results/release-preflight/hub-baseline-20260828.json \
  --release-manifest /results/hub-stage-v1.0.0/manifest.json \
  --output /results/hub-v1.0.0.json
```
