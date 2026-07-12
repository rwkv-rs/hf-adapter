# Published model repositories

| Model | Parameters | Repository |
|---|---:|---|
| G1d 0.1B | 191M | [wangyue114514/rwkv7-g1d-0.1b-hf](https://huggingface.co/wangyue114514/rwkv7-g1d-0.1b-hf) |
| G1d 0.4B | 451M | [wangyue114514/rwkv7-g1d-0.4b-hf](https://huggingface.co/wangyue114514/rwkv7-g1d-0.4b-hf) |
| G1g 1.5B | 1.53B | [wangyue114514/rwkv7-g1g-1.5b-hf](https://huggingface.co/wangyue114514/rwkv7-g1g-1.5b-hf) |
| G1g 2.9B | 2.95B | [wangyue114514/rwkv7-g1g-2.9b-hf](https://huggingface.co/wangyue114514/rwkv7-g1g-2.9b-hf) |
| G1g 7.2B | 7.20B | [wangyue114514/rwkv7-g1g-7.2b-hf](https://huggingface.co/wangyue114514/rwkv7-g1g-7.2b-hf) |
| G1g 13.3B | 13.27B | [wangyue114514/rwkv7-g1g-13.3b-hf](https://huggingface.co/wangyue114514/rwkv7-g1g-13.3b-hf) |

Each size is an independent HF repository. Release `v1.0.0` updates the
self-contained reference code, model card and architecture-only configuration
in place; old tags and weight revisions stay immutable. Safetensors are not
uploaded again when SHA256 matches the frozen pre-release baseline.

Every repository remains package-free for normal Transformers inference. The
optional performance path is installed separately and does not change the Hub
files or model class:

```bash
python -m pip install "rwkv7-hf==1.0.0" "rwkv7-kernels==1.0.0"
```

The equivalent single requirement is
`python -m pip install "rwkv7-hf[kernels]==1.0.0"`.

The final release process stages all six repositories from the same tagged
source SHA, commits code/config/model-card changes, creates Hub tag `v1.0.0`,
then redownloads every repository through a new empty cache. Weight hashes,
resolved Hub revisions, finite forward/cache-generation results and the exact
reference class names are retained in the release audit.
