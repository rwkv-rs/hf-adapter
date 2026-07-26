#!/usr/bin/env python3
"""Stage the six self-contained Hugging Face model repositories for release.

This script never downloads or rewrites model weights.  It resolves each
current Hub commit, copies the reference implementation, updates config.json,
and writes a model card that explicitly documents the package-free contract.
Publishing remains a separate, auditable Hub commit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPOSITORIES = (
    "wangyue114514/rwkv7-g1d-0.1b-hf",
    "wangyue114514/rwkv7-g1d-0.4b-hf",
    "wangyue114514/rwkv7-g1g-1.5b-hf",
    "wangyue114514/rwkv7-g1g-2.9b-hf",
    "wangyue114514/rwkv7-g1g-7.2b-hf",
    "wangyue114514/rwkv7-g1g-13.3b-hf",
)
REFERENCE_FILES = (
    "configuration_rwkv7.py",
    "cache_rwkv7.py",
    "ops_rwkv7.py",
    "modeling_rwkv7.py",
    "tokenization_rwkv7.py",
    "chat_template.jinja",
)
AUTO_MAP = {
    "AutoConfig": "configuration_rwkv7.RWKV7Config",
    "AutoModel": "modeling_rwkv7.RWKV7Model",
    "AutoModelForCausalLM": "modeling_rwkv7.RWKV7ForCausalLM",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weight_rows(siblings) -> dict[str, dict[str, int | str | None]]:
    rows = {}
    for sibling in siblings or []:
        name = str(sibling.rfilename)
        if not name.endswith(".safetensors"):
            continue
        lfs = sibling.lfs
        if isinstance(lfs, dict):
            size = lfs.get("size")
            digest = lfs.get("sha256")
        else:
            size = getattr(lfs, "size", None)
            digest = getattr(lfs, "sha256", None)
        rows[name] = {"size": size, "sha256": digest}
    if not rows or any(
        row["size"] is None or not row["sha256"] for row in rows.values()
    ):
        raise ValueError("Hub weights are missing complete LFS SHA256/size metadata")
    return rows


def git_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def verify_reference_checkout(root: Path, source_sha: str) -> None:
    """Bind staged source bytes to the named checkout commit."""

    head = git_sha(root)
    if source_sha != head or not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError(
            f"source SHA {source_sha!r} does not equal checkout HEAD {head}"
        )
    for name in REFERENCE_FILES:
        disk = (root / "rwkv7_hf" / name).read_bytes()
        committed = subprocess.check_output(
            ["git", "show", f"{source_sha}:rwkv7_hf/{name}"], cwd=root
        )
        if disk != committed:
            raise ValueError(f"reference source differs from {source_sha}: {name}")


def size_label(repo_id: str) -> str:
    match = re.search(r"-(\d+(?:\.\d+)?b)-hf$", repo_id)
    return match.group(1).upper() if match else repo_id.rsplit("/", 1)[-1]


def canonical_config(config: dict) -> dict:
    """Return the release config with no optional-backend policy fields."""

    config = dict(config)
    config.update(
        {
            "model_type": "rwkv7",
            "architectures": ["RWKV7ForCausalLM"],
            "auto_map": AUTO_MAP,
        }
    )
    for name in (
        "attn_mode",
        "fuse_norm",
        "kernel_impl",
        "model_kernel_impl",
        "rwkv7_backend",
    ):
        config.pop(name, None)
    attention_width = int(config.get("attention_hidden_size", config["hidden_size"]))
    config.setdefault("head_dim", 64 if attention_width % 64 == 0 else attention_width)
    config.setdefault("num_heads", attention_width // int(config["head_dim"]))
    config["num_attention_heads"] = int(config["num_heads"])
    return config


def model_card(
    repo_id: str, config: dict, manifest: dict, source_sha: str, tag: str
) -> str:
    source = manifest.get("source", {})
    weights = manifest.get("weights", {})
    family = repo_id.rsplit("/", 1)[-1].split("-")[1].upper()
    parameter_count = weights.get("parameter_count")
    parameters = (
        f"{int(parameter_count):,}"
        if parameter_count is not None
        else "see weights index"
    )
    source_name = source.get("filename", "the original RWKV-7 checkpoint")
    source_hash = source.get("sha256", "recorded in conversion_manifest.json")
    return f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
  - rwkv
  - rwkv7
  - recurrent
  - causal-lm
  - custom_code
---

# RWKV-7 {family} {size_label(repo_id)} — Hugging Face reference model

This repository is a self-contained Hugging Face conversion of
`{source_name}`. Release `{tag}` contains the complete readable,
pure-PyTorch reference implementation next to the weights.  Normal inference
does **not** require `rwkv7-hf`, FLA, a custom CUDA wheel, JIT, or CUDA Graphs.

## Install and use

```bash
python -m pip install torch transformers
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{repo_id}"
revision = "{tag}"
tokenizer = AutoTokenizer.from_pretrained(
    model_id, revision=revision, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id, revision=revision, trust_remote_code=True, torch_dtype="auto"
).eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
inputs = tokenizer("The future of recurrent language models is", return_tensors="pt")
inputs = {{key: value.to(device) for key, value in inputs.items()}}
with torch.inference_mode():
    tokens = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
```

`trust_remote_code=True` loads the checked-in files
`configuration_rwkv7.py`, `cache_rwkv7.py`, `ops_rwkv7.py`,
`modeling_rwkv7.py`, and `tokenization_rwkv7.py`.  The recurrent state has the
canonical `[batch, heads, key, value]` layout.  Cache lifecycle, loss, padding,
generation, gradients, and the layer structure are visible in those files.

## Fine-tuning and evaluation

The model follows the standard `AutoModelForCausalLM` contract and supports
`Trainer`, Accelerate, PEFT LoRA, TRL SFT/DPO/GRPO, and `lm_eval` through the
ordinary Transformers path.  Set `model.config.use_cache = False` during
training.  Reproducible examples and evaluation manifests are in
[rwkv-rs/hf-adapter](https://github.com/rwkv-rs/hf-adapter).

## Optional optimized execution

The checked-in model code is always the readable correctness path. Installing
the separately versioned `rwkv7-kernels` package may replace recurrence or an
inference-only whole-model boundary on supported NVIDIA devices. Formal HF
training always executes one complete readable reference program. The
companion's recurrent, linear, and Mix6 training leaves are isolated
diagnostics only; they are not certified HF training routes.
It does not replace the model/config/cache classes, checkpoint keys, or HF
forward/generation contract. Unsupported
devices, dtypes and shapes fail closed to the corresponding reference tensor
operation.

```bash
python -m pip install "rwkv7-hf==1.0.0" "rwkv7-kernels==1.0.0"
```

Equivalent: `python -m pip install "rwkv7-hf[kernels]==1.0.0"`.

The optional package includes recurrent, fused prefill/decode, projection,
norm, FFN/LoRA, CUDA Graph/state-pool, SM70/Ada/Blackwell, quantization and
training-leaf operator families for isolated diagnostics. Exact supported
routes and required-device evidence are recorded in the source repository;
requested environment settings alone are not accepted as proof that an
optimized route executed.

## Model and provenance

- Architecture: RWKV-7 recurrent causal language model
- Checkpoint family: {family}
- Parameters: {parameters}
- Layers: {config.get("num_hidden_layers")}
- Hidden size: {config.get("hidden_size")}
- Vocabulary size: {config.get("vocab_size")}
- Stored weight dtype: {weights.get("dtype", "recorded in safetensors")}
- Original checkpoint SHA256: `{source_hash}`
- Reference source revision: `{source_sha}`
- Release tag: `{tag}`

`conversion_manifest.json` retains the immutable original conversion and
weight provenance. {tag} changes the checked-in runtime code and config only;
the safetensors bytes are not rewritten or re-uploaded.

## License

Apache-2.0. See `LICENSE`.
"""


def stage(
    repo_id: str,
    output: Path,
    source_root: Path,
    source_sha: str,
    tag: str,
) -> dict:
    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    weights = weight_rows(info.siblings)
    target = output / repo_id.rsplit("/", 1)[-1]
    target.mkdir(parents=True, exist_ok=True)

    config = json.loads(Path(hf_hub_download(repo_id, "config.json")).read_text())
    manifest = json.loads(
        Path(hf_hub_download(repo_id, "conversion_manifest.json")).read_text()
    )
    config = canonical_config(config)
    (target / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (target / "README.md").write_text(
        model_card(repo_id, config, manifest, source_sha, tag), encoding="utf-8"
    )
    for name in REFERENCE_FILES:
        shutil.copy2(source_root / "rwkv7_hf" / name, target / name)
    files = ["README.md", "config.json", *REFERENCE_FILES]
    release = {
        "repo_id": repo_id,
        "parent_commit": info.sha,
        "source_sha": source_sha,
        "tag": tag,
        "files": files,
        "file_sha256": {name: sha256(target / name) for name in files},
        "weights": weights,
        "weights_unchanged": True,
    }
    (target / "release.json").write_text(
        json.dumps(release, indent=2) + "\n", encoding="utf-8"
    )
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", action="append", default=[])
    parser.add_argument("--source-sha", default=None)
    parser.add_argument("--tag", default="v1.0.0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source_sha = args.source_sha or git_sha(root)
    verify_reference_checkout(root, source_sha)
    repositories = tuple(args.repo or REPOSITORIES)
    if len(repositories) != len(set(repositories)) or set(repositories) != set(
        REPOSITORIES
    ):
        raise SystemExit("a release stage must cover each of the six repositories once")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        stage(repo, args.output_dir, root, source_sha, args.tag)
        for repo in repositories
    ]
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "rwkv7-hub-release-stage-v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_sha": source_sha,
                "tag": args.tag,
                "repositories": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "staged", "repositories": len(rows)}))


if __name__ == "__main__":
    main()
