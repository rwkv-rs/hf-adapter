#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path


REQUIRED_CODE = {
    "config.json",
    "configuration_rwkv7.py",
    "cache_rwkv7.py",
    "ops_rwkv7.py",
    "modeling_rwkv7.py",
    "tokenization_rwkv7.py",
    "rwkv_vocab_v20230424.txt",
    "tokenizer_config.json",
    "chat_template.jinja",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Verify an RWKV-7 HF release")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="v1.0.0")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--require-package-free",
        action="store_true",
        help="fail if either rwkv7-hf or rwkv7-kernels is installed",
    )
    parser.add_argument(
        "--require-empty-cache",
        action="store_true",
        help="fail unless --cache-dir is absent or empty before the Hub download",
    )
    return parser.parse_args()


def prepare_cache_dir(cache_dir: Path | None, require_empty: bool) -> dict:
    if require_empty and cache_dir is None:
        raise ValueError("--require-empty-cache requires --cache-dir")
    if cache_dir is None:
        return {"path": None, "was_empty": None}
    cache_dir = cache_dir.expanduser().resolve()
    was_empty = not cache_dir.exists() or not any(cache_dir.iterdir())
    if require_empty and not was_empty:
        raise ValueError(f"Hub smoke cache is not empty: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {"path": str(cache_dir), "was_empty": was_empty}


def installed_rwkv_distributions() -> dict[str, str | None]:
    result = {}
    for name in ("rwkv7-hf", "rwkv7-kernels"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def main():
    args = parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    cache = prepare_cache_dir(args.cache_dir, args.require_empty_cache)
    installed = installed_rwkv_distributions()
    if args.require_package_free and any(installed.values()):
        raise SystemExit(f"RWKV distributions are installed: {installed}")
    download_options = {
        "cache_dir": cache["path"],
        "force_download": args.force_download,
    }
    info = HfApi().model_info(args.model, revision=args.revision, files_metadata=True)
    siblings = {row.rfilename: row for row in info.siblings}
    missing = sorted(REQUIRED_CODE - siblings.keys())
    weights = sorted(name for name in siblings if name.endswith(".safetensors"))
    if missing or not weights:
        raise SystemExit(f"missing={missing} weights={weights}")

    config_path = hf_hub_download(
        args.model,
        "config.json",
        revision=args.revision,
        **download_options,
    )
    config = json.loads(Path(config_path).read_text())
    expected_map = {
        "AutoConfig": "configuration_rwkv7.RWKV7Config",
        "AutoModel": "modeling_rwkv7.RWKV7Model",
        "AutoModelForCausalLM": "modeling_rwkv7.RWKV7ForCausalLM",
    }
    if config.get("model_type") != "rwkv7" or config.get("auto_map") != expected_map:
        raise SystemExit("config.json is not the canonical RWKV-7 HF contract")
    forbidden_config = {
        "attn_mode",
        "fuse_norm",
        "kernel_impl",
        "model_kernel_impl",
        "rwkv7_backend",
    }
    leaked = sorted(forbidden_config & config.keys())
    if leaked:
        raise SystemExit(f"hardware/backend policy leaked into config.json: {leaked}")

    report = {
        "status": "passed",
        "model": args.model,
        "revision": args.revision,
        "commit": info.sha,
        "required_files": sorted(REQUIRED_CODE),
        "weights": [
            {
                "name": name,
                "bytes": siblings[name].size,
                "sha256": (
                    siblings[name].lfs.get("sha256")
                    if isinstance(siblings[name].lfs, dict)
                    else getattr(siblings[name].lfs, "sha256", None)
                ),
            }
            for name in weights
        ],
        "python": platform.python_version(),
        "download": {
            "cache_dir": cache["path"],
            "cache_was_empty": cache["was_empty"],
            "require_empty_cache": args.require_empty_cache,
            "force_download": args.force_download,
        },
        "package_free": {
            "required": args.require_package_free,
            "installed_distributions": installed,
            "passed": not any(installed.values()),
        },
    }

    if not args.metadata_only:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device(args.device)
        dtype = torch.float32 if device.type == "cpu" else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=True,
            **download_options,
        )
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                revision=args.revision,
                trust_remote_code=True,
                torch_dtype=dtype,
                **download_options,
            )
            .to(device)
            .eval()
        )
        encoded = tokenizer("User: Hello! Assistant:", return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model(**encoded, use_cache=True)
            generated = model.generate(**encoded, max_new_tokens=2)
        if not torch.isfinite(output.logits).all():
            raise SystemExit("model produced non-finite logits")
        report.update(
            {
                "model_class": type(model).__name__,
                "cache_class": type(output.past_key_values).__name__,
                "generated": generated[0].tolist(),
            }
        )

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
