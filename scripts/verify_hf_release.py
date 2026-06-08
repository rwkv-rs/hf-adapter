#!/usr/bin/env python3
"""Verify a published RWKV-7 Hugging Face model from metadata to generation.

The default command performs a real Hub download, loads the pinned PyPI runtime,
checks the repository manifest and remote LFS metadata, runs one forward pass,
and generates a short continuation. Use ``--metadata-only`` for large models
when downloading their full weights is intentionally deferred.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar


DEFAULT_MODEL = "wangyue114514/rwkv7-g1d-0.1b-hf"
DEFAULT_REVISION = "v0.7.0"
DEFAULT_RUNTIME = "0.8.1"
DEFAULT_MANIFEST_RUNTIME = "0.7.0"
T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a public RWKV-7 HF repository and optionally run generation."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID")
    parser.add_argument(
        "--revision", default=DEFAULT_REVISION, help="Hub revision or tag"
    )
    parser.add_argument(
        "--expected-runtime-version",
        default=DEFAULT_RUNTIME,
        help="Required installed rwkv7-hf version",
    )
    parser.add_argument(
        "--expected-manifest-runtime-version",
        default=DEFAULT_MANIFEST_RUNTIME,
        help="Publishing runtime recorded by the immutable conversion manifest",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Verify config, tokenizer, manifest, and remote LFS metadata without weights",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device for the full forward/generation gate",
    )
    parser.add_argument(
        "--prompt", default="Hello", help="Generation smoke-test prompt"
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--network-attempts",
        type=int,
        default=3,
        help="Attempts for Hub metadata and file operations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON result; stdout is always emitted",
    )
    return parser.parse_args()


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def network_retry(label: str, attempts: int, operation: Callable[[], T]) -> T:
    require(attempts >= 1, "--network-attempts must be at least 1")
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception:
            if attempt == attempts:
                raise
            delay = 2 ** (attempt - 1)
            print(
                f"network retry {attempt}/{attempts - 1} for {label} in {delay}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def verify_remote_weights(info: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    siblings = {entry.rfilename: entry for entry in info.siblings}
    weights = manifest["weights"]
    layout = weights.get("layout")
    if layout is None and "filename" in weights:
        # Early v0.7.0 manifests predate the explicit layout field.
        layout = "single"

    if layout == "single":
        filename = weights["filename"]
        require(filename in siblings, f"missing remote weight file: {filename}")
        remote = siblings[filename]
        require(remote.size == weights["size_bytes"], f"size mismatch for {filename}")
        require(remote.lfs is not None, f"missing LFS metadata for {filename}")
        require(
            remote.lfs.sha256 == weights["sha256"], f"SHA256 mismatch for {filename}"
        )
        return {"layout": layout, "files": 1, "size_bytes": remote.size}

    require(layout == "sharded", f"unsupported weight layout: {layout}")
    index_filename = weights["index_filename"]
    require(index_filename in siblings, f"missing remote index: {index_filename}")
    total = 0
    for shard in weights["shards"]:
        filename = shard["filename"]
        require(filename in siblings, f"missing remote shard: {filename}")
        remote = siblings[filename]
        require(remote.size == shard["size_bytes"], f"size mismatch for {filename}")
        require(remote.lfs is not None, f"missing LFS metadata for {filename}")
        require(remote.lfs.sha256 == shard["sha256"], f"SHA256 mismatch for {filename}")
        total += remote.size
    return {"layout": layout, "files": len(weights["shards"]), "size_bytes": total}


def main() -> int:
    args = parse_args()
    started = time.time()

    import torch
    import transformers
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    try:
        runtime_version = importlib.metadata.version("rwkv7-hf")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError(
            f"rwkv7-hf is not installed; run: python -m pip install "
            f"'rwkv7-hf=={args.expected_runtime_version}'"
        ) from None
    require(
        runtime_version == args.expected_runtime_version,
        f"rwkv7-hf=={args.expected_runtime_version} required, found {runtime_version}",
    )

    api = HfApi()
    info = network_retry(
        "model metadata",
        args.network_attempts,
        lambda: api.model_info(args.model, revision=args.revision, files_metadata=True),
    )
    manifest_path = network_retry(
        "conversion manifest",
        args.network_attempts,
        lambda: hf_hub_download(
            args.model, "conversion_manifest.json", revision=args.revision
        ),
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    require(
        manifest["runtime"]["version"] == args.expected_manifest_runtime_version,
        "manifest publishing runtime mismatch",
    )
    remote_weights = verify_remote_weights(info, manifest)

    config = network_retry(
        "model config",
        args.network_attempts,
        lambda: AutoConfig.from_pretrained(
            args.model, revision=args.revision, trust_remote_code=True
        ),
    )
    tokenizer = network_retry(
        "tokenizer",
        args.network_attempts,
        lambda: AutoTokenizer.from_pretrained(
            args.model, revision=args.revision, trust_remote_code=True
        ),
    )
    result: dict[str, Any] = {
        "status": "passed",
        "mode": "metadata-only" if args.metadata_only else "full",
        "model_id": args.model,
        "revision": args.revision,
        "resolved_commit": info.sha,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "rwkv7_hf": runtime_version,
        "config_class": f"{config.__class__.__module__}.{config.__class__.__name__}",
        "tokenizer_class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}",
        "vocabulary_size": len(tokenizer),
        "parameter_count_manifest": manifest["weights"]["parameter_count"],
        "remote_weights": remote_weights,
    }

    if not args.metadata_only:
        device = choose_device(torch, args.device)
        model, loading_info = network_retry(
            "model weights",
            args.network_attempts,
            lambda: AutoModelForCausalLM.from_pretrained(
                args.model,
                revision=args.revision,
                trust_remote_code=True,
                dtype="auto",
                output_loading_info=True,
            ),
        )
        model = model.to(device).eval()
        input_ids = torch.tensor([[1]], device=device)
        encoded = tokenizer(args.prompt, return_tensors="pt")
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=True).logits
            generated = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens, do_sample=False
            )

        missing = len(loading_info.get("missing_keys", []))
        unexpected = len(loading_info.get("unexpected_keys", []))
        mismatched = len(loading_info.get("mismatched_keys", []))
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        require(missing == 0, f"missing keys: {missing}")
        require(unexpected == 0, f"unexpected keys: {unexpected}")
        require(mismatched == 0, f"mismatched keys: {mismatched}")
        require(torch.isfinite(logits).all().item(), "non-finite logits")
        require(
            parameter_count == manifest["weights"]["parameter_count"],
            "loaded parameter count does not match manifest",
        )
        result.update(
            {
                "device": device,
                "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
                "parameter_count_loaded": parameter_count,
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "mismatched_keys": mismatched,
                "logits_shape": list(logits.shape),
                "logits_all_finite": True,
                "generated_token_ids": generated[0].tolist(),
                "generated_text": tokenizer.decode(
                    generated[0], skip_special_tokens=True
                ),
            }
        )

    result["elapsed_seconds"] = round(time.time() - started, 3)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"RESULT: FAIL: {exc}", file=sys.stderr)
        raise
