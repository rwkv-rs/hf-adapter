#!/usr/bin/env python3
"""Build a reviewable backend-v2 evidence bundle without raw samples or weights."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Iterable


EVIDENCE_SUFFIXES = {".json", ".jsonl", ".md", ".toml", ".txt", ".yaml", ".yml"}
BINARY_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".npy",
    ".npz",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".whl",
}
EXCLUDED_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "checkpoints",
    "models",
    "overlays",
    "venvs",
    "wandb",
}
SECRET_PATTERNS = {
    "hugging-face token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    "PyPI token": re.compile(rb"\bpypi-[A-Za-z0-9_-]{20,}\b"),
    "W&B API key": re.compile(rb"\bWANDB_API_KEY\s*="),
    "HF token variable": re.compile(rb"\b(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN)\s*="),
    "bearer authorization": re.compile(rb"(?i)\bauthorization\s*:\s*bearer\s+\S+"),
}


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--harness-sha", required=True)
    parser.add_argument("--max-file-mib", type=float, default=5.0)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exclusion_reason(path: Path) -> str | None:
    parts = path.parts
    if any(
        part in EXCLUDED_DIRECTORIES or part.startswith("checkpoint-") for part in parts
    ):
        return "runtime-or-checkpoint-directory"
    name = path.name
    if name.startswith("._") or name in {".DS_Store"}:
        return "transport-metadata"
    if name.startswith("samples_"):
        return "raw-samples"
    if name.startswith("results_"):
        return "raw-lm-eval-result"
    if (
        name.endswith((".log", ".stdout", ".stderr"))
        or ".stdout." in name
        or ".stderr." in name
    ):
        return "large-runtime-log"
    if "safetensors" in name or path.suffix.lower() in BINARY_SUFFIXES:
        return "weights-or-binary-artifact"
    if path.suffix.lower() not in EVIDENCE_SUFFIXES and name not in {
        "SHA256SUMS",
        "MANIFEST.sha256",
    }:
        return "unsupported-file-type"
    return None


def assert_no_secrets(path: Path, payload: bytes) -> None:
    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(payload):
            raise ValueError(f"refusing to bundle {label} from {path}")


def candidate_files(root: Path) -> tuple[list[Path], Counter[str]]:
    selected: list[Path] = []
    excluded: Counter[str] = Counter()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"symbolic links are not accepted in evidence: {relative}")
        if not path.is_file():
            continue
        reason = exclusion_reason(relative)
        if reason is None:
            selected.append(path)
        else:
            excluded[reason] += 1
    return selected, excluded


def manifest_rows(root: Path, files: Iterable[Path]) -> list[str]:
    return [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(files)
    ]


def validate_bundle(root: Path) -> None:
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        raise ValueError("bundle is missing MANIFEST.sha256")
    expected_paths: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"manifest path is missing or unsafe: {relative}")
        if sha256_file(path) != digest:
            raise ValueError(f"manifest hash mismatch: {relative}")
        expected_paths.add(relative)
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    if actual_paths != expected_paths:
        raise ValueError("manifest does not cover every bundled file")


def build_bundle(args: argparse.Namespace) -> Path:
    source = args.input_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"input directory does not exist: {source}")
    if output == source or output.is_relative_to(source):
        raise ValueError("output directory must not be inside the evidence input")
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    if args.max_file_mib <= 0:
        raise ValueError("--max-file-mib must be positive")

    files, excluded = candidate_files(source)
    size_limit = int(args.max_file_mib * 1024 * 1024)
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=parent
    ) as temp_name:
        staging = Path(temp_name)
        for path in files:
            payload = path.read_bytes()
            relative = path.relative_to(source)
            if len(payload) > size_limit:
                raise ValueError(
                    f"evidence file exceeds {args.max_file_mib:g} MiB: {relative}"
                )
            assert_no_secrets(relative, payload)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            destination.chmod(0o644)

        metadata = {
            "schema": "rwkv7-backend-v2-compact-evidence-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "device": args.device,
            "harness_sha": args.harness_sha,
            "input_dir": str(source),
            "builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "python": sys.version,
                "command": sys.argv,
            },
            "evidence_file_count": len(files),
            "excluded_file_counts": dict(sorted(excluded.items())),
            "policy": {
                "max_file_mib": args.max_file_mib,
                "raw_samples": "excluded",
                "lm_eval_result_payloads": "excluded",
                "stdout_stderr_logs": "excluded",
                "weights_checkpoints_wheels": "excluded",
                "secret_scan": sorted(SECRET_PATTERNS),
            },
        }
        bundle_json = staging / "BUNDLE.json"
        bundle_json.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        bundle_json.chmod(0o644)
        manifest = staging / "MANIFEST.sha256"
        rows = manifest_rows(
            staging, (path for path in staging.rglob("*") if path.is_file())
        )
        manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
        manifest.chmod(0o644)
        validate_bundle(staging)
        staging.rename(output)
    validate_bundle(output)
    return output


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    output = build_bundle(args)
    print(json.dumps({"output": str(output), "status": "passed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
