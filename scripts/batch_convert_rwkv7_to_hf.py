#!/usr/bin/env python3
# coding=utf-8
"""Batch-convert RWKV-7 .pth checkpoints and write a SHA256 manifest.

This wrapper keeps conversion orchestration separate from the heavyweight model
load in `convert_rwkv7_to_hf.py`. It can be run in `--dry-run` mode on a laptop
without torch/FLA installed because it only enumerates files, computes SHA256,
builds output paths, and records the commands that would be executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def discover_inputs(input_dir: Path | None, inputs: list[Path], pattern: str) -> list[Path]:
    found: list[Path] = []
    if input_dir is not None:
        found.extend(input_dir.glob(pattern))
    found.extend(inputs)
    unique: dict[Path, Path] = {}
    for path in found:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Input checkpoint not found: {path}")
        unique[resolved] = resolved
    return sorted(unique)


def output_dir_for(input_path: Path, output_root: Path, suffix: str) -> Path:
    return output_root / f"{input_path.stem}{suffix}"


def convert_command(args: argparse.Namespace, input_path: Path, output_path: Path) -> list[str]:
    cmd = [
        args.python,
        str(args.converter),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--precision",
        args.precision,
        "--attn-mode",
        args.attn_mode,
        "--max-shard-size",
        args.max_shard_size,
    ]
    cmd.append("--fuse-norm" if args.fuse_norm else "--no-fuse-norm")
    if args.low_memory:
        cmd.append("--low-memory")
    if args.vocab_file:
        cmd += ["--vocab-file", str(args.vocab_file)]
    return cmd


def entry_base(args: argparse.Namespace, input_path: Path, output_path: Path) -> dict[str, Any]:
    return {
        "source": str(input_path),
        "output": str(output_path),
        "size_bytes": input_path.stat().st_size,
        "sha256": sha256_file(input_path),
        "precision": args.precision,
        "attn_mode": args.attn_mode,
        "fuse_norm": bool(args.fuse_norm),
        "low_memory": bool(args.low_memory),
    }


def load_existing_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, entries: list[dict[str, Any]], append: bool) -> None:
    if append and path.exists():
        existing = load_existing_manifest(path) or {}
        old_entries = existing.get("entries") or []
    else:
        old_entries = []
    manifest = {
        "manifest_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": old_entries + entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def process_one(args: argparse.Namespace, input_path: Path) -> dict[str, Any]:
    output_path = output_dir_for(input_path, args.output_root, args.output_suffix)
    entry = entry_base(args, input_path, output_path)
    cmd = convert_command(args, input_path, output_path)
    entry["command"] = cmd

    if args.dry_run:
        entry["status"] = "dry_run"
        return entry

    if output_path.exists() and not args.force:
        entry["status"] = "skipped"
        entry["reason"] = "output_exists"
        return entry

    output_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    entry["returncode"] = proc.returncode
    entry["stdout"] = proc.stdout[-4000:]
    entry["stderr"] = proc.stderr[-4000:]
    entry["status"] = "converted" if proc.returncode == 0 else "failed"
    return entry


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=None, help="Directory containing official RWKV-7 .pth checkpoints")
    ap.add_argument("--inputs", nargs="*", type=Path, default=[], help="Explicit .pth checkpoint paths")
    ap.add_argument("--glob", default="*.pth", help="Glob used with --input-dir")
    ap.add_argument("--output-root", type=Path, required=True, help="Root directory for generated HF model directories")
    ap.add_argument("--output-suffix", default="-hf", help="Suffix appended to each checkpoint stem")
    ap.add_argument("--manifest", type=Path, default=None, help="Manifest JSON path; defaults to OUTPUT_ROOT/manifest.json")
    ap.add_argument("--append-manifest", action="store_true", help="Append new entries to an existing manifest")
    ap.add_argument("--python", default=sys.executable, help="Python executable used to run the converter")
    ap.add_argument("--converter", type=Path, default=repo_root / "scripts" / "convert_rwkv7_to_hf.py")
    ap.add_argument("--vocab-file", type=Path, default=None)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16", "fp32", "float32"])
    ap.add_argument("--attn-mode", choices=["chunk", "fused_recurrent"], default="chunk")
    norm_group = ap.add_mutually_exclusive_group()
    norm_group.add_argument("--fuse-norm", dest="fuse_norm", action="store_true")
    norm_group.add_argument("--no-fuse-norm", dest="fuse_norm", action="store_false")
    ap.set_defaults(fuse_norm=False)
    ap.add_argument("--max-shard-size", default="1000GB")
    ap.add_argument(
        "--low-memory",
        action="store_true",
        help="Use the meta-template/mmap converter path for 13B-class checkpoints on memory-constrained hosts",
    )
    ap.add_argument("--dry-run", action="store_true", help="Only enumerate, hash, and write the manifest")
    ap.add_argument("--force", action="store_true", help="Convert even if the output directory already exists")
    args = ap.parse_args()

    if args.input_dir is None and not args.inputs:
        ap.error("provide --input-dir and/or --inputs")
    args.converter = args.converter.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.manifest = (args.manifest or (args.output_root / "manifest.json")).expanduser().resolve()
    if args.vocab_file is not None:
        args.vocab_file = args.vocab_file.expanduser().resolve()
    if not args.converter.is_file():
        raise FileNotFoundError(f"Converter script not found: {args.converter}")

    inputs = discover_inputs(args.input_dir, args.inputs, args.glob)
    entries = [process_one(args, path) for path in inputs]
    write_manifest(args.manifest, entries, append=args.append_manifest)
    print(json.dumps({"manifest": str(args.manifest), "entries": len(entries)}, ensure_ascii=False))
    return 1 if any(entry.get("status") == "failed" for entry in entries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
