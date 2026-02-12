#!/usr/bin/env python3
# coding=utf-8
"""Download and convert the pinned RWKV-7 G1 validation matrix.

Every source checkpoint is size- and SHA256-gated before conversion. Models are
processed sequentially so a validation host can delete each ``.pth`` after its
HF safetensors have been written, avoiding a second full copy of the matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


MODELSCOPE_BASE = "https://modelscope.cn/models/Blink_DL/rwkv7-g1/resolve/master"


@dataclass(frozen=True)
class Checkpoint:
    label: str
    filename: str
    sha256: str
    size_bytes: int
    output_name: str


CHECKPOINTS = (
    Checkpoint(
        "0.4b",
        "rwkv7-g1d-0.4b-20260210-ctx8192.pth",
        "947cb9b8013224e06b112b72204256bec65096cc935a7767ce63d8e3ddef83bb",
        901_776_749,
        "rwkv7-g1d-0.4b-hf",
    ),
    Checkpoint(
        "1.5b",
        "rwkv7-g1h-1.5b-20260710-ctx10240.pth",
        "737079d81865801fd85e5459488d89a36d5304a524e890244eb83d44f531c89c",
        3_055_444_605,
        "rwkv7-g1h-1.5b-hf",
    ),
    Checkpoint(
        "2.9b",
        "rwkv7-g1h-2.9b-20260710-ctx10240.pth",
        "295595b3b8dbff3f8c2a0585975622ddaba4feea7a377022f0bd75347c90c9b3",
        5_896_273_469,
        "rwkv7-g1h-2.9b-hf",
    ),
    Checkpoint(
        "7.2b",
        "rwkv7-g1h-7.2b-20260710-ctx10240.pth",
        "1fe61e5c4b9037ffd4723a11c4de146d99c26bcd89e00a61afa67ef653d215e8",
        14_400_007_869,
        "rwkv7-g1h-7.2b-hf",
    ),
    Checkpoint(
        "13.3b",
        "rwkv7-g1h-13.3b-20260710-ctx10240.pth",
        "5bd705d13497d23530e544d5afb45bdf542b5f67dffee31e3e2b35e4042cfcfb",
        26_540_868_485,
        "rwkv7-g1h-13.3b-hf",
    ),
)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(path: Path, spec: Checkpoint) -> None:
    actual_size = path.stat().st_size
    if actual_size != spec.size_bytes:
        raise RuntimeError(
            f"size mismatch for {path}: expected {spec.size_bytes}, got {actual_size}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != spec.sha256:
        raise RuntimeError(
            f"SHA256 mismatch for {path}: expected {spec.sha256}, got {actual_sha}"
        )


def valid_hf_output(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    weights = list(path.glob("*.safetensors"))
    return bool(weights and all(weight.stat().st_size > 0 for weight in weights))


def run(command: list[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def selected_specs(labels: Iterable[str]) -> list[Checkpoint]:
    requested = set(labels)
    if "all" in requested:
        return list(CHECKPOINTS)
    by_label = {spec.label: spec for spec in CHECKPOINTS}
    unknown = requested.difference(by_label)
    if unknown:
        raise ValueError(f"unknown model labels: {sorted(unknown)}")
    return [spec for spec in CHECKPOINTS if spec.label in requested]


def write_manifest(path: Path, rows: list[dict]) -> None:
    payload = {
        "manifest_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": MODELSCOPE_BASE,
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        choices=["all", *(spec.label for spec in CHECKPOINTS)],
        default=["all"],
    )
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--vocab-file", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument(
        "--attn-mode", choices=["chunk", "fused_recurrent"], default="fused_recurrent"
    )
    ap.add_argument("--max-shard-size", default="5GB")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--force-convert", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    vocab_file = args.vocab_file.expanduser().resolve()
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else output_root / "rwkv7_g1_validation_manifest.json"
    )
    converter = repo_root / "scripts" / "convert_rwkv7_to_hf.py"
    sync_script = repo_root / "scripts" / "sync_hf_adapter_code.py"
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for resumable ModelScope downloads")
    if not args.dry_run and not vocab_file.is_file():
        raise FileNotFoundError(vocab_file)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for spec in selected_specs(args.models):
        source = checkpoint_dir / spec.filename
        output = output_root / spec.output_name
        row = {
            **asdict(spec),
            "source_path": str(source),
            "output_path": str(output),
            "status": "pending",
        }
        rows.append(row)
        write_manifest(manifest, rows)
        try:
            # Resume an interrupted download, but never append to an oversized
            # or checksum-invalid full file. The latter must fail closed.
            needs_download = not source.is_file() or source.stat().st_size < spec.size_bytes
            if needs_download:
                run(
                    [
                        curl,
                        "-L",
                        "--fail",
                        "--retry",
                        "8",
                        "--retry-delay",
                        "3",
                        "--continue-at",
                        "-",
                        "--output",
                        str(source),
                        f"{MODELSCOPE_BASE}/{spec.filename}",
                    ],
                    dry_run=args.dry_run,
                )
            if not args.dry_run:
                validate_checkpoint(source, spec)
            row["checkpoint_verified"] = not args.dry_run

            if args.download_only:
                row["status"] = "downloaded"
                continue
            if valid_hf_output(output) and not args.force_convert:
                row["status"] = "existing_output"
            else:
                if output.exists() and not args.dry_run:
                    shutil.rmtree(output)
                command = [
                    args.python,
                    str(converter),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--vocab-file",
                    str(vocab_file),
                    "--precision",
                    args.precision,
                    "--attn-mode",
                    args.attn_mode,
                    "--no-fuse-norm",
                    "--max-shard-size",
                    args.max_shard_size,
                ]
                if spec.size_bytes >= 8_000_000_000:
                    command.append("--low-memory")
                run(command, dry_run=args.dry_run)
                run(
                    [args.python, str(sync_script), str(output)], dry_run=args.dry_run
                )
                if not args.dry_run and not valid_hf_output(output):
                    raise RuntimeError(f"converter did not create valid output: {output}")
                row["status"] = "converted" if not args.dry_run else "dry_run"
            if not args.keep_checkpoints and not args.dry_run:
                source.unlink(missing_ok=True)
                row["source_deleted"] = True
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            write_manifest(manifest, rows)
            raise
        finally:
            write_manifest(manifest, rows)

    print(json.dumps({"manifest": str(manifest), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
