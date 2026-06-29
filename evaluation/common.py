from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(model_dir: Path) -> dict:
    files = []
    for path in sorted(model_dir.glob("*.safetensors")):
        files.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    config = model_dir / "config.json"
    return {
        "path": str(model_dir.resolve()),
        "config_sha256": sha256_file(config) if config.is_file() else None,
        "weights": files,
    }


def environment() -> dict:
    import torch
    import transformers

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_bundle(output_dir: Path, name: str, report: dict) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}.json"
    jsonl_path = output_dir / f"{name}.jsonl"
    markdown_path = output_dir / f"{name}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    jsonl_path.write_text(
        json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {name}",
        "",
        f"- status: **{report.get('status', 'unknown')}**",
        f"- model: {report.get('model', {}).get('path')}",
        f"- dtype: {report.get('dtype')}",
        f"- device: {report.get('environment', {}).get('gpu')}",
        f"- code: {report.get('code_sha')}",
        f"- FLA: {report.get('fla_commit')}",
        "",
        "| case | cosine | max abs | mean abs | argmax |",
        "|---|---:|---:|---:|---|",
    ]
    for case, row in report.get("comparisons", {}).items():
        lines.append(
            f"| {case} | {row['cosine']:.8f} | {row['max_abs']:.8f} | "
            f"{row['mean_abs']:.8f} | {row['argmax_same']} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl_path, markdown_path
