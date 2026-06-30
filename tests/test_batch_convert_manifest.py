#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "batch_convert_rwkv7_to_hf.py"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "weights"
        out = root / "hf"
        src.mkdir()
        a_data = b"rwkv7-a"
        b_data = b"rwkv7-b"
        (src / "rwkv7-g1-0.4b.pth").write_bytes(a_data)
        explicit = root / "rwkv7-g1-1.5b.pth"
        explicit.write_bytes(b_data)
        manifest = root / "manifest.json"

        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--input-dir",
                str(src),
                "--inputs",
                str(explicit),
                "--output-root",
                str(out),
                "--manifest",
                str(manifest),
                "--precision",
                "bf16",
                "--attn-mode",
                "fused_recurrent",
                "--dry-run",
            ],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        status = json.loads(proc.stdout)
        assert status["entries"] == 2
        data = json.loads(manifest.read_text())
        assert data["manifest_version"] == 1
        entries = sorted(data["entries"], key=lambda x: x["source"])
        assert len(entries) == 2
        by_name = {Path(e["source"]).name: e for e in entries}
        a = by_name["rwkv7-g1-0.4b.pth"]
        b = by_name["rwkv7-g1-1.5b.pth"]
        assert a["status"] == "dry_run"
        assert b["status"] == "dry_run"
        assert a["sha256"] == sha256_bytes(a_data)
        assert b["sha256"] == sha256_bytes(b_data)
        assert a["output"].endswith("rwkv7-g1-0.4b-hf")
        assert b["output"].endswith("rwkv7-g1-1.5b-hf")
        assert a["precision"] == "bf16"
        assert a["attn_mode"] == "fused_recurrent"
        assert a["fuse_norm"] is False
        assert "--no-fuse-norm" in a["command"]

        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--inputs",
                str(explicit),
                "--output-root",
                str(out),
                "--manifest",
                str(manifest),
                "--append-manifest",
                "--dry-run",
            ],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        appended = json.loads(manifest.read_text())
        assert len(appended["entries"]) == 3

        missing = subprocess.run(
            [
                sys.executable,
                str(script),
                "--inputs",
                str(root / "missing.pth"),
                "--output-root",
                str(out),
                "--dry-run",
            ],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        assert missing.returncode != 0
        assert "Input checkpoint not found" in (missing.stderr + missing.stdout)

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
