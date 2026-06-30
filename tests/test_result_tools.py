#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from bench.compare_fast_token_layouts import fast_micro_rows, fast_speed_rows, latest_by_layout, load_rows, nested_num, num, ratio


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> int:
    rows = [
        {
            "axis": "speed_mem",
            "backend": "hf_adapter",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "hf_decode_api": "rwkv7_forward_token",
            "decode_tokps": 60.0,
        },
        {
            "axis": "speed_mem",
            "backend": "hf_adapter",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "hf_decode_api": "rwkv7_forward_token",
            "fast_token_layout": "2d",
            "decode_tokps": 66.0,
        },
        {
            "axis": "decode_micro",
            "backend": "hf_adapter",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "fast_decode_api_name": "rwkv7_forward_token",
            "fast_decode_fixed": {"tokps": 59.0},
        },
        {
            "axis": "decode_micro",
            "backend": "hf_adapter",
            "dtype": "fp16",
            "device": "Tesla V100-PCIE-32GB",
            "fast_decode_api_name": "rwkv7_forward_token",
            "fast_token_layout": "2d",
            "fast_decode_fixed": {"tokps": 64.9},
        },
    ]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "results.jsonl"
        write_jsonl(path, rows)
        loaded = load_rows(path)
        passed = subprocess.run(
            [
                sys.executable,
                "bench/compare_fast_token_layouts.py",
                "--results",
                str(path),
                "--device",
                "V100",
                "--dtype",
                "fp16",
                "--require-candidate",
                "--min-speedup",
                "1.0",
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
        assert passed.returncode == 0, passed.stdout + passed.stderr

        missing_path = Path(td) / "missing.jsonl"
        write_jsonl(missing_path, rows[:1])
        failed = subprocess.run(
            [
                sys.executable,
                "bench/compare_fast_token_layouts.py",
                "--results",
                str(missing_path),
                "--device",
                "V100",
                "--dtype",
                "fp16",
                "--require-candidate",
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
        assert failed.returncode != 0
        assert "candidate layout rows missing" in failed.stdout
    args = argparse.Namespace(device="V100", dtype="fp16")
    speeds = latest_by_layout(fast_speed_rows(loaded, args))
    micros = latest_by_layout(fast_micro_rows(loaded, args))
    assert speeds["3d"]["_lineno"] == 1
    assert speeds["2d"]["_lineno"] == 2
    assert round(ratio(num(speeds["2d"], "decode_tokps"), num(speeds["3d"], "decode_tokps")), 4) == 1.1
    assert round(ratio(nested_num(micros["2d"], "fast_decode_fixed", "tokps"), nested_num(micros["3d"], "fast_decode_fixed", "tokps")), 4) == 1.1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
