#!/usr/bin/env python3
from __future__ import annotations

import argparse

from bench.analyze_results import analyze


def row(label: str, hidden_size: int, lineno: int) -> dict:
    return {
        "_lineno": lineno,
        "axis": "larger_model_smoke",
        "backend": "hf_adapter",
        "status": "pass",
        "dtype": "fp16",
        "device": "Tesla V100-PCIE-32GB",
        "model_size_label": label,
        "model_name": f"rwkv7-{label}-hf",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_size_bytes": hidden_size * 1000,
        "vocab_size": 65536,
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 4,
        "num_hidden_layers": 24,
        "head_dim": 64,
        "num_heads": hidden_size // 64,
        "value_dim_first": hidden_size,
        "value_dim_last": hidden_size,
        "value_dim_unique": [hidden_size],
        "generated_tokens": 2,
        "top5": [1, 2, 3, 4, 5],
        "fast_token_backend_effective": "native_graph",
    }


def main() -> int:
    args = argparse.Namespace(
        device="V100",
        dtype="fp16",
        target_prefill_ratio=0.9,
        target_decode_ratio=0.9,
        target_memory_ratio=1.1,
    )
    report = analyze([row("0.4b", 1024, 1), row("1.5b", 2048, 2), row("2.9b", 2560, 3)], args)
    labels = {r["model_size_label"] for r in report["larger_model_smoke"]}
    assert labels == {"0.4b", "1.5b", "2.9b"}
    assert any("0.4B converted HF model loads" in item for item in report["next_focus"])
    assert any("1.5B converted HF model loads" in item for item in report["next_focus"])
    assert any("2.9B converted HF model loads" in item for item in report["next_focus"])
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
