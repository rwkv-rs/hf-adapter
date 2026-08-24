from __future__ import annotations

import json
from pathlib import Path

from bench.analyzers.summarize_4090_4080_routes import MODELS, summarize


DEVICE = "NVIDIA GeForce RTX 4090"


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def matrices(root: Path) -> tuple[Path, Path, Path]:
    bmm = []
    accum = []
    policy = []
    for (hidden, layers), name in MODELS.items():
        for _ in range(3):
            bmm.append(
                {
                    "device": DEVICE,
                    "hidden_size": hidden,
                    "num_hidden_layers": layers,
                    "batch_size": 8,
                    "status": "pass",
                    "correctness_pass": True,
                    "greedy_match": 1024,
                    "greedy_total": 1024,
                    "min_cosine_first_step": 1.0,
                    "speedup": 1.1,
                    "vram_delta_mb": 10.0,
                }
            )
        for batch in (1, 8):
            for prompt in (128, 512, 2048):
                for order in (0, 1):
                    for mode in ("off", "global", "block"):
                        accum.append(
                            {
                                "device": DEVICE,
                                "hidden_size": hidden,
                                "num_hidden_layers": layers,
                                "model_size_label": name,
                                "batch_size": batch,
                                "prompt_tokens": prompt,
                                "order_index": order,
                                "mode": mode,
                                "status": "pass",
                                "route_effective_match": True,
                                "prompt_greedy_match": True,
                                "decode_greedy_match": True,
                                "speedup_vs_off": 1.0 if mode == "off" else 1.1,
                                "prompt_min_cosine": 1.0,
                                "decode_min_cosine": 1.0,
                            }
                        )
                policy.append(
                    {
                        "device": DEVICE,
                        "hidden_size": hidden,
                        "num_hidden_layers": layers,
                        "batch_size": batch,
                        "prompt_tokens": prompt,
                        "status": "pass",
                        "prefill_block_fp16_accum_effective": True,
                        "prefill_global_fp16_accum_effective": False,
                        "greedy_match": True,
                        "decode_after_prefill_greedy_match": True,
                        "min_cosine": 1.0,
                        "decode_after_prefill_min_cosine": 1.0,
                    }
                )
    return (
        write_rows(root / "bmm.jsonl", bmm),
        write_rows(root / "accum.jsonl", accum),
        write_rows(root / "policy.jsonl", policy),
    )


def test_summary_passes_complete_exact_card_matrix(tmp_path: Path) -> None:
    report = summarize(*matrices(tmp_path))
    assert report["status"] == "pass"
    assert report["bmm"]["pass"]
    assert report["block_fp16_accumulation"]["pass"]
    assert report["default_policy_prefill"]["pass"]


def test_summary_fails_one_nonpositive_block_speedup(tmp_path: Path) -> None:
    bmm, accum, policy = matrices(tmp_path)
    rows = [json.loads(line) for line in accum.read_text().splitlines()]
    next(row for row in rows if row["mode"] == "block")["speedup_vs_off"] = 0.99
    write_rows(accum, rows)
    report = summarize(bmm, accum, policy)
    assert report["status"] == "fail"
    assert any("accumulation" in error and "failed" in error for error in report["errors"])


def test_summary_accepts_policy_rows_with_model_label_only(tmp_path: Path) -> None:
    bmm, accum, policy = matrices(tmp_path)
    rows = [json.loads(line) for line in policy.read_text().splitlines()]
    for row in rows:
        row["model_size_label"] = MODELS[
            (row.pop("hidden_size"), row.pop("num_hidden_layers"))
        ]
    write_rows(policy, rows)
    assert summarize(bmm, accum, policy)["status"] == "pass"
