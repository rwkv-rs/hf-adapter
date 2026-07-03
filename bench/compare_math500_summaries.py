#!/usr/bin/env python3
# coding=utf-8
"""Compare HF-adapter and Albatross MATH500 summary artifacts.

This is intentionally lightweight: it reads the JSON summaries produced by
`bench/eval_math500_hf.py` and BlinkDL/Albatross `eval_math500.py`, optionally
parses the Albatross run log for the steady-state `dynamic done ... decode_s=...`
line, and prints a compact acceptance-oriented comparison.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DYNAMIC_DONE_RE = re.compile(r"dynamic done .*?decode_s=([0-9.]+) tokens=([0-9]+)")


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{path} did not contain a JSON object")
    return obj


def parse_albatross_decode_log(path: str | Path | None) -> dict[str, float] | None:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    matches = DYNAMIC_DONE_RE.findall(text)
    if not matches:
        return None
    decode_s, tokens = matches[-1]
    decode_sec = float(decode_s)
    decoded_tokens = float(tokens)
    return {
        "decode_sec_from_log": decode_sec,
        "decoded_token_events_from_log": decoded_tokens,
        "steady_token_per_sec_from_log": decoded_tokens / decode_sec if decode_sec > 0 else 0.0,
    }


def number(obj: dict[str, Any], key: str, default: float = 0.0) -> float:
    val = obj.get(key, default)
    if val is None:
        return default
    return float(val)


def speed_fields(summary: dict[str, Any], log_decode: dict[str, float] | None = None) -> dict[str, float]:
    decode_sec = number(summary, "decode_sec")
    decoded = number(summary, "decoded_token_events")
    if log_decode is not None:
        decode_sec = log_decode["decode_sec_from_log"]
        decoded = log_decode["decoded_token_events_from_log"]
    steady = decoded / decode_sec if decode_sec > 0 and decoded > 0 else 0.0
    return {
        "token_per_sec_summary": number(summary, "token_per_sec"),
        "sample_per_sec_summary": number(summary, "sample_per_sec"),
        "elapsed_sec": number(summary, "elapsed_sec"),
        "speed_elapsed_sec": number(summary, "speed_elapsed_sec"),
        "wall_token_per_sec": number(summary, "wall_token_per_sec"),
        "generation_token_per_sec": number(summary, "generation_token_per_sec"),
        "decode_sec": decode_sec,
        "decoded_token_events": decoded,
        "steady_token_per_sec": steady,
    }


def ratio(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b


def fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if abs(x) >= 1000:
            return f"{x:,.3f}"
        return f"{x:.6g}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-summary", required=True, help="HF adapter summary.json")
    ap.add_argument("--albatross-summary", required=True, help="Albatross summary.json")
    ap.add_argument("--albatross-log", default="", help="Optional Albatross run.log for steady decode_s/tokens")
    ap.add_argument("--json", action="store_true", help="Emit JSON only")
    args = ap.parse_args()

    hf = load_json(args.hf_summary)
    alb = load_json(args.albatross_summary)
    alb_log = parse_albatross_decode_log(args.albatross_log or None)

    hf_speed = speed_fields(hf)
    alb_speed = speed_fields(alb, alb_log)

    hf_pass = number(hf, "pass_at_rollout_accuracy")
    alb_pass = number(alb, "pass_at_rollout_accuracy")
    hf_rollout_acc = number(hf, "rollout_accuracy")
    alb_rollout_acc = number(alb, "rollout_accuracy")
    hf_correct = number(hf, "correct_generations")
    alb_correct = number(alb, "correct_generations")

    out: dict[str, Any] = {
        "compatible_shape": {
            "num_tasks_match": hf.get("num_tasks") == alb.get("num_tasks"),
            "rollout_match": hf.get("rollout") == alb.get("rollout"),
            "total_generations_match": hf.get("total_generations") == alb.get("total_generations"),
            "hf_num_tasks": hf.get("num_tasks"),
            "albatross_num_tasks": alb.get("num_tasks"),
            "hf_rollout": hf.get("rollout"),
            "albatross_rollout": alb.get("rollout"),
            "hf_total_generations": hf.get("total_generations"),
            "albatross_total_generations": alb.get("total_generations"),
        },
        "accuracy": {
            "hf_correct_generations": hf_correct,
            "albatross_correct_generations": alb_correct,
            "correct_delta": hf_correct - alb_correct,
            "hf_rollout_accuracy": hf_rollout_acc,
            "albatross_rollout_accuracy": alb_rollout_acc,
            "rollout_accuracy_delta": hf_rollout_acc - alb_rollout_acc,
            "hf_pass_at_rollout_accuracy": hf_pass,
            "albatross_pass_at_rollout_accuracy": alb_pass,
            "pass_at_rollout_delta": hf_pass - alb_pass,
        },
        "speed": {
            "hf": hf_speed,
            "albatross": alb_speed,
            "hf_speed_timing": hf.get("speed_timing", "wall"),
            "albatross_speed_timing": alb.get("speed_timing", "wall"),
            "summary_token_per_sec_ratio_hf_over_albatross": ratio(
                hf_speed["token_per_sec_summary"], alb_speed["token_per_sec_summary"]
            ),
            "steady_decode_token_per_sec_ratio_hf_over_albatross": ratio(
                hf_speed["steady_token_per_sec"], alb_speed["steady_token_per_sec"]
            ),
            "sample_per_sec_ratio_hf_over_albatross": ratio(
                hf_speed["sample_per_sec_summary"], alb_speed["sample_per_sec_summary"]
            ),
        },
        "albatross_log_decode": alb_log,
    }

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    shape = out["compatible_shape"]
    print("MATH500 comparison")
    print(f"  shape: tasks {shape['hf_num_tasks']} vs {shape['albatross_num_tasks']}, "
          f"rollout {shape['hf_rollout']} vs {shape['albatross_rollout']}, "
          f"generations {shape['hf_total_generations']} vs {shape['albatross_total_generations']}")
    if not all([shape["num_tasks_match"], shape["rollout_match"], shape["total_generations_match"]]):
        print("  WARNING: shapes differ; do not use this as final acceptance.")
    print("accuracy:")
    acc = out["accuracy"]
    print(f"  correct: HF {fmt(acc['hf_correct_generations'])} vs Albatross {fmt(acc['albatross_correct_generations'])} "
          f"(delta {fmt(acc['correct_delta'])})")
    print(f"  rollout_accuracy: HF {fmt(acc['hf_rollout_accuracy'])} vs Albatross {fmt(acc['albatross_rollout_accuracy'])} "
          f"(delta {fmt(acc['rollout_accuracy_delta'])})")
    print(f"  pass@rollout: HF {fmt(acc['hf_pass_at_rollout_accuracy'])} vs Albatross {fmt(acc['albatross_pass_at_rollout_accuracy'])} "
          f"(delta {fmt(acc['pass_at_rollout_delta'])})")
    print("speed:")
    sp = out["speed"]
    hf_speed_timing = sp["hf_speed_timing"]
    alb_speed_timing = sp["albatross_speed_timing"]
    print(f"  summary token/s: HF {fmt(hf_speed['token_per_sec_summary'])} vs Albatross {fmt(alb_speed['token_per_sec_summary'])} "
          f"(ratio {fmt(sp['summary_token_per_sec_ratio_hf_over_albatross'])})")
    print(f"  summary timing: HF {hf_speed_timing} vs Albatross {alb_speed_timing}")
    if hf_speed["generation_token_per_sec"] or alb_speed["generation_token_per_sec"]:
        print(f"  generation token/s: HF {fmt(hf_speed['generation_token_per_sec'])} vs Albatross {fmt(alb_speed['generation_token_per_sec'])}")
    if hf_speed["wall_token_per_sec"] or alb_speed["wall_token_per_sec"]:
        print(f"  wall token/s: HF {fmt(hf_speed['wall_token_per_sec'])} vs Albatross {fmt(alb_speed['wall_token_per_sec'])}")
    print(f"  steady decode token/s: HF {fmt(hf_speed['steady_token_per_sec'])} vs Albatross {fmt(alb_speed['steady_token_per_sec'])} "
          f"(ratio {fmt(sp['steady_decode_token_per_sec_ratio_hf_over_albatross'])})")
    print(f"  sample/s: HF {fmt(hf_speed['sample_per_sec_summary'])} vs Albatross {fmt(alb_speed['sample_per_sec_summary'])} "
          f"(ratio {fmt(sp['sample_per_sec_ratio_hf_over_albatross'])})")


if __name__ == "__main__":
    main()
