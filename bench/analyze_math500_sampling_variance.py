#!/usr/bin/env python3
# coding=utf-8
"""Estimate MATH500 rollout stochasticity from HF/Albatross generation artifacts.

This is a companion to ``analyze_math500_gap.py`` for the avg@64 parity loop.
It does not require full completions in the committed repo: run it beside the
large ``generations.jsonl`` files and commit only the compact report.

The script answers two questions:

1. Does the pass@64 gap look larger than sampling/refill/RNG variance given the
   observed per-task correct counts?
2. How sensitive are pass rates to sample-id prefix length (pass@1/2/4/.../64)?
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _as_bool(value: Any) -> bool:
    return bool(value)


def load_correct_by_task(path: str | Path) -> dict[int, dict[int, bool]]:
    by_task: dict[int, dict[int, bool]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            task = int(row["task_index"])
            sample = int(row["sample_id"])
            if sample in by_task[task]:
                raise ValueError(f"duplicate task/sample {(task, sample)} in {path} at line {line_no}")
            by_task[task][sample] = _as_bool(row.get("correct", False))
    return dict(by_task)


def _task_counts(by_task: dict[int, dict[int, bool]]) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for task, samples in by_task.items():
        n = len(samples)
        c = sum(int(v) for v in samples.values())
        out[task] = (c, n)
    return out


def _observed_summary(by_task: dict[int, dict[int, bool]]) -> dict[str, Any]:
    counts = _task_counts(by_task)
    total_correct = sum(c for c, _ in counts.values())
    total_rows = sum(n for _, n in counts.values())
    pass_tasks = sum(1 for c, _ in counts.values() if c > 0)
    task_count = len(counts)
    rollout_sizes = Counter(n for _, n in counts.values())
    return {
        "tasks": task_count,
        "rows": total_rows,
        "rollout_sizes": dict(sorted(rollout_sizes.items())),
        "correct_generations": total_correct,
        "rollout_accuracy": total_correct / total_rows if total_rows else 0.0,
        "pass_tasks": pass_tasks,
        "pass_at_rollout": pass_tasks / task_count if task_count else 0.0,
    }


def _prefix_curve(by_task: dict[int, dict[int, bool]], ks: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_count = max(len(by_task), 1)
    for k in ks:
        pass_tasks = 0
        correct = 0
        denom = 0
        for samples in by_task.values():
            selected = [samples[s] for s in sorted(samples)[:k]]
            if not selected:
                continue
            correct += sum(int(v) for v in selected)
            denom += len(selected)
            if any(selected):
                pass_tasks += 1
        rows.append(
            {
                "k": k,
                "pass_tasks": pass_tasks,
                "pass_at_k": pass_tasks / task_count,
                "correct_generations": correct,
                "rollout_accuracy": correct / denom if denom else 0.0,
            }
        )
    return rows


def _quantiles(xs: list[float], qs: list[float]) -> dict[str, float]:
    if not xs:
        return {str(q): 0.0 for q in qs}
    xs = sorted(xs)
    out: dict[str, float] = {}
    n = len(xs)
    for q in qs:
        if n == 1:
            val = xs[0]
        else:
            pos = q * (n - 1)
            lo = int(pos)
            hi = min(lo + 1, n - 1)
            frac = pos - lo
            val = xs[lo] * (1.0 - frac) + xs[hi] * frac
        out[f"p{int(q * 1000):03d}"] = val
    return out


def _empirical_bootstrap(
    hf: dict[int, dict[int, bool]],
    alb: dict[int, dict[int, bool]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap repeated rollout pass tasks from observed per-task outcomes.

    Each task keeps its observed correct rate ``c/n``.  A repeated rollout of the
    same size is simulated as ``Binomial(n, c/n)`` and counted as pass if at
    least one sample is correct.  Tasks with zero observed successes remain zero
    in this conservative empirical model.
    """
    rng = random.Random(seed)
    tasks = sorted(set(hf) & set(alb))
    hf_counts = _task_counts(hf)
    alb_counts = _task_counts(alb)
    hf_pass_draws: list[int] = []
    alb_pass_draws: list[int] = []
    delta_draws: list[int] = []

    def pass_prob(c: int, n: int) -> float:
        return 0.0 if n <= 0 else 1.0 - (1.0 - (c / n)) ** n

    hf_expected_pass = sum(pass_prob(*hf_counts[t]) for t in tasks)
    alb_expected_pass = sum(pass_prob(*alb_counts[t]) for t in tasks)

    for _ in range(samples):
        hp = 0
        ap = 0
        for task in tasks:
            hc, hn = hf_counts[task]
            ac, an = alb_counts[task]
            hp += int(rng.random() < pass_prob(hc, hn))
            ap += int(rng.random() < pass_prob(ac, an))
        hf_pass_draws.append(hp)
        alb_pass_draws.append(ap)
        delta_draws.append(hp - ap)

    observed_hf = _observed_summary(hf)["pass_tasks"]
    observed_alb = _observed_summary(alb)["pass_tasks"]
    observed_delta = int(observed_hf) - int(observed_alb)
    return {
        "method": "empirical per-task Binomial(n, observed_correct_rate) repeated-rollout bootstrap",
        "samples": samples,
        "seed": seed,
        "expected_pass_tasks": {
            "hf": hf_expected_pass,
            "albatross": alb_expected_pass,
            "hf_minus_albatross": hf_expected_pass - alb_expected_pass,
        },
        "observed_pass_tasks": {
            "hf": observed_hf,
            "albatross": observed_alb,
            "hf_minus_albatross": observed_delta,
        },
        "draw_quantiles": {
            "hf_pass_tasks": _quantiles([float(x) for x in hf_pass_draws], [0.025, 0.05, 0.5, 0.95, 0.975]),
            "albatross_pass_tasks": _quantiles([float(x) for x in alb_pass_draws], [0.025, 0.05, 0.5, 0.95, 0.975]),
            "delta_hf_minus_albatross": _quantiles([float(x) for x in delta_draws], [0.025, 0.05, 0.5, 0.95, 0.975]),
        },
        "probabilities": {
            "delta_ge_0": sum(1 for x in delta_draws if x >= 0) / samples,
            "delta_le_observed": sum(1 for x in delta_draws if x <= observed_delta) / samples,
            "abs_delta_ge_observed_abs": sum(1 for x in delta_draws if abs(x) >= abs(observed_delta)) / samples,
        },
    }


def analyze(hf_path: str | Path, alb_path: str | Path, *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    hf = load_correct_by_task(hf_path)
    alb = load_correct_by_task(alb_path)
    common_tasks = sorted(set(hf) & set(alb))
    if set(hf) != set(alb):
        missing = {"missing_in_hf": sorted(set(alb) - set(hf)), "missing_in_albatross": sorted(set(hf) - set(alb))}
    else:
        missing = {"missing_in_hf": [], "missing_in_albatross": []}
    hf_obs = _observed_summary(hf)
    alb_obs = _observed_summary(alb)
    max_rollout = 0
    for task in common_tasks:
        max_rollout = max(max_rollout, len(hf[task]), len(alb[task]))
    ks = [k for k in [1, 2, 4, 8, 16, 32, 64, 128] if k <= max_rollout]
    report = {
        "inputs": {"hf_generations": str(hf_path), "albatross_generations": str(alb_path)},
        "shape": {"common_tasks": len(common_tasks), **missing},
        "observed": {
            "hf": hf_obs,
            "albatross": alb_obs,
            "delta_hf_minus_albatross": {
                "correct_generations": hf_obs["correct_generations"] - alb_obs["correct_generations"],
                "rollout_accuracy": hf_obs["rollout_accuracy"] - alb_obs["rollout_accuracy"],
                "pass_tasks": hf_obs["pass_tasks"] - alb_obs["pass_tasks"],
                "pass_at_rollout": hf_obs["pass_at_rollout"] - alb_obs["pass_at_rollout"],
            },
        },
        "prefix_curves": {"hf": _prefix_curve(hf, ks), "albatross": _prefix_curve(alb, ks)},
        "bootstrap": _empirical_bootstrap(hf, alb, samples=bootstrap_samples, seed=seed),
    }
    return report


def render_markdown(report: dict[str, Any], *, title: str) -> str:
    obs = report["observed"]
    post = report["bootstrap"]
    delta = obs["delta_hf_minus_albatross"]
    lines: list[str] = [f"# {title}", ""]
    lines.append("## Observed full-run gap")
    lines.append("")
    lines.append("| Metric | HF | Albatross | HF - Albatross |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Correct generations | {obs['hf']['correct_generations']} | {obs['albatross']['correct_generations']} | {delta['correct_generations']} |")
    lines.append(f"| Rollout accuracy | {obs['hf']['rollout_accuracy']:.8f} | {obs['albatross']['rollout_accuracy']:.8f} | {delta['rollout_accuracy']:+.8f} |")
    lines.append(f"| Pass tasks | {obs['hf']['pass_tasks']} | {obs['albatross']['pass_tasks']} | {delta['pass_tasks']} |")
    lines.append(f"| Pass@rollout | {obs['hf']['pass_at_rollout']:.6f} | {obs['albatross']['pass_at_rollout']:.6f} | {delta['pass_at_rollout']:+.6f} |")
    lines.append("")
    lines.append("## Prefix pass curve")
    lines.append("")
    lines.append("| k | HF pass@k | Albatross pass@k | HF - Albatross | HF correct | Albatross correct |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    hcur = {r["k"]: r for r in report["prefix_curves"]["hf"]}
    acur = {r["k"]: r for r in report["prefix_curves"]["albatross"]}
    for k in sorted(set(hcur) & set(acur)):
        h = hcur[k]
        a = acur[k]
        lines.append(
            f"| {k} | {h['pass_at_k']:.6f} | {a['pass_at_k']:.6f} | {h['pass_at_k'] - a['pass_at_k']:+.6f} | {h['correct_generations']} | {a['correct_generations']} |"
        )
    lines.append("")
    lines.append("## Empirical stochasticity estimate")
    lines.append("")
    lines.append(f"Method: `{post['method']}` with `{post['samples']}` draws and seed `{post['seed']}`.")
    lines.append("")
    lines.append("| Quantity | Value |")
    lines.append("|---|---:|")
    e = post["expected_pass_tasks"]
    lines.append(f"| Expected HF pass tasks | {e['hf']:.3f} |")
    lines.append(f"| Expected Albatross pass tasks | {e['albatross']:.3f} |")
    lines.append(f"| Expected delta | {e['hf_minus_albatross']:+.3f} |")
    q = post["draw_quantiles"]["delta_hf_minus_albatross"]
    lines.append(f"| Delta draw p2.5 / p50 / p97.5 | {q['p025']:.1f} / {q['p500']:.1f} / {q['p975']:.1f} |")
    p = post["probabilities"]
    lines.append(f"| P(delta >= 0) | {p['delta_ge_0']:.4f} |")
    lines.append(f"| P(delta <= observed delta) | {p['delta_le_observed']:.4f} |")
    lines.append(f"| P(abs(delta) >= abs(observed)) | {p['abs_delta_ge_observed_abs']:.4f} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lo = q["p025"]
    hi = q["p975"]
    if lo <= 0 <= hi:
        lines.append("- The empirical bootstrap 95% interval for pass-task delta includes zero, so the observed pass@64 gap is not strong evidence of a large deterministic model-math mismatch by itself.")
    else:
        lines.append("- The empirical bootstrap 95% interval for pass-task delta excludes zero; investigate deterministic accuracy differences before treating this as sampling noise.")
    lines.append("- Use this together with logits parity and targeted reruns; it is not a substitute for the final full MATH500 avg@64 acceptance run.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-generations", required=True)
    ap.add_argument("--albatross-generations", required=True)
    ap.add_argument("--json-output", default="")
    ap.add_argument("--markdown-output", default="")
    ap.add_argument("--title", default="MATH500 sampling/refill stochasticity analysis")
    ap.add_argument("--bootstrap-samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report = analyze(args.hf_generations, args.albatross_generations, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = render_markdown(report, title=args.title)
    if args.markdown_output:
        Path(args.markdown_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_output).write_text(md, encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(md)


if __name__ == "__main__":
    main()
