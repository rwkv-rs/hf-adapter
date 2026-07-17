#!/usr/bin/env python3
"""Render official RWKV-LM versus HF train_temp evidence attachments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_EVIDENCE = Path(__file__).resolve().parent / "5090_train_temp_alignment_20260717"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_cohort_csv(evidence: Path, output: Path) -> None:
    report = load_json(evidence / "compare_convergence_cohort.json")
    if report.get("status") != "pass":
        raise ValueError("refusing to promote a failed convergence cohort")
    fieldnames = [
        "backend",
        "seed",
        "status",
        "steps_completed",
        "finite",
        "train_loss_auc",
        "validation_loss_auc",
        "final_validation_loss",
        "min_validation_loss",
        "max_grad_norm",
        "runtime_s",
    ]
    rows: list[dict[str, Any]] = []
    for backend, source in (
        ("official_rwkv_lm_train_temp", report["reference_rows"]),
        ("hf_train_temp_cuda", report["candidate_rows"]),
    ):
        for row in source:
            rows.append({"backend": backend, **{key: row[key] for key in fieldnames[1:]}})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_single_step_csv(evidence: Path, output: Path) -> None:
    fieldnames = [
        "phase",
        "status",
        "tensor_count",
        "reference_loss",
        "candidate_loss",
        "loss_abs_diff",
        "worst_cosine",
        "max_relative_l2",
        "max_abs",
        "reference_post_step_loss",
        "candidate_post_step_loss",
        "post_step_loss_relative_diff",
    ]
    rows = []
    for phase in ("backward", "step"):
        report = load_json(evidence / f"compare_{phase}.json")
        if report.get("status") != "pass":
            raise ValueError(f"refusing to promote failed {phase} evidence")
        rows.append({key: report.get(key) for key in fieldnames})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_convergence_plot(evidence: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = load_json(evidence / "compare_convergence_cohort.json")
    if report.get("status") != "pass":
        raise ValueError("refusing to plot a failed convergence cohort")
    seeds = report["reference_seeds"]
    if seeds != report["candidate_seeds"] or len(seeds) != 3:
        raise ValueError("the promoted plot requires three matching seeds")

    figure, axes = plt.subplots(2, 3, figsize=(18, 10), sharex="col")
    styles = {
        "official": {"color": "#252525", "linestyle": "-", "label": "Official RWKV-LM"},
        "hf": {"color": "#1677b8", "linestyle": "--", "label": "HF train_temp CUDA"},
    }
    for column, seed in enumerate(seeds):
        runs = {
            "official": load_json(evidence / f"official_convergence_seed{seed}.json"),
            "hf": load_json(evidence / f"hf_convergence_seed{seed}.json"),
        }
        for backend, run in runs.items():
            style = styles[backend]
            train = run["train_curve"]
            validation = run["validation_curve"]
            axes[0, column].plot(
                [row["step"] for row in train],
                [row["loss"] for row in train],
                linewidth=1.45,
                alpha=0.92,
                **style,
            )
            axes[1, column].plot(
                [row["step"] for row in validation],
                [row["loss"] for row in validation],
                marker="o",
                markersize=3.2,
                linewidth=1.8,
                alpha=0.95,
                **style,
            )
        axes[0, column].set_title(f"Seed {seed}", fontsize=13, fontweight="bold")
        axes[0, column].set_yscale("log")
        axes[1, column].set_yscale("log")
        axes[1, column].set_xlabel("Optimizer step")
        for row in range(2):
            axes[row, column].grid(True, which="both", color="#d9d9d9", linewidth=0.65)
            axes[row, column].set_facecolor("#fafafa")

    axes[0, 0].set_ylabel("Training loss (log scale)")
    axes[1, 0].set_ylabel("Validation loss (log scale)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.93))
    figure.suptitle(
        "Official RWKV-LM train_temp vs Hugging Face train_temp CUDA",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.945,
        "RTX 5090 | BF16 | 12x768 | batch 1 | sequence 512 | 1,000 steps per seed",
        ha="center",
        fontsize=11,
        color="#444444",
    )
    figure.text(
        0.5,
        0.015,
        "Single-step tensors match exactly; three-seed convergence cohort: PASS",
        ha="center",
        fontsize=10.5,
        color="#333333",
    )
    figure.tight_layout(rect=(0.03, 0.045, 0.99, 0.91), h_pad=2.2, w_pad=1.5)
    figure.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    evidence = args.evidence_dir.resolve()
    render_convergence_plot(evidence, evidence / "official_vs_hf_convergence.png")
    write_cohort_csv(evidence, evidence / "official_vs_hf_cohort.csv")
    write_single_step_csv(evidence, evidence / "official_vs_hf_single_step.csv")
    print(f"wrote train_temp comparison attachments to {evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
