#!/usr/bin/env python3
"""Render official RWKV-LM versus HF train_temp evidence attachments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_EVIDENCE = (
    Path(__file__).resolve().parent / "5090_train_temp_alignment_20260717"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def backend_label(backend: str) -> str:
    labels = {
        "official_train_temp": "Official RWKV-LM",
        "hf_train_temp_cuda": "HF train_temp CUDA",
        "native_train_temp_cuda": "Native HF train_temp CUDA",
        "hf_native_train_temp_cuda": "Native HF train_temp CUDA",
    }
    return labels.get(backend, backend.replace("_", " "))


def run_subtitle(run: dict[str, Any], *, steps: int) -> str:
    gpu = run.get("gpu_name", "GPU")
    precision = str(run.get("precision", "unknown")).upper()
    batch_size = run.get("batch_size", "?")
    seq_len = run.get("seq_len", "?")
    return (
        f"{gpu} | {precision} | batch {batch_size} | sequence {seq_len} | "
        f"{steps:,} steps per seed"
    )


def write_cohort_csv(
    evidence: Path,
    output: Path,
    *,
    cohort_report: str = "compare_convergence_cohort.json",
) -> None:
    report = load_json(evidence / cohort_report)
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
        (report["reference_backend"], report["reference_rows"]),
        (report["candidate_backend"], report["candidate_rows"]),
    ):
        for row in source:
            rows.append(
                {"backend": backend, **{key: row[key] for key in fieldnames[1:]}}
            )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_single_step_csv(
    evidence: Path,
    output: Path,
    *,
    backward_report: str = "compare_backward.json",
    step_report: str = "compare_step.json",
) -> None:
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
    for phase, filename in (
        ("backward", backward_report),
        ("step", step_report),
    ):
        report = load_json(evidence / filename)
        if report.get("status") != "pass":
            raise ValueError(f"refusing to promote failed {phase} evidence")
        rows.append({key: report.get(key) for key in fieldnames})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_best_observed_plot(
    evidence: Path,
    output: Path,
    *,
    reference_template: str = "official_convergence_seed{seed}.json",
    candidate_template: str = "hf_convergence_seed{seed}.json",
    cohort_report: str = "compare_convergence_cohort.json",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = load_json(evidence / cohort_report)
    if report.get("status") != "pass":
        raise ValueError("refusing to plot a failed convergence cohort")

    reference = {row["seed"]: row for row in report["reference_rows"]}
    candidate = {row["seed"]: row for row in report["candidate_rows"]}
    if reference.keys() != candidate.keys():
        raise ValueError("the presentation plot requires matching paired seeds")
    seed = min(
        reference,
        key=lambda item: (
            candidate[item]["final_validation_loss"]
            / max(reference[item]["final_validation_loss"], 1e-12)
        ),
    )
    runs = {
        "official": load_json(evidence / reference_template.format(seed=seed)),
        "hf": load_json(evidence / candidate_template.format(seed=seed)),
    }
    styles = {
        "official": {
            "color": "#252525",
            "linestyle": "-",
            "label": backend_label(report["reference_backend"]),
        },
        "hf": {
            "color": "#1677b8",
            "linestyle": "--",
            "label": backend_label(report["candidate_backend"]),
        },
    }

    figure, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for backend, run in runs.items():
        style = styles[backend]
        train = run["train_curve"]
        validation = run["validation_curve"]
        axes[0].plot(
            [row["step"] for row in train],
            [row["loss"] for row in train],
            linewidth=1.5,
            alpha=0.92,
            **style,
        )
        axes[1].plot(
            [row["step"] for row in validation],
            [row["loss"] for row in validation],
            marker="o",
            markersize=4,
            linewidth=2,
            alpha=0.95,
            **style,
        )

    axes[0].set_title("Training loss", fontsize=13, fontweight="bold")
    axes[1].set_title("Validation loss", fontsize=13, fontweight="bold")
    for axis in axes:
        axis.set_yscale("log")
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Loss (log scale)")
        axis.grid(True, which="both", color="#d9d9d9", linewidth=0.65)
        axis.set_facecolor("#fafafa")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.89),
    )
    figure.suptitle(
        f"Best observed paired run: Seed {seed}",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.925,
        run_subtitle(runs["official"], steps=int(runs["official"]["steps_completed"])),
        ha="center",
        fontsize=11,
        color="#444444",
    )
    figure.text(
        0.5,
        0.015,
        "Presentation view selected by final validation-loss ratio; not the three-seed aggregate",
        ha="center",
        fontsize=10.5,
        color="#333333",
    )
    figure.tight_layout(rect=(0.03, 0.055, 0.99, 0.86), w_pad=2.2)
    figure.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_convergence_plot(
    evidence: Path,
    output: Path,
    *,
    reference_template: str = "official_convergence_seed{seed}.json",
    candidate_template: str = "hf_convergence_seed{seed}.json",
    cohort_report: str = "compare_convergence_cohort.json",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = load_json(evidence / cohort_report)
    if report.get("status") != "pass":
        raise ValueError("refusing to plot a failed convergence cohort")
    seeds = report["reference_seeds"]
    if seeds != report["candidate_seeds"] or len(seeds) != 3:
        raise ValueError("the promoted plot requires three matching seeds")

    figure, axes = plt.subplots(2, 3, figsize=(18, 10), sharex="col")
    styles = {
        "official": {
            "color": "#252525",
            "linestyle": "-",
            "label": backend_label(report["reference_backend"]),
        },
        "hf": {
            "color": "#1677b8",
            "linestyle": "--",
            "label": backend_label(report["candidate_backend"]),
        },
    }
    first_reference: dict[str, Any] | None = None
    for column, seed in enumerate(seeds):
        runs = {
            "official": load_json(evidence / reference_template.format(seed=seed)),
            "hf": load_json(evidence / candidate_template.format(seed=seed)),
        }
        if first_reference is None:
            first_reference = runs["official"]
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
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    figure.suptitle(
        f"{backend_label(report['reference_backend'])} vs "
        f"{backend_label(report['candidate_backend'])}",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.945,
        run_subtitle(
            first_reference or {},
            steps=int((first_reference or {}).get("steps_completed", 0)),
        ),
        ha="center",
        fontsize=11,
        color="#444444",
    )
    figure.text(
        0.5,
        0.015,
        "Strict single-step tensors are reported separately; three-seed cohort: PASS",
        ha="center",
        fontsize=10.5,
        color="#333333",
    )
    figure.tight_layout(rect=(0.03, 0.045, 0.99, 0.91), h_pad=2.2, w_pad=1.5)
    figure.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_long_run_plot(
    evidence: Path,
    output: Path,
    *,
    reference_json: str,
    candidate_json: str,
    resumed_json: str | None = None,
) -> None:
    """Render uninterrupted official/Native curves and optional resumed Native."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [
        (
            backend_label("official_train_temp"),
            load_json(evidence / reference_json),
            {"color": "#252525", "linestyle": "-", "linewidth": 1.7},
        ),
        (
            backend_label("hf_native_train_temp_cuda"),
            load_json(evidence / candidate_json),
            {"color": "#1677b8", "linestyle": "--", "linewidth": 1.7},
        ),
    ]
    if resumed_json:
        runs.append(
            (
                "Native HF resumed",
                load_json(evidence / resumed_json),
                {"color": "#c45a16", "linestyle": ":", "linewidth": 1.6},
            )
        )
    if any(run.get("status") != "pass" for _, run, _ in runs):
        raise ValueError("refusing to plot a failed long convergence run")

    figure, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for label, run, style in runs:
        train = run["train_curve"]
        validation = run["validation_curve"]
        axes[0].plot(
            [row["step"] for row in train],
            [row["loss"] for row in train],
            label=label,
            alpha=0.82,
            **style,
        )
        axes[1].plot(
            [row["step"] for row in validation],
            [row["loss"] for row in validation],
            label=label,
            marker="o",
            markersize=2.8,
            alpha=0.94,
            **style,
        )

    axes[0].set_title("Training loss", fontsize=13, fontweight="bold")
    axes[1].set_title("Held-out validation loss", fontsize=13, fontweight="bold")
    for axis in axes:
        axis.set_yscale("log")
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Loss (log scale)")
        axis.grid(True, which="both", color="#d9d9d9", linewidth=0.65)
        axis.set_facecolor("#fafafa")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(runs),
        frameon=False,
        bbox_to_anchor=(0.5, 0.90),
    )
    reference = runs[0][1]
    figure.suptitle(
        "Official RWKV-LM vs Native HF: long real-data run",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.93,
        run_subtitle(reference, steps=int(reference["steps_completed"])),
        ha="center",
        fontsize=11,
        color="#444444",
    )
    figure.text(
        0.5,
        0.015,
        "The resumed line reloads model, optimizer and RNG state at step 2,500",
        ha="center",
        fontsize=10.5,
        color="#333333",
    )
    figure.tight_layout(rect=(0.03, 0.055, 0.99, 0.87), w_pad=2.2)
    figure.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument(
        "--output-stem",
        help="Use generic <stem>_{best_observed,convergence,cohort,single_step} names.",
    )
    parser.add_argument(
        "--reference-template",
        default="official_convergence_seed{seed}.json",
        help="Reference convergence filename template relative to the evidence directory.",
    )
    parser.add_argument(
        "--candidate-template",
        default="hf_convergence_seed{seed}.json",
        help="Candidate convergence filename template relative to the evidence directory.",
    )
    parser.add_argument("--cohort-report", default="compare_convergence_cohort.json")
    parser.add_argument("--backward-report", default="compare_backward.json")
    parser.add_argument("--step-report", default="compare_step.json")
    parser.add_argument("--long-reference")
    parser.add_argument("--long-candidate")
    parser.add_argument("--long-resumed")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    evidence = args.evidence_dir.resolve()
    if args.output_stem:
        outputs = {
            "best": evidence / f"{args.output_stem}_best_observed.png",
            "convergence": evidence / f"{args.output_stem}_convergence.png",
            "cohort": evidence / f"{args.output_stem}_cohort.csv",
            "single_step": evidence / f"{args.output_stem}_single_step.csv",
            "long": evidence / f"{args.output_stem}_long_run.png",
        }
    else:
        outputs = {
            "best": evidence / "official_vs_hf_best_seed131.png",
            "convergence": evidence / "official_vs_hf_convergence.png",
            "cohort": evidence / "official_vs_hf_cohort.csv",
            "single_step": evidence / "official_vs_hf_single_step.csv",
            "long": evidence / "official_vs_hf_long_run.png",
        }
    render_best_observed_plot(
        evidence,
        outputs["best"],
        reference_template=args.reference_template,
        candidate_template=args.candidate_template,
        cohort_report=args.cohort_report,
    )
    render_convergence_plot(
        evidence,
        outputs["convergence"],
        reference_template=args.reference_template,
        candidate_template=args.candidate_template,
        cohort_report=args.cohort_report,
    )
    write_cohort_csv(evidence, outputs["cohort"], cohort_report=args.cohort_report)
    write_single_step_csv(
        evidence,
        outputs["single_step"],
        backward_report=args.backward_report,
        step_report=args.step_report,
    )
    if bool(args.long_reference) != bool(args.long_candidate):
        raise ValueError("--long-reference and --long-candidate must be provided together")
    if args.long_reference and args.long_candidate:
        render_long_run_plot(
            evidence,
            outputs["long"],
            reference_json=args.long_reference,
            candidate_json=args.long_candidate,
            resumed_json=args.long_resumed,
        )
    print(f"wrote train_temp comparison attachments to {evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
