from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "bench" / "5090_train_temp_alignment_20260717"


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_promoted_train_temp_evidence_is_self_consistent() -> None:
    summary = load("summary.json")
    environment = load("environment.json")
    backward = load("compare_backward.json")
    step = load("compare_step.json")
    cohort = load("compare_convergence_cohort.json")

    assert (EVIDENCE / "compile.exit").read_text(encoding="utf-8").strip() == "0"
    assert {
        summary["status"],
        backward["status"],
        step["status"],
        cohort["status"],
    } == {"pass"}
    assert summary["provenance"]["official_commit"] == environment["official_commit"]
    assert (
        summary["provenance"]["checkpoint_sha256"]
        == environment["checkpoint_sha256"]
    )

    assert summary["backward"]["tensor_count"] == backward["tensor_count"] == 400
    assert backward["worst_cosine"] == 1.0
    assert backward["max_relative_l2"] == backward["max_abs"] == 0.0

    assert summary["optimizer_step"]["tensor_count"] == step["tensor_count"] == 800
    assert step["optimizer_groups_match"] is True
    assert step["post_step_loss_relative_diff"] == 0.0
    assert step["worst_cosine"] == 1.0
    assert step["max_relative_l2"] == step["max_abs"] == 0.0

    convergence = summary["convergence_cohort"]
    assert (
        cohort["reference_seeds"]
        == cohort["candidate_seeds"]
        == convergence["seeds"]
    )
    assert cohort["runs_complete"] is True
    assert (
        cohort["reference_success_count"]
        == convergence["reference_success_count"]
        == 2
    )
    assert (
        cohort["candidate_success_count"]
        == convergence["candidate_success_count"]
        == 2
    )
    assert cohort["reference_deep_success_count"] == 2
    assert cohort["candidate_deep_success_count"] == 2
    assert cohort["median_train_loss_auc_relative_diff"] < 0.01
    assert cohort["median_validation_loss_auc_relative_diff"] < 0.06
    assert cohort["median_grad_norm_ratio"] < 1.34
