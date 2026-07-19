from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "bench" / "5090_native_train_temp_b16_20260718"


def load_json(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_b16_single_step_tensor_contract_is_exact() -> None:
    backward = load_json("compare_backward_b16_t512.json")
    step = load_json("compare_step_b16_t512.json")

    assert backward["status"] == "pass"
    assert backward["gated_tensor_count"] == 399
    assert backward["worst_cosine"] == 1.0
    assert backward["max_relative_l2"] == backward["max_abs"] == 0.0
    assert step["status"] == "pass"
    assert step["gated_tensor_count"] == step["delta_tensor_count"] == 399
    assert step["optimizer_groups_match"] is True
    assert step["worst_cosine"] == 1.0
    assert step["max_relative_l2"] == step["max_abs"] == 0.0


def test_b16_three_seed_cohort_is_complete_and_fail_closed() -> None:
    cohort = load_json("compare_convergence_cohort_b16_t512_s1000.json")

    assert cohort["status"] == "pass"
    assert cohort["reference_seeds"] == cohort["candidate_seeds"] == [131, 232, 333]
    assert cohort["provenance_mismatches"] == cohort["failures"] == []
    assert cohort["runs_complete"] is True
    assert cohort["reference_deep_success_count"] == 3
    assert cohort["candidate_deep_success_count"] == 3
    assert all(
        row["finite"] and row["steps_completed"] == 1000
        for row in cohort["reference_rows"]
    )
    assert all(
        row["finite"] and row["steps_completed"] == 1000
        for row in cohort["candidate_rows"]
    )


def test_b16_resume_restores_all_state_and_matches_continuous_run() -> None:
    resumed = load_json("native_long_resume_final_seed131_s1000.json")
    comparison = load_json("compare_native_long_resume_seed131_s1000.json")

    assert resumed["status"] == "pass"
    assert resumed["start_step"] == 500
    assert resumed["steps_completed"] == 1000
    assert resumed["resumed_from"]["model_state_restored"] is True
    assert resumed["resumed_from"]["optimizer_state_restored"] is True
    assert resumed["resumed_from"]["rng_state_restored"] is True
    assert comparison["status"] == "pass"
    assert comparison["provenance_mismatches"] == comparison["failures"] == []
    assert comparison["max_validation_threshold_step_diff"] == 0


def test_b16_long_run_has_no_steady_cuda_memory_growth() -> None:
    stability = load_json("native_stability_memory_seed131_b16_t512_s1000.json")
    memory = stability["memory_stability"]

    assert stability["status"] == "pass"
    assert stability["steps_completed"] == 1000
    assert memory["sample_count"] == 20
    assert memory["first_step"] == 50
    assert memory["last_step"] == 1000
    assert memory["allocated_growth_mb"] <= 1.0
    assert memory["allocated_range_mb"] <= 1.0
    assert memory["reserved_growth_mb"] <= 0.0
    assert memory["reserved_range_mb"] <= 0.0


def test_b16_evidence_excludes_training_payloads() -> None:
    forbidden_suffixes = {".pt", ".pth", ".safetensors"}
    files = [path for path in EVIDENCE.iterdir() if path.is_file()]

    assert files
    assert not [path for path in files if path.suffix in forbidden_suffixes]
    assert max(path.stat().st_size for path in files) < 5 * 1024 * 1024
