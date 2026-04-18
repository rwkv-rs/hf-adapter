from __future__ import annotations

import json
from pathlib import Path

from bench.summarize_4080_adjusted_pd import PAIRS, summarize


PARAMS = {
    "rwkv-0.4b__qwen3.5-0.8b": (450_767_872, 752_393_024),
    "rwkv-1.5b__qwen3.5-2b": (1_527_404_544, 1_881_825_088),
    "rwkv-2.9b__qwen3.5-4b": (2_947_735_040, 4_205_751_296),
}


def write_matrix(root: Path, adjusted_ratio: float = 1.1) -> tuple[Path, Path]:
    candidates = []
    references = []
    for pair in PAIRS:
        candidate_params, reference_params = PARAMS[pair]
        raw_ratio = adjusted_ratio * reference_params / candidate_params
        for batch in (1, 8):
            for prompt in (128, 512, 2048):
                for decode in (128, 512):
                    common = {
                        "model_pair": pair,
                        "batch_size": batch,
                        "prompt_tokens": prompt,
                        "decode_tokens": decode,
                        "device": "NVIDIA GeForce RTX 4080",
                        "dtype": "fp16",
                        "status": "pass",
                        "logits_finite": True,
                        "torch_version": "test",
                    }
                    references.append(
                        {
                            **common,
                            "model_role": "reference",
                            "active_parameter_count": reference_params,
                            "prefill_tokps_total": 100.0,
                            "decode_tokps_total": 10.0,
                            "qwen_fast_path_verified": True,
                            "effective_backend": "qwen_fla_gated_delta_rule",
                        }
                    )
                    candidates.append(
                        {
                            **common,
                            "model_role": "candidate",
                            "active_parameter_count": candidate_params,
                            "prefill_tokps_total": 100.0 * raw_ratio,
                            "decode_tokps_total": 10.0 * raw_ratio,
                        }
                    )
    candidate_path = root / "candidate.jsonl"
    reference_path = root / "reference.jsonl"
    candidate_path.write_text(
        "".join(json.dumps(row) + "\n" for row in candidates),
        encoding="utf-8",
    )
    reference_path.write_text(
        "".join(json.dumps(row) + "\n" for row in references),
        encoding="utf-8",
    )
    return candidate_path, reference_path


def test_adjusted_pd_summary_passes_all_36_cells(tmp_path: Path) -> None:
    candidate, reference = write_matrix(tmp_path, adjusted_ratio=1.1)
    report = summarize(candidate, reference)

    assert report["status"] == "pass"
    assert len(report["groups"]) == 6
    assert all(row["adjusted_pd_pass"] for row in report["groups"])
    assert all(row["adjusted_prefill_median"] == 1.1 for row in report["groups"])
    assert report["adjusted_prefill_cells_passed"] == 36
    assert report["adjusted_decode_cells_passed"] == 36


def test_adjusted_pd_summary_fails_below_one(tmp_path: Path) -> None:
    candidate, reference = write_matrix(tmp_path, adjusted_ratio=0.99)
    report = summarize(candidate, reference)

    assert report["status"] == "fail"
    assert len(report["errors"]) == 6


def test_adjusted_pd_summary_rejects_parameter_count_drift(tmp_path: Path) -> None:
    candidate, reference = write_matrix(tmp_path, adjusted_ratio=1.1)
    rows = [json.loads(line) for line in candidate.read_text().splitlines()]
    rows[0]["active_parameter_count"] += 1
    candidate.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = summarize(candidate, reference)

    assert report["status"] == "fail"
    assert any("parameter count drifted" in error for error in report["errors"])


def test_adjusted_pd_summary_rejects_one_cell_below_gate(tmp_path: Path) -> None:
    candidate, reference = write_matrix(tmp_path, adjusted_ratio=1.1)
    rows = [json.loads(line) for line in candidate.read_text().splitlines()]
    rows[0]["prefill_tokps_total"] *= 0.8
    candidate.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = summarize(candidate, reference)

    assert report["status"] == "fail"
    assert report["adjusted_prefill_cells_passed"] == 35
    assert any("cell minima" in error for error in report["errors"])
