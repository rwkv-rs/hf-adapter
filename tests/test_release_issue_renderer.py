from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "render_release_issue", ROOT / "scripts" / "render_release_issue.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
from scripts.release_route_contract import (  # noqa: E402
    FORMAL_REFERENCE_BACKEND_ENVIRONMENT,
    READABLE_TRAINING_MODEL_ROUTE,
    REQUIRED_REFERENCE_TRAINING_ROUTES,
)

AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_github_release_for_issue_test",
    ROOT / "evaluation" / "audit_github_release.py",
)
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
sys.modules[AUDIT_SPEC.name] = AUDIT
AUDIT_SPEC.loader.exec_module(AUDIT)


def fixtures():
    version = "1.0.0"
    hf_sha = "1" * 64
    kernel_sha = "2" * 64
    device_row = {
        **{
            f"{gate}_status": "passed"
            for gate in (
                "correctness",
                "hf_ecosystem",
                "training",
                "quantization",
                "fla",
                "speed",
                "sft",
                "dpo",
                "grpo",
            )
        },
        "lm_eval_units": 144,
        "lm_eval_status": "passed",
        "training_policy": "reference",
        "training_backend_environment": dict(FORMAL_REFERENCE_BACKEND_ENVIRONMENT),
        "actual_routes": {
            "prefill": ["native-nvidia-prefill-v2[self_chunk]"],
            "decode": ["native-nvidia-fused-decode-v2[cuda_graph]"],
            "training": sorted(REQUIRED_REFERENCE_TRAINING_ROUTES),
            "quantization": ["native-w8-mm8-v1"],
        },
    }
    acceptance_times = {
        "rtx-4080": ("2026-08-28T00:00:00+00:00", "2026-08-28T01:00:00+00:00"),
        "rtx-4090": ("2026-08-28T01:00:00+00:00", "2026-08-28T02:00:00+00:00"),
    }
    provenance = {
        "version": version,
        "harness_sha": "b" * 40,
        "artifacts": {
            f"rwkv7_hf-{version}-py3-none-any.whl": {"sha256": hf_sha},
            f"rwkv7_kernels-{version}-py3-none-any.whl": {"sha256": kernel_sha},
        },
        "validation": {
            "status": "passed",
            "devices": {
                device: {
                    **device_row,
                    "acceptance_started_at": acceptance_times[device][0],
                    "acceptance_completed_at": acceptance_times[device][1],
                }
                for device in MODULE.DEVICES
            },
        },
    }
    lane = {
        "prefill": {"b1-t128": {"median_ms": 3.0}},
        "decode": {"b1": {"median_ms": 2.0}},
    }
    optimized_lane = {
        "prefill": {
            "b1-t128": {
                "median_ms": 1.0,
                "speedup_vs_reference": 3.0,
                "speedup_vs_fla": 2.0,
            }
        },
        "decode": {
            "b1": {
                "median_ms": 1.0,
                "speedup_vs_reference": 2.0,
                "speedup_vs_fla": 1.5,
            }
        },
    }
    fla_lane = {
        "prefill": {"b1-t128": {"median_ms": 2.0}},
        "decode": {"b1": {"median_ms": 1.5}},
    }
    speed = {
        "schema": "rwkv7-backend-v2-three-way-speed-v1",
        "status": "passed",
        "code_sha": provenance["harness_sha"],
        "fla": {"commit": MODULE.FLA_COMMIT},
        "wheels": {
            "rwkv7_hf": {"sha256": hf_sha},
            "rwkv7_kernels": {"sha256": kernel_sha},
        },
        "models": {
            "0.4b": {
                "lanes": {
                    "reference": lane,
                    "optimized": optimized_lane,
                    "fla": fla_lane,
                }
            }
        },
        "operator": {
            "lanes": {
                "reference": {"b1-t1": {"forward": {"median_ms": 3.0}}},
                "optimized": {
                    "b1-t1": {
                        "forward": {
                            "median_ms": 1.0,
                            "speedup_vs_reference": 3.0,
                            "speedup_vs_fla": 2.0,
                        }
                    }
                },
                "fla": {"b1-t1": {"forward": {"median_ms": 2.0}}},
            }
        },
        "training": {"status": "not_applicable", "mode": "reference-fallback"},
    }
    metrics = {
        lane_name: {f"0.1b-b1-task-{index}": {"acc,none": 0.5} for index in range(48)}
        for lane_name in ("reference", "optimized", "fla")
    }
    lm_eval = {
        "schema": "rwkv7-lm-eval-three-way-validation-v1",
        "status": "passed",
        "units": 144,
        "require_model_routes": True,
        "comparison_summary": dict(MODULE.ZERO_COMPARISON_SUMMARY),
        "aggregate_metrics": metrics,
    }
    return (
        provenance,
        {device: dict(speed) for device in MODULE.DEVICES},
        {device: dict(lm_eval) for device in MODULE.DEVICES},
    )


def test_release_issue_is_rendered_from_complete_speed_and_eval_matrices():
    provenance, speeds, lm_evals = fixtures()
    MODULE.validate_inputs(provenance=provenance, speeds=speeds, lm_evals=lm_evals)
    body = MODULE.render_issue(
        version="1.0.0",
        source_sha="a" * 40,
        provenance=provenance,
        speeds=speeds,
        lm_evals=lm_evals,
    )
    assert "Whole-model speed matrix" in body
    assert "Formal lm_eval accuracy/NLL/PPL matrix" in body
    assert "native-nvidia-prefill-v2[self_chunk]" in body
    assert READABLE_TRAINING_MODEL_ROUTE in body
    assert all(route in body for route in REQUIRED_REFERENCE_TRAINING_ROUTES)
    assert "not an admissible formal HF training route" in body
    assert "| same | same |" in body
    assert "SFT" in body and "DPO" in body and "GRPO" in body
    assert "Complete optional-kernel capability migration" in body
    assert "dense decode" in body and "DPLR/self-chunk" in body
    assert "SM70, Ada and Blackwell" in body
    assert "source scope" in body and "all 153 files" in body
    migration = MODULE.migration_transfer_summary()
    assert migration["total"] == 102
    assert f"{migration['byte_identical']} are byte-identical" in body
    assert (
        f"{migration['adapted_clean_boundary']} are declared clean-boundary adaptations"
        in body
    )
    assert "sequentially" in body and "non-overlapping" in body
    normalized = body.lower().replace("lm-eval", "lm_eval")
    assert not [term for term in AUDIT.REQUIRED_ISSUE_TERMS if term not in normalized]
    assert len(body.encode()) < 65_000


def test_release_issue_rejects_historical_whole_model_training_route():
    provenance, speeds, lm_evals = fixtures()
    for row in provenance["validation"]["devices"].values():
        row["actual_routes"]["training"] = ["native-nvidia-official-training-autograd-v2"]
    with pytest.raises(ValueError, match="not publishable.*historical whole-model"):
        MODULE.validate_inputs(
            provenance=provenance,
            speeds=speeds,
            lm_evals=lm_evals,
        )


def test_release_issue_rejects_non_reference_training_provenance():
    provenance, speeds, lm_evals = fixtures()
    provenance["validation"]["devices"]["rtx-4090"]["training_backend_environment"][
        "RWKV7_TRAINING_KERNEL_IMPL"
    ] = "adaptive"
    with pytest.raises(ValueError, match="reference training provenance is incomplete"):
        MODULE.validate_inputs(
            provenance=provenance,
            speeds=speeds,
            lm_evals=lm_evals,
        )


def test_public_release_tools_reject_migration_transfer_count_drift(
    tmp_path: Path,
):
    path = tmp_path / "MIGRATION_MANIFEST.json"
    path.write_text(
        json.dumps(
            {
                "files": [
                    *({"transfer": "byte_identical"} for _ in range(89)),
                    *({"transfer": "adapted_clean_boundary"} for _ in range(13)),
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical transfer counts differ"):
        MODULE.migration_transfer_summary(path)
    with pytest.raises(ValueError, match="canonical transfer counts differ"):
        AUDIT.migration_transfer_summary(path)
