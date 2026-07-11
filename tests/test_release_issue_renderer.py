from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "render_release_issue", ROOT / "scripts" / "render_release_issue.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
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
        "actual_routes": {
            "prefill": ["native-nvidia-prefill-v2[self_chunk]"],
            "decode": ["native-nvidia-fused-decode-v2[cuda_graph]"],
            "training": ["native-nvidia-train-temp-autograd-v2"],
            "quantization": ["native-w8-mm8-v1"],
        },
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
            "devices": {device: dict(device_row) for device in MODULE.DEVICES},
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
    assert "SFT" in body and "DPO" in body and "GRPO" in body
    normalized = body.lower().replace("lm-eval", "lm_eval")
    assert not [term for term in AUDIT.REQUIRED_ISSUE_TERMS if term not in normalized]
    assert len(body.encode()) < 65_000
