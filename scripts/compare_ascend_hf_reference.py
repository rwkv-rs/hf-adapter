#!/usr/bin/env python3
"""Fail-closed comparison of an HF NPU capture with the pinned CPU oracle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from safetensors.torch import load_file

from rwkv7_hf.ascend_reference_oracle import evaluate_capture, sha256_file, tensor_map_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--reference-tensors", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--candidate-tensors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    args = parse_args()
    failures = []
    try:
        reference_meta = _load_json(args.reference_json)
        candidate_meta = _load_json(args.candidate_json)
        reference = load_file(args.reference_tensors, device="cpu")
        candidate = load_file(args.candidate_tensors, device="cpu")
    except Exception as exc:
        report = {"status": "fail", "failures": [f"capture_load: {type(exc).__name__}: {exc}"]}
    else:
        try:
            expected = reference_meta.get("capture", {})
            candidate_expected = candidate_meta.get("capture", {})
            reference_capture_sha256 = tensor_map_sha256(reference)
            candidate_capture_sha256 = tensor_map_sha256(candidate)
            if expected.get("tensor_file_sha256") != sha256_file(args.reference_tensors):
                failures.append("reference tensor-file SHA256 mismatch")
            if expected.get("capture_sha256") != reference_capture_sha256:
                failures.append("reference canonical capture SHA256 mismatch")
            if candidate_expected.get("tensor_file_sha256") != sha256_file(args.candidate_tensors):
                failures.append("candidate tensor-file SHA256 mismatch")
            if candidate_expected.get("capture_sha256") != candidate_capture_sha256:
                failures.append("candidate canonical capture SHA256 mismatch")
            if reference_meta.get("status") != "reference_generated":
                failures.append("reference metadata is not a generated oracle")
            if candidate_meta.get("status") != "candidate_captured":
                failures.append("candidate metadata is not a completed capture")
            if reference_meta.get("candidate_adapter_forward_called") is not False:
                failures.append("reference is not independent from the adapter candidate")
            for field in ("fla_source", "checkpoint_files_sha256", "tokenizer_files_sha256", "config_repair"):
                if candidate_meta.get(field) != reference_meta.get(field):
                    failures.append(f"candidate metadata mismatch: {field}")
            if candidate_meta.get("scenario") != reference_meta.get("scenario"):
                failures.append("candidate metadata mismatch: scenario")
            comparison = evaluate_capture(
                reference,
                candidate,
                thresholds=reference_meta.get("acceptance_thresholds", {}),
            )
            if comparison["status"] != "pass":
                failures.append("numeric_or_greedy_gate_failed")
            report = {
                "axis": "huawei_ascend_hf_vs_independent_cpu_oracle",
                "status": "pass" if not failures else "fail",
                "failures": failures,
                "reference_capture_sha256": expected.get("capture_sha256"),
                "candidate_capture_sha256": candidate_capture_sha256,
                "comparison": comparison,
            }
        except Exception as exc:
            report = {
                "axis": "huawei_ascend_hf_vs_independent_cpu_oracle",
                "status": "fail",
                "failures": failures + [f"validation_error: {type(exc).__name__}: {exc}"],
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
