#!/usr/bin/env python3
"""Validate strict RTX 3090 RWKV/Qwen raw and parameter-adjusted P/D parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from bench import validate_qwen35_4090_paired_pd_v2 as base


CONTRACT_PROTOCOL = "qwen35_3090_reference_contract_v1"
PROTOCOL = "qwen35_3090_paired_pd_v2"
CORRECTNESS_PROTOCOL = "rwkv_native_graph_fla_correctness_3090_v2"
EXPECTED_DEVICE = "NVIDIA GeForce RTX 3090"
EXPECTED_RUNTIME = {
    "python": "3.10.12",
    "torch": "2.7.1+cu126",
    "torch_cuda": "12.6",
    "triton": "3.3.1",
    "transformers": "5.12.1",
    "fla": "0.5.1",
    "causal_conv1d": "1.6.2.post1",
}


def _load_contract(path: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, [f"reference contract cannot be read: {exc}"]
    if not isinstance(doc, dict):
        return {}, ["reference contract must be an object"]
    expected = {
        "schema_version": 1,
        "protocol": CONTRACT_PROTOCOL,
        "device": EXPECTED_DEVICE,
        "gpu_arch": "sm_86",
        "compute_cap": "8.6",
        "driver_version": "550.142",
        "memory_total_mib": 24576,
        "runtime": EXPECTED_RUNTIME,
        "torch_cuda_arch_list": "8.6",
    }
    for field, value in expected.items():
        if not base._strict(doc.get(field), value):
            errors.append(f"reference contract {field} mismatch")
    sha = doc.get("reference_sha256")
    if (
        type(sha) is not str
        or len(sha) != 64
        or any(char not in "0123456789abcdef" for char in sha.lower())
    ):
        errors.append("reference contract has invalid reference_sha256")
    routes = doc.get("routes_by_pair")
    if type(routes) is not dict or set(routes) != set(base.PAIRS):
        errors.append("reference contract routes_by_pair coverage mismatch")
    elif any(
        route
        not in {
            "static_cache_inductor_cudagraph",
            "static_cache_raw_cudagraph",
        }
        for route in routes.values()
    ):
        errors.append("reference contract contains an unsupported Qwen route")
    return doc, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-contract", type=Path, required=True)
    parser.add_argument("--correctness-manifest", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--model-hashes", type=Path, required=True)
    parser.add_argument("--model-hashes-after", type=Path, required=True)
    parser.add_argument("--system", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--paired-table", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base.REPORT_TITLE = "RTX 3090 strict RWKV/Qwen Prefill+Decode v2"
    contract, contract_errors = _load_contract(args.reference_contract)
    if not contract_errors:
        base.PROTOCOL = PROTOCOL
        base.EXPECTED_DEVICE = EXPECTED_DEVICE
        base.REFERENCE_SHA256 = contract["reference_sha256"]
        base.CORRECTNESS_PROTOCOL = CORRECTNESS_PROTOCOL
        base.QWEN_CONTRACT = (
            "official_fla_causal_conv1d_static_cache_cudagraph_same_cache_3090_v2"
        )
        base.EXPECTED_RUNTIME = EXPECTED_RUNTIME
        base.EXPECTED_ARCH = "8.6"
        base.EXPECTED_DRIVER = "550.142"
        base.EXPECTED_MEMORY = "24576 MiB"
        base.SPECIAL_SMALL_B8_BUNDLE = False
        base.BASE_ADA_WAGV_BMM_EXPECTED = False
        base.QWEN_ROUTES = dict(contract["routes_by_pair"])
        summary = base.validate(args)
    else:
        summary = {
            "schema_version": 2,
            "protocol": PROTOCOL,
            "status": "fail",
            "paired_pd_table_eligible": False,
            "cells": [],
            "metrics": {},
            "correctness": {"status": "fail"},
            "errors": contract_errors,
        }
    summary["reference_contract_sha256"] = (
        hashlib.sha256(args.reference_contract.read_bytes()).hexdigest()
        if args.reference_contract.is_file()
        else None
    )
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(base.render(summary), encoding="utf-8")
    if summary["paired_pd_table_eligible"]:
        args.paired_table.write_text(
            "".join(
                json.dumps(cell, separators=(",", ":")) + "\n"
                for cell in summary["cells"]
            ),
            encoding="utf-8",
        )
    elif args.paired_table.exists():
        args.paired_table.unlink()
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
