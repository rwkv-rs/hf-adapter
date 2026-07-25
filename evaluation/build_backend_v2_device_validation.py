#!/usr/bin/env python3
"""Consolidate one GPU's passed backend-v2 reports into release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.fla_common import EXPECTED_FLA_COMMIT  # noqa: E402
from evaluation.validate_finetune_runs import (  # noqa: E402
    validate_reference_finetune_route_evidence,
)
from scripts.release_route_contract import (  # noqa: E402
    HISTORICAL_WHOLE_MODEL_TRAINING_ROUTE,
    validate_actual_routes,
    validate_formal_reference_environment,
)


DEVICES = {"rtx-4080", "rtx-4090"}
PRIMARY_REPORTS = (
    "correctness",
    "hf_ecosystem",
    "training",
    "quantization",
    "fla",
    "speed",
)
REPORT_SCHEMAS = {
    "correctness": "rwkv7-backend-v2-inference-validation-v1",
    "hf_ecosystem": "rwkv7-backend-v2-hf-ecosystem-v2",
    "training": "rwkv7-backend-v2-training-validation-v3",
    "quantization": "rwkv7-backend-v2-quantization-validation-v1",
    "fla": "rwkv7-backend-v2-three-way-validation-v3",
    "speed": "rwkv7-backend-v2-three-way-speed-v1",
}
REPORT_SCHEMA = "rwkv7-device-release-validation-v1"


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=sorted(DEVICES), required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--harness-sha", required=True)
    parser.add_argument("--hf-wheel", type=Path, required=True)
    parser.add_argument("--kernel-wheel", type=Path, required=True)
    parser.add_argument("--correctness-report", type=Path, required=True)
    parser.add_argument("--hf-ecosystem-report", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--quantization-report", type=Path, required=True)
    parser.add_argument("--fla-report", type=Path, required=True)
    parser.add_argument("--speed-report", type=Path, required=True)
    parser.add_argument("--finetune-report", type=Path, required=True)
    parser.add_argument("--lm-eval-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_json(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or unsafe validation report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"validation report is not an object: {path}")
    return payload


def wheel_hash(path: Path) -> str:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or unsafe wheel: {path}")
    if path.suffix != ".whl":
        raise ValueError(f"release artifact is not a wheel: {path}")
    return sha256_file(path)


def require_report_wheels(
    label: str,
    report: dict[str, Any],
    *,
    hf_wheel_sha256: str,
    kernel_wheel_sha256: str,
) -> None:
    wheels = report.get("wheels") or {}
    expected = {
        "rwkv7_hf": hf_wheel_sha256,
        "rwkv7_kernels": kernel_wheel_sha256,
    }
    actual = {name: (wheels.get(name) or {}).get("sha256") for name in expected}
    if actual != expected:
        raise ValueError(f"{label} report wheel identity mismatch")


def implementations(payload: Any) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        implementation = payload.get("implementation")
        if isinstance(implementation, str) and implementation.strip():
            rows.append((str(payload.get("phase", "")), implementation))
        for value in payload.values():
            rows.extend(implementations(value))
    elif isinstance(payload, list):
        for value in payload:
            rows.extend(implementations(value))
    return rows


def phase_routes(payload: Any, phase: str) -> list[str]:
    selected = {
        implementation
        for route_phase, implementation in implementations(payload)
        if route_phase == phase or phase in implementation.lower()
    }
    return sorted(selected)


def quantization_routes(report: dict[str, Any]) -> list[str]:
    routes = set()
    for row in report.get("methods") or []:
        if row.get("status") != "passed":
            continue
        method = str(row.get("method", "")).strip()
        if not method:
            raise ValueError("passed quantization row has no method identity")
        for key in ("prefill_route", "decode_route"):
            implementation = str((row.get(key) or {}).get("implementation", "")).strip()
            if implementation:
                routes.add(f"{method}:{implementation}")
    return sorted(routes)


def require_fla_commit(label: str, report: dict[str, Any]) -> None:
    fla = report.get("fla") or {}
    if fla.get("commit") != EXPECTED_FLA_COMMIT:
        raise ValueError(f"{label} report FLA commit mismatch")


def require_environment(label: str, report: dict[str, Any]) -> None:
    environment = report.get("environment") or {}
    command = environment.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(value, str) and value for value in command)
    ):
        raise ValueError(f"{label} report command provenance is missing")
    if not environment.get("gpu") or not environment.get("torch"):
        raise ValueError(f"{label} report runtime environment is incomplete")


def require_clean_leaf_training_toolkit(report: dict[str, Any]) -> None:
    environment = report.get("environment") or {}
    toolkit = environment.get("cuda_toolkit") or {}
    provenance = toolkit.get("provenance") or {}
    if not toolkit.get("nvcc") or not toolkit.get("nvcc_version"):
        raise ValueError("clean training leaf report lacks CUDA compiler provenance")
    cuda_home = Path(str(toolkit.get("cuda_home", "")))
    nvcc = Path(str(toolkit["nvcc"]))
    extensions = Path(str(toolkit.get("torch_extensions_dir", "")))
    if (
        not cuda_home.is_absolute()
        or nvcc != cuda_home / "bin" / "nvcc"
        or not extensions.is_absolute()
    ):
        raise ValueError("clean training leaf CUDA build paths are not bound")
    digest = str(provenance.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("clean training leaf report lacks CUDA toolkit identity")
    torch_match = re.search(r"(\d+)\.(\d+)", str(environment.get("cuda", "")))
    nvcc_match = re.search(
        r"release\s+(\d+)\.(\d+)",
        "\n".join(str(value) for value in toolkit["nvcc_version"]),
        re.IGNORECASE,
    )
    if not torch_match or not nvcc_match:
        raise ValueError("clean training leaf CUDA version provenance is incomplete")
    torch_cuda = tuple(int(value) for value in torch_match.groups())
    nvcc_cuda = tuple(int(value) for value in nvcc_match.groups())
    if torch_cuda != nvcc_cuda:
        raise ValueError("clean training leaf compiler does not match PyTorch CUDA")


def validate_finetune(
    report: dict[str, Any],
    *,
    hf_wheel_sha256: str,
    kernel_wheel_sha256: str,
) -> dict[str, str]:
    if report.get("status") != "passed":
        raise ValueError("finetune report did not pass")
    runs = report.get("runs") or {}
    statuses: dict[str, str] = {}
    for name in ("sft", "dpo", "grpo"):
        row = runs.get(name) or {}
        if row.get("status") != "passed":
            raise ValueError(f"{name} gate did not pass")
        artifacts = row.get("artifacts") or {}
        actual = {
            "rwkv7_hf": (artifacts.get("rwkv7_hf") or {}).get("sha256"),
            "rwkv7_kernels": (artifacts.get("rwkv7_kernels") or {}).get("sha256"),
        }
        expected = {
            "rwkv7_hf": hf_wheel_sha256,
            "rwkv7_kernels": kernel_wheel_sha256,
        }
        if actual != expected:
            raise ValueError(f"{name} report wheel identity mismatch")
        backend_routes = row.get("backend_routes") or []
        trace = row.get("kernel_route_trace") or {}
        route_evidence = validate_reference_finetune_route_evidence(
            backend_routes, trace
        )
        if not route_evidence["passed"]:
            raise ValueError(
                f"{name} report has invalid reference training routes: "
                f"{route_evidence['failures']}"
            )
        statuses[f"{name}_status"] = "passed"
    return statuses


def validate_lm_eval(
    report: dict[str, Any],
    *,
    hf_wheel_sha256: str,
    kernel_wheel_sha256: str,
) -> None:
    if report.get("status") != "passed" or report.get("units") != 144:
        raise ValueError("formal three-way lm_eval gate did not pass")
    if report.get("require_model_routes") is not True:
        raise ValueError("formal lm_eval did not require whole-model route evidence")
    aggregate_metrics = report.get("aggregate_metrics") or {}
    if set(aggregate_metrics) != {"reference", "optimized", "fla"} or any(
        len(aggregate_metrics[lane]) != 48 for lane in aggregate_metrics
    ):
        raise ValueError("formal lm_eval aggregate metric matrix is incomplete")
    summary = report.get("comparison_summary") or {}
    if summary != {
        "candidate_comparisons": 96,
        "metric_failures": 0,
        "prediction_mismatches": 0,
        "continuous_mismatches": 0,
        "missing_docs": 0,
    }:
        raise ValueError("formal lm_eval per-sample or aggregate comparison failed")
    artifacts = report.get("artifacts") or {}
    if set(artifacts) != {"reference", "optimized", "fla"}:
        raise ValueError("formal lm_eval artifact lanes are incomplete")
    for lane, payload in artifacts.items():
        wheels = payload.get("wheels") or {}
        actual = {
            "rwkv7_hf": (wheels.get("rwkv7_hf") or {}).get("sha256"),
            "rwkv7_kernels": (wheels.get("rwkv7_kernels") or {}).get("sha256"),
        }
        expected = {
            "rwkv7_hf": hf_wheel_sha256,
            "rwkv7_kernels": kernel_wheel_sha256,
        }
        if actual != expected:
            raise ValueError(f"formal lm_eval wheel identity mismatch: {lane}")
        if (payload.get("fla") or {}).get("commit") != EXPECTED_FLA_COMMIT:
            raise ValueError(f"formal lm_eval FLA commit mismatch: {lane}")


def write_atomic(path: Path, report: dict[str, Any]) -> None:
    payload = (
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
    temporary.chmod(0o644)
    temporary.replace(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.device not in DEVICES:
        raise ValueError(f"unexpected release device: {args.device}")
    for label, value in (("source", args.source_sha), ("harness", args.harness_sha)):
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError(f"{label} SHA must be a lowercase 40-character Git SHA")
    hf_wheel_sha256 = wheel_hash(args.hf_wheel)
    kernel_wheel_sha256 = wheel_hash(args.kernel_wheel)
    paths = {
        label: getattr(args, f"{label}_report").expanduser().resolve()
        for label in PRIMARY_REPORTS
    }
    reports = {label: safe_json(path) for label, path in paths.items()}
    for label, report in reports.items():
        if report.get("schema") != REPORT_SCHEMAS[label]:
            raise ValueError(f"{label} report schema mismatch")
        if report.get("status") != "passed":
            raise ValueError(f"{label} report did not pass")
        if report.get("code_sha") != args.harness_sha:
            raise ValueError(f"{label} report harness SHA mismatch")
        require_report_wheels(
            label,
            report,
            hf_wheel_sha256=hf_wheel_sha256,
            kernel_wheel_sha256=kernel_wheel_sha256,
        )
        require_environment(label, report)
    require_fla_commit("fla", reports["fla"])
    require_fla_commit("speed", reports["speed"])
    fla_report = reports["fla"]
    if (fla_report.get("release_gates") or {}).get("role") != "blocking":
        raise ValueError("FLA report is missing explicit blocking release gates")
    fla_diagnostics = fla_report.get("fla_diagnostics") or {}
    if (
        fla_diagnostics.get("role") != "diagnostic-non-blocking"
        or fla_diagnostics.get("complete") is not True
    ):
        raise ValueError("FLA report is missing complete non-blocking diagnostics")

    expected_training_mode = "reference"
    if (reports["training"].get("settings") or {}).get(
        "candidate_route"
    ) != expected_training_mode:
        raise ValueError("training capability report does not match the release device")
    if (reports["hf_ecosystem"].get("training_expectation") or {}).get(
        "mode"
    ) != expected_training_mode:
        raise ValueError("HF ecosystem training mode does not match the release device")
    if (reports["speed"].get("training") or {}).get("mode") != expected_training_mode:
        raise ValueError("speed training mode does not match the release device")
    training_backend_environment = validate_formal_reference_environment(
        reports["training"].get("environment")
    )
    hf_backend_environment = validate_formal_reference_environment(
        reports["hf_ecosystem"].get("environment")
    )
    if hf_backend_environment != training_backend_environment:
        raise ValueError("formal training and HF ecosystem environments differ")

    reported_training_implementations = {
        implementation for _, implementation in implementations(reports["training"])
    }
    historical_training = sorted(
        route
        for route in reported_training_implementations
        if route == HISTORICAL_WHOLE_MODEL_TRAINING_ROUTE
        or route.startswith(f"{HISTORICAL_WHOLE_MODEL_TRAINING_ROUTE}[")
    )
    if historical_training:
        raise ValueError(
            "training report contains the historical whole-model train-temp route: "
            f"{historical_training}"
        )

    actual_routes = {
        "prefill": phase_routes(reports["correctness"], "prefill"),
        "decode": phase_routes(reports["correctness"], "decode"),
        "training": phase_routes(reports["training"], "training"),
        "quantization": quantization_routes(reports["quantization"]),
    }
    try:
        actual_routes = validate_actual_routes(actual_routes)
    except ValueError as exc:
        raise ValueError(f"formal device route evidence is invalid: {exc}") from exc

    finetune_path = args.finetune_report.expanduser().resolve()
    finetune = safe_json(finetune_path)
    finetune_statuses = validate_finetune(
        finetune,
        hf_wheel_sha256=hf_wheel_sha256,
        kernel_wheel_sha256=kernel_wheel_sha256,
    )
    lm_eval_path = args.lm_eval_report.expanduser().resolve()
    lm_eval = safe_json(lm_eval_path)
    validate_lm_eval(
        lm_eval,
        hf_wheel_sha256=hf_wheel_sha256,
        kernel_wheel_sha256=kernel_wheel_sha256,
    )
    input_paths = {**paths, "finetune": finetune_path, "lm_eval": lm_eval_path}
    report = {
        "schema": REPORT_SCHEMA,
        "device": args.device,
        "status": "passed",
        "source_sha": args.source_sha,
        "harness_sha": args.harness_sha,
        "fla_commit": EXPECTED_FLA_COMMIT,
        "hf_wheel_sha256": hf_wheel_sha256,
        "kernel_wheel_sha256": kernel_wheel_sha256,
        "lm_eval_units": 144,
        "lm_eval_status": "passed",
        "lm_eval_comparison_summary": lm_eval["comparison_summary"],
        "training_policy": "reference",
        "training_backend_environment": training_backend_environment,
        **{f"{label}_status": "passed" for label in PRIMARY_REPORTS},
        **finetune_statuses,
        "actual_routes": actual_routes,
        "evidence_inputs": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in sorted(input_paths.items())
        },
    }
    write_atomic(args.output.expanduser().resolve(), report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    report = build(args)
    print(
        json.dumps(
            {
                "device": report["device"],
                "output": str(args.output),
                "status": "passed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
