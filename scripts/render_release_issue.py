#!/usr/bin/env python3
"""Render the public v1 validation Issue directly from passed JSON evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_release_assets import (  # noqa: E402
    DEVICE_ORDER,
    DEVICES,
    FLA_COMMIT,
    verify as verify_release_assets,
)
from scripts.release_route_contract import (  # noqa: E402
    FORMAL_REFERENCE_BACKEND_ENVIRONMENT,
    validate_actual_routes,
)


ZERO_COMPARISON_SUMMARY = {
    "candidate_comparisons": 96,
    "metric_failures": 0,
    "prediction_mismatches": 0,
    "continuous_mismatches": 0,
    "missing_docs": 0,
}

MIGRATION_MANIFEST = (
    ROOT / "kernels" / "rwkv7_kernels" / "nvidia" / "MIGRATION_MANIFEST.json"
)
EXPECTED_MIGRATION_TRANSFER_SUMMARY = {
    "total": 102,
    "byte_identical": 86,
    "adapted_clean_boundary": 16,
}


def migration_transfer_summary(path: Path = MIGRATION_MANIFEST) -> dict[str, int]:
    """Read the one authoritative migration denominator used in Issue prose."""

    payload = safe_json(path)
    files = payload.get("files") or []
    if not isinstance(files, list):
        raise ValueError("NVIDIA migration manifest files must be a list")
    result = {"total": len(files), "byte_identical": 0, "adapted_clean_boundary": 0}
    for row in files:
        transfer = str((row or {}).get("transfer", ""))
        if transfer not in {"byte_identical", "adapted_clean_boundary"}:
            raise ValueError(f"unexpected NVIDIA migration transfer class: {transfer}")
        result[transfer] += 1
    if result != EXPECTED_MIGRATION_TRANSFER_SUMMARY:
        raise ValueError(
            "NVIDIA migration manifest canonical transfer counts differ: "
            f"expected={EXPECTED_MIGRATION_TRANSFER_SUMMARY} actual={result}"
        )
    return result


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--speed", action="append", required=True, help="device=JSON")
    parser.add_argument(
        "--lm-eval", action="append", required=True, help="device=validation JSON"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
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
        raise ValueError(f"missing or unsafe Issue evidence: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Issue evidence is not an object: {path}")
    return payload


def parse_device_paths(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use device=path: {value}")
        device, raw_path = value.split("=", 1)
        if device in result or device not in DEVICES or not raw_path:
            raise ValueError(f"invalid or duplicate {label}: {value}")
        result[device] = Path(raw_path)
    if set(result) != set(DEVICES):
        raise ValueError(f"{label} does not cover all required devices")
    return result


def wheel_hashes(report: dict[str, Any]) -> dict[str, str | None]:
    wheels = report.get("wheels") or {}
    return {
        "rwkv7_hf": (wheels.get("rwkv7_hf") or {}).get("sha256"),
        "rwkv7_kernels": (wheels.get("rwkv7_kernels") or {}).get("sha256"),
    }


def validate_inputs(
    *,
    provenance: dict[str, Any],
    speeds: dict[str, dict[str, Any]],
    lm_evals: dict[str, dict[str, Any]],
) -> None:
    validation = provenance.get("validation") or {}
    if validation.get("status") != "passed" or set(
        validation.get("devices") or {}
    ) != set(DEVICES):
        raise ValueError("required-device release provenance did not pass")
    expected_wheels = {
        "rwkv7_hf": provenance["artifacts"][
            f"rwkv7_hf-{provenance['version']}-py3-none-any.whl"
        ]["sha256"],
        "rwkv7_kernels": provenance["artifacts"][
            f"rwkv7_kernels-{provenance['version']}-py3-none-any.whl"
        ]["sha256"],
    }
    for device in DEVICE_ORDER:
        device_validation = validation["devices"][device]
        if device_validation.get("training_policy") != "reference" or (
            device_validation.get("training_backend_environment")
            != FORMAL_REFERENCE_BACKEND_ENVIRONMENT
        ):
            raise ValueError(
                f"formal reference training provenance is incomplete: {device}"
            )
        try:
            validate_actual_routes(device_validation.get("actual_routes"))
        except ValueError as exc:
            raise ValueError(
                f"release route evidence is not publishable: {device}: {exc}"
            ) from exc
        speed = speeds[device]
        if (
            speed.get("schema") != "rwkv7-backend-v2-three-way-speed-v1"
            or speed.get("status") != "passed"
            or speed.get("code_sha") != provenance.get("harness_sha")
            or (speed.get("fla") or {}).get("commit") != FLA_COMMIT
            or wheel_hashes(speed) != expected_wheels
        ):
            raise ValueError(f"speed evidence is not release-bound: {device}")
        lm_eval = lm_evals[device]
        if (
            lm_eval.get("schema") != "rwkv7-lm-eval-three-way-validation-v1"
            or lm_eval.get("status") != "passed"
            or lm_eval.get("units") != 144
            or lm_eval.get("require_model_routes") is not True
            or lm_eval.get("comparison_summary") != ZERO_COMPARISON_SUMMARY
            or set(lm_eval.get("aggregate_metrics") or {})
            != {"reference", "optimized", "fla"}
            or any(
                len((lm_eval.get("aggregate_metrics") or {})[lane]) != 48
                for lane in ("reference", "optimized", "fla")
            )
        ):
            raise ValueError(f"formal lm_eval evidence is incomplete: {device}")


def number(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def metric_text(metrics: dict[str, Any]) -> str:
    return ", ".join(
        f"{name}={float(value):.8g}" for name, value in sorted(metrics.items())
    )


def candidate_metric_text(reference: dict[str, Any], candidate: dict[str, Any]) -> str:
    if candidate == reference:
        return "same"
    return metric_text(candidate)


def render_issue(
    *,
    version: str,
    source_sha: str,
    provenance: dict[str, Any],
    speeds: dict[str, dict[str, Any]],
    lm_evals: dict[str, dict[str, Any]],
) -> str:
    migration = migration_transfer_summary()
    devices = (provenance.get("validation") or {}).get("devices") or {}
    hf_name = f"rwkv7_hf-{version}-py3-none-any.whl"
    kernel_name = f"rwkv7_kernels-{version}-py3-none-any.whl"
    lines = [
        f"# RWKV7 HF v{version}: reference + optional kernels validation",
        "",
        "This Issue records the release evidence generated by the checked-in harness. "
        "The readable `rwkv7_hf` model owns modeling/config/cache/ops; the optional "
        "`rwkv7-kernels` wheel owns NVIDIA policy and implementations.",
        "",
        "## Immutable artifacts",
        "",
        f"- source SHA256/commit: `{source_sha}`",
        f"- harness commit: `{provenance['harness_sha']}`",
        f"- FLA commit: `{FLA_COMMIT}`",
        f"- `{hf_name}` SHA256: `{provenance['artifacts'][hf_name]['sha256']}`",
        f"- `{kernel_name}` SHA256: `{provenance['artifacts'][kernel_name]['sha256']}`",
        "",
        "The same wheel pair was used sequentially, with non-overlapping "
        "acceptance runs in the fixed RTX 4080 -> RTX 4090 order.",
        "",
        "| device | acceptance started (UTC) | acceptance completed (UTC) |",
        "|---|---|---|",
        *[
            f"| {device} | {devices[device]['acceptance_started_at']} | "
            f"{devices[device]['acceptance_completed_at']} |"
            for device in DEVICE_ORDER
        ],
        "",
        "## Gate matrix",
        "",
        "| device | correctness | HF ecosystem | training | quantization | FLA | speed | SFT | DPO | GRPO | lm_eval |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for device in DEVICE_ORDER:
        row = devices[device]
        lines.append(
            "| "
            + " | ".join(
                [
                    device,
                    *[
                        str(row[f"{gate}_status"])
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
                    ],
                    f"{row['lm_eval_units']}/144 {row['lm_eval_status']}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "HF coverage includes AutoModel/save-reload, state/cache, left/right padding, "
            "greedy and beam generation, Trainer, Accelerate, PEFT and TRL. Training "
            "keeps one complete readable reference program; optional recurrent, "
            "flattened-linear and Mix6 leaves remain diagnostics. Evidence "
            "includes loss/all gradients plus SFT, DPO and GRPO. Quantization "
            "covers W8, W4, A8W8, Bn/Tn, BitsAndBytes, Marlin and TorchAO.",
            "Formal training uses `RWKV7_BACKEND=auto`, `RWKV7_KERNEL_IMPL=auto`, "
            "`RWKV7_MODEL_KERNEL_IMPL=auto` and "
            "`RWKV7_TRAINING_KERNEL_IMPL=auto`; compact provenance validates that "
            "environment and the full reference model/program/recurrent/linear/Mix6 route.",
            "",
            "## Complete optional-kernel capability migration",
            "",
            "The optional wheel contains the complete audited NVIDIA families: recurrent; "
            "dense decode; fused DPLR/self-chunk prefill; CUDA Graph and state pools; "
            "exact-card SM70, Ada and Blackwell policy; native W8, W4 and A8W8; "
            "Bn/Tn, BitsAndBytes, Marlin and TorchAO adapters; and independent training "
            "autograd leaves. "
            "Every migrated payload is mapped to an adapted runtime route by the embedded "
            "capability inventory; model/config/cache ownership remains in rwkv7_hf.",
            "The embedded migration manifest and source scope also classify all 153 "
            "files from the frozen "
            "historical performance tree and reconstructs its Git tree identity, so the "
            f"{migration['total']}-file NVIDIA migration denominator has no silent "
            f"omissions: {migration['byte_identical']} are byte-identical and "
            f"{migration['adapted_clean_boundary']} are declared clean-boundary "
            "adaptations for the "
            "canonical cache and non-monkeypatch training protocol. The preserved "
            "whole-model train-temp runtime is historical diagnostic material and is "
            "not an admissible formal HF training route.",
            "The later v0.10 recurrent wheel is independently covered: its complete "
            "three-file package subtree is reconstructed and its Graph/Triton "
            "implementations remain byte-identical behind API v3.",
            "",
            "## Actual routes",
            "",
            "| device | boundary/phase | implementation route |",
            "|---|---|---|",
        ]
    )
    for device in DEVICE_ORDER:
        for phase, routes in sorted(devices[device]["actual_routes"].items()):
            values = [routes] if isinstance(routes, str) else routes
            lines.append(f"| {device} | {phase} | `{', '.join(values)}` |")

    lines.extend(
        [
            "",
            "## Whole-model speed matrix vs reference and FLA",
            "",
            "Cold compile/capture is excluded from steady-state medians.",
            "",
            "| device | model | phase | shape | reference ms | optimized ms | FLA ms | optimized/reference | optimized/FLA |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for device in DEVICE_ORDER:
        for model, payload in sorted(speeds[device]["models"].items()):
            for phase in ("prefill", "decode"):
                lanes = payload["lanes"]
                for shape, optimized in sorted(lanes["optimized"][phase].items()):
                    reference = lanes["reference"][phase][shape]
                    fla = lanes["fla"][phase][shape]
                    lines.append(
                        f"| {device} | {model} | {phase} | {shape} | "
                        f"{number(reference['median_ms'])} | {number(optimized['median_ms'])} | "
                        f"{number(fla['median_ms'])} | {number(optimized['speedup_vs_reference'])}x | "
                        f"{number(optimized['speedup_vs_fla'])}x |"
                    )

    lines.extend(
        [
            "",
            "## Operator forward and training speed",
            "",
            "| device | scope | shape | reference ms | optimized ms | FLA ms | optimized/reference | optimized/FLA |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for device in DEVICE_ORDER:
        operator = speeds[device].get("operator", {}).get("lanes", {})
        for shape, optimized_row in sorted(operator.get("optimized", {}).items()):
            optimized = optimized_row["forward"]
            reference = operator["reference"][shape]["forward"]
            fla = operator["fla"][shape]["forward"]
            lines.append(
                f"| {device} | recurrent forward | {shape} | {number(reference['median_ms'])} | "
                f"{number(optimized['median_ms'])} | {number(fla['median_ms'])} | "
                f"{number(optimized['speedup_vs_reference'])}x | {number(optimized['speedup_vs_fla'])}x |"
            )
        training = speeds[device].get("training") or {}
        if training.get("status") == "not_applicable":
            lines.append(
                f"| {device} | training ({training['mode']}) | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
        else:
            lanes = training.get("lanes") or {}
            for shape, optimized in sorted(lanes.get("optimized", {}).items()):
                reference = lanes["reference"][shape]
                fla = lanes["fla"][shape]
                lines.append(
                    f"| {device} | training ({training['mode']}) | {shape} | "
                    f"{number(reference['median_ms'])} | {number(optimized['median_ms'])} | "
                    f"{number(fla['median_ms'])} | {number(optimized['speedup_vs_reference'])}x | "
                    f"{number(optimized['speedup_vs_fla'])}x |"
                )

    lines.extend(
        [
            "",
            "## Formal lm_eval accuracy/NLL/PPL matrix",
            "",
            "Each device ran reference/optimized/FLA for 0.1B, 0.4B and 1.5B, "
            "eight tasks and batch 1/8: 144 formal units. Per-sample selected answers, "
            "aggregate accuracy, Wikitext NLL/PPL, and batch invariance passed.",
            "",
            "| device | unit | reference | optimized | FLA |",
            "|---|---|---|---|---|",
        ]
    )
    for device in DEVICE_ORDER:
        metrics = lm_evals[device]["aggregate_metrics"]
        for unit in sorted(metrics["reference"]):
            reference_metrics = metrics["reference"][unit]
            lines.append(
                f"| {device} | {unit} | {metric_text(reference_metrics)} | "
                f"{candidate_metric_text(reference_metrics, metrics['optimized'][unit])} | "
                f"{candidate_metric_text(reference_metrics, metrics['fla'][unit])} |"
            )
        summary = lm_evals[device]["comparison_summary"]
        lines.append(
            f"| {device} | comparison summary | 96 comparisons | "
            f"metric failures={summary['metric_failures']}, selected-answer mismatches="
            f"{summary['prediction_mismatches']} | continuous NLL mismatches="
            f"{summary['continuous_mismatches']}, missing docs={summary['missing_docs']} |"
        )
    lines.extend(
        [
            "",
            "Raw samples, logs, commands, environments, model/wheel SHA256 values and "
            "actual route traces remain in the external evidence bundles; compact "
            "manifest-covered summaries are retained in the repository.",
        ]
    )
    body = "\n".join(lines) + "\n"
    if len(body.encode("utf-8")) > 65_000:
        raise ValueError("rendered GitHub Issue exceeds the 65,000-byte safety limit")
    return body


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    release = verify_release_assets(
        SimpleNamespace(
            directory=args.directory,
            version=args.version,
            source_sha=args.source_sha,
            require_validation_passed=True,
        )
    )
    provenance_path = args.directory.expanduser().resolve() / "release-provenance.json"
    provenance = safe_json(provenance_path)
    speed_paths = parse_device_paths(args.speed, "--speed")
    lm_paths = parse_device_paths(args.lm_eval, "--lm-eval")
    speeds = {device: safe_json(path) for device, path in speed_paths.items()}
    lm_evals = {device: safe_json(path) for device, path in lm_paths.items()}
    validate_inputs(provenance=provenance, speeds=speeds, lm_evals=lm_evals)
    body = render_issue(
        version=args.version,
        source_sha=args.source_sha,
        provenance=provenance,
        speeds=speeds,
        lm_evals=lm_evals,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body, encoding="utf-8")
    report = {
        "schema": "rwkv7-release-issue-render-v1",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": args.version,
        "source_sha": args.source_sha,
        "harness_sha": release["harness_sha"],
        "output": str(args.output.expanduser().resolve()),
        "output_bytes": len(body.encode("utf-8")),
        "output_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "inputs": {
            "release_provenance": sha256_file(provenance_path),
            "speed": {
                device: sha256_file(path.resolve())
                for device, path in speed_paths.items()
            },
            "lm_eval": {
                device: sha256_file(path.resolve()) for device, path in lm_paths.items()
            },
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "report": str(args.report), "status": "passed"}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
