from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def row(
    role: str,
    quantization: str,
    prompt: int,
    prefill: float,
    decode: float,
    *,
    footprint: float,
    peak: float,
) -> dict:
    candidate = role == "candidate"
    return {
        "axis": "qwen35_cross_model_speed",
        "model_pair": "rwkv-7.2b__qwen3.5-9b",
        "model_role": role,
        "model_kind": "rwkv" if candidate else "qwen35",
        "prompt_tokens": prompt,
        "decode_tokens": 128,
        "batch_size": 1,
        "dtype": "fp16",
        "quantization": quantization,
        "status": "pass",
        "logits_finite": True,
        "prefill_tokps_total": prefill,
        "decode_tokps_total": decode,
        "prefill_sec_median": prompt / prefill,
        "decode_sec_median": 128 / decode,
        "model_footprint_mb": footprint,
        "peak_vram_mb": peak,
        "prefill_chunk_size": 0,
        "prefill_effective_backend": "native_prefill" if candidate else "module_call",
        "effective_backend": "native_graph" if candidate else "fla+causal_conv1d",
        "qwen_fast_path_verified": None if candidate else True,
    }


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


def test_route_composer_selects_only_gate_passing_measured_profiles(tmp_path: Path) -> None:
    rows = []
    for prompt in (128, 2048):
        rows.extend(
            [
                row("candidate", "none", prompt, 100.0, 100.0, footprint=100.0, peak=120.0),
                row("reference", "none", prompt, 80.0, 80.0, footprint=110.0, peak=130.0),
                row("reference", "bnb8", prompt, 100.0, 80.0, footprint=80.0, peak=100.0),
                row("reference", "bnb4", prompt, 90.0, 80.0, footprint=60.0, peak=90.0),
                row("candidate", "mm4", prompt, 110.0, 110.0, footprint=80.0, peak=100.0),
            ]
        )
        if prompt == 128:
            rows.extend(
                [
                    row("candidate", "bnb8", prompt, 80.0, 120.0, footprint=50.0, peak=80.0),
                    row("candidate", "a8w8", prompt, 110.0, 110.0, footprint=90.0, peak=100.0),
                ]
            )
        else:
            rows.extend(
                [
                    row("candidate", "bnb8", prompt, 120.0, 120.0, footprint=50.0, peak=80.0),
                    row("candidate", "a8w8", prompt, 80.0, 110.0, footprint=90.0, peak=100.0),
                ]
            )
    source = tmp_path / "rows.jsonl"
    output = tmp_path / "selected.jsonl"
    manifest = tmp_path / "manifest.json"
    write(source, rows)
    proc = subprocess.run(
        [
            sys.executable,
            "bench/compose_qwen35_quant_routes.py",
            "--results",
            str(source),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--fail-on-gate",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(manifest.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["output_rows"] == 12
    selected = {
        (item["cell"]["prompt_tokens"], item["cell"]["quantization"]): item["selected"]["implementation"]
        for item in report["decisions"]
    }
    assert selected[(128, "w8")] == "a8w8"
    assert selected[(2048, "w8")] == "bnb8"
    assert selected[(128, "w4")] == "mm4"
    assert selected[(2048, "w4")] == "mm4"
    selected_rows = [json.loads(line) for line in output.read_text().splitlines()]
    routed = [item for item in selected_rows if item.get("quantization_route")]
    assert len(routed) == 4
    assert all(item["quantization_route"] == "measured_profile_auto" for item in routed)


def test_route_composer_fails_closed_when_no_profile_passes(tmp_path: Path) -> None:
    rows = [
        row("candidate", "none", 128, 100.0, 100.0, footprint=100.0, peak=120.0),
        row("reference", "none", 128, 80.0, 80.0, footprint=110.0, peak=130.0),
        row("reference", "bnb8", 128, 100.0, 80.0, footprint=80.0, peak=100.0),
        row("reference", "bnb4", 128, 90.0, 80.0, footprint=60.0, peak=90.0),
        row("candidate", "a8w8", 128, 99.0, 110.0, footprint=90.0, peak=100.0),
        row("candidate", "mm4", 128, 110.0, 110.0, footprint=110.0, peak=130.0),
    ]
    source = tmp_path / "rows.jsonl"
    output = tmp_path / "selected.jsonl"
    manifest = tmp_path / "manifest.json"
    write(source, rows)
    proc = subprocess.run(
        [
            sys.executable,
            "bench/compose_qwen35_quant_routes.py",
            "--results",
            str(source),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--fail-on-gate",
        ],
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode == 1
    report = json.loads(manifest.read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert len(report["failures"]) == 2


def test_route_composer_can_accept_exact_cell_total_latency_noninferiority(
    tmp_path: Path,
) -> None:
    rows = [
        row("candidate", "none", 128, 100.0, 100.0, footprint=100.0, peak=120.0),
        row("reference", "none", 128, 80.0, 80.0, footprint=110.0, peak=130.0),
        row("candidate", "a8w8", 128, 105.0, 105.0, footprint=80.0, peak=100.0),
        # Prefill is 2% slower, but faster decode makes exact-cell total latency
        # lower than dense while both physical-memory gates remain strict.
        row("candidate", "mm4", 128, 98.0, 104.0, footprint=75.0, peak=95.0),
    ]
    source = tmp_path / "rows.jsonl"
    output = tmp_path / "selected.jsonl"
    manifest = tmp_path / "manifest.json"
    write(source, rows)
    proc = subprocess.run(
        [
            sys.executable,
            "bench/compose_qwen35_quant_routes.py",
            "--results",
            str(source),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--no-quant-qwen-gate",
            "--allow-dense-total-noninferiority",
            "--fail-on-gate",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(manifest.read_text(encoding="utf-8"))
    w4 = next(item for item in report["decisions"] if item["cell"]["quantization"] == "w4")
    assert w4["selected"]["checks"]["dense_prefill"] is False
    assert w4["selected"]["checks"]["dense_total"] is True
    assert w4["selected"]["checks"]["dense_speed"] is True


def test_route_composer_evaluates_duplicate_implementation_variants(tmp_path: Path) -> None:
    slow = row("candidate", "bnb8", 128, 99.0, 130.0, footprint=50.0, peak=70.0)
    slow["tuning_variant"] = "slow"
    fast = row("candidate", "bnb8", 128, 130.0, 140.0, footprint=55.0, peak=75.0)
    fast["tuning_variant"] = "fast"
    rows = [
        row("candidate", "none", 128, 100.0, 100.0, footprint=100.0, peak=120.0),
        row("reference", "none", 128, 80.0, 80.0, footprint=110.0, peak=130.0),
        row("reference", "bnb8", 128, 120.0, 80.0, footprint=80.0, peak=100.0),
        row("reference", "bnb4", 128, 90.0, 80.0, footprint=60.0, peak=90.0),
        slow,
        fast,
        row("candidate", "mm4", 128, 110.0, 110.0, footprint=80.0, peak=100.0),
    ]
    source = tmp_path / "rows.jsonl"
    output = tmp_path / "selected.jsonl"
    manifest = tmp_path / "manifest.json"
    write(source, rows)
    proc = subprocess.run(
        [
            sys.executable,
            "bench/compose_qwen35_quant_routes.py",
            "--results",
            str(source),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--fail-on-gate",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    selected_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    selected = next(item for item in selected_rows if item.get("quantization_route_source") == "bnb8")
    assert selected["tuning_variant"] == "fast"
    report = json.loads(manifest.read_text(encoding="utf-8"))
    w8 = next(item for item in report["decisions"] if item["cell"]["quantization"] == "w8")
    assert len(w8["alternatives"]) == 2


def test_route_composer_can_gate_quant_only_against_rwkv_dense(tmp_path: Path) -> None:
    rows = [
        row("candidate", "none", 128, 100.0, 100.0, footprint=100.0, peak=120.0),
        row("reference", "none", 128, 80.0, 80.0, footprint=110.0, peak=130.0),
        # No quantized Qwen rows are present. Both candidates beat RWKV dense
        # and lower physical footprint/peak, so the RWKV-only quant gate passes.
        row("candidate", "a8w8", 128, 101.0, 102.0, footprint=90.0, peak=110.0),
        row("candidate", "mm4", 128, 103.0, 101.0, footprint=80.0, peak=105.0),
    ]
    source = tmp_path / "rows.jsonl"
    output = tmp_path / "selected.jsonl"
    manifest = tmp_path / "manifest.json"
    write(source, rows)
    proc = subprocess.run(
        [
            sys.executable,
            "bench/compose_qwen35_quant_routes.py",
            "--results",
            str(source),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--no-quant-qwen-gate",
            "--fail-on-gate",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(manifest.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["thresholds"]["quant_qwen_gate"] is False
    selected_rows = [json.loads(line) for line in output.read_text().splitlines()]
    references = [item for item in selected_rows if item.get("quantization_reference_non_gating")]
    assert {item["quantization"] for item in references} == {"w8", "w4"}
