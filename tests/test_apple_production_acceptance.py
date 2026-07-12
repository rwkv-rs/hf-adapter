from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bench.check_apple_production_acceptance import audit, evaluate_proof
from bench.render_apple_production_acceptance import render


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "bench" / "apple_production_gates.json"


def _write_jsonl(path: Path, *rows: dict[str, object]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_jsonl_query_checks_filters_requirements_and_coverage(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    _write_jsonl(
        evidence,
        {"axis": "bench", "status": "pass", "model": "a", "metrics": {"exact": True, "speed": 2.0}},
        {"axis": "bench", "status": "pass", "model": "b", "metrics": {"exact": True, "speed": 3.0}},
        {"axis": "other", "status": "pass", "model": "c", "metrics": {"exact": True, "speed": 4.0}},
    )
    proof = {
        "type": "jsonl_query",
        "path": "evidence.jsonl",
        "axis": "bench",
        "where": [{"field": "status", "op": "eq", "value": "pass"}],
        "require": [
            {"field": "metrics.exact", "op": "truthy"},
            {"field": "metrics.speed", "op": "ge", "value": 2.0},
        ],
        "min_matches": 2,
        "coverage": [{"field": "model", "values": ["a", "b"]}],
    }
    status, reason, details = evaluate_proof(proof, root=tmp_path)
    assert status == "pass"
    assert "2 evidence rows" in reason
    assert details["coverage"]["model"]["actual"] == ["a", "b"]


def test_missing_and_compound_proofs_remain_incomplete(tmp_path: Path) -> None:
    assert evaluate_proof(None, root=tmp_path)[0] == "missing"
    _write_jsonl(tmp_path / "ok.jsonl", {"axis": "x", "status": "pass"})
    compound = {
        "type": "all",
        "proofs": [
            {
                "type": "jsonl_query",
                "path": "ok.jsonl",
                "axis": "x",
                "require": [{"field": "status", "op": "eq", "value": "pass"}],
            },
            {"type": "files", "paths": ["missing.file"]},
        ],
    }
    status, reason, details = evaluate_proof(compound, root=tmp_path)
    assert status == "missing"
    assert reason == "1/2 child proofs pass"
    assert [item["status"] for item in details["proofs"]] == ["pass", "missing"]


def test_manifest_rejects_duplicate_or_incomplete_gate_ids(tmp_path: Path) -> None:
    duplicate = {
        "gates": [
            {"id": "same", "category": "x", "title": "a", "criterion": "a"},
            {"id": "same", "category": "x", "title": "b", "criterion": "b"},
        ]
    }
    with pytest.raises(ValueError, match="unique"):
        audit(duplicate, root=tmp_path)

    incomplete = {"gates": [{"id": "x", "category": "x", "title": "x"}]}
    with pytest.raises(ValueError, match="criterion"):
        audit(incomplete, root=tmp_path)


def test_default_manifest_is_exhaustive_and_rendered_without_drift() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    gates = manifest["gates"]
    assert len(gates) >= 130
    assert all(gate.get("required", True) for gate in gates)
    assert len({gate["id"] for gate in gates}) == len(gates)
    assert {gate["category"] for gate in gates} == {
        "release",
        "hf_mps_inference",
        "hf_mps_training",
        "mlx_correctness",
        "mlx_serving",
        "performance",
        "quantization",
        "coreml_ane",
        "reliability",
        "hardware",
    }

    report = render(manifest, root=ROOT)
    committed = (ROOT / "docs" / "hardware" / "APPLE_PRODUCTION_ACCEPTANCE.md").read_text(encoding="utf-8")
    assert report == committed
    for gate in gates:
        assert f"`{gate['id']}`" in report


def test_registered_evidence_never_silently_fails() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = audit(manifest, root=ROOT)
    assert len(rows) == len(manifest["gates"])
    assert not [row for row in rows if row["status"] in {"fail", "unknown"}]
    # This branch deliberately cannot claim production completion yet.
    assert any(row["status"] == "missing" for row in rows if row["required"])


def test_strict_cli_exits_nonzero_and_appends_summary(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    results = tmp_path / "audit.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "version": "test",
                "gates": [
                    {
                        "id": "missing",
                        "category": "test",
                        "title": "missing",
                        "criterion": "must have proof",
                        "required": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "check_apple_production_acceptance.py"),
            "--manifest",
            str(manifest),
            "--root",
            str(tmp_path),
            "--results",
            str(results),
            "--summary-only",
            "--fail-on-incomplete",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    output = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    assert output[-1]["axis"] == "apple_production_acceptance_summary"
    assert output[-1]["status"] == "incomplete"


def test_one_command_wrapper_can_run_nonstrict_audit(tmp_path: Path) -> None:
    results = tmp_path / "wrapper.jsonl"
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "STRICT": "0",
            "SUMMARY_ONLY": "1",
            "RESULTS": str(results),
        }
    )
    result = subprocess.run(
        [str(ROOT / "scripts" / "run_apple_production_acceptance.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["status"] == "incomplete"
    assert rows[-1]["required_gates"] >= 130
