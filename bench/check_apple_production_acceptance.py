#!/usr/bin/env python3
"""Strict, evidence-driven Apple production acceptance auditor.

The auditor never infers completion from a feature existing in source code.
Every required gate needs a machine-verifiable proof in the manifest. Missing,
failed, or unknown proof keeps the aggregate result incomplete.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "bench" / "apple_production_gates.json"
GATE_STATUSES = ("pass", "fail", "missing", "unknown")


def _field(row: dict[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _condition(value: Any, condition: dict[str, Any]) -> bool:
    op = condition.get("op", "eq")
    expected = condition.get("value")
    try:
        if op == "eq":
            return value == expected
        if op == "ne":
            return value != expected
        if op == "ge":
            return value is not None and float(value) >= float(expected)
        if op == "gt":
            return value is not None and float(value) > float(expected)
        if op == "le":
            return value is not None and float(value) <= float(expected)
        if op == "lt":
            return value is not None and float(value) < float(expected)
        if op == "in":
            values = condition.get("values", [])
            return isinstance(values, list) and value in values
        if op == "not_none":
            return value is not None
        if op == "truthy":
            return bool(value)
        if op == "falsy":
            return not bool(value)
        if op == "contains":
            return value is not None and expected in value
        if op == "not_empty":
            return value is not None and len(value) > 0
        if op == "empty":
            return value is not None and len(value) == 0
    except (TypeError, ValueError, OverflowError):
        return False
    raise ValueError(f"unsupported proof condition op {op!r}")


def _conditions(proof: dict[str, Any], name: str) -> list[dict[str, Any]]:
    conditions = proof.get(name, [])
    if not isinstance(conditions, list):
        raise ValueError(f"{name} must be a list")
    for condition in conditions:
        if not isinstance(condition, dict) or not isinstance(condition.get("field"), str):
            raise ValueError(f"every {name} condition must have a string field")
    return conditions


def _row_matches(row: dict[str, Any], conditions: list[dict[str, Any]]) -> bool:
    return all(_condition(_field(row, item["field"]), item) for item in conditions)


def _evidence_path(root: Path, declared: Any) -> Path:
    if not isinstance(declared, str) or not declared:
        raise ValueError("evidence path must be a non-empty string")
    relative = Path(declared)
    if relative.is_absolute():
        raise ValueError(f"evidence path must be repository-relative: {declared}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"evidence path escapes repository root: {declared}")
    return resolved


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row at {path}:{line_number} must be an object")
        rows.append(value)
    return rows


def _evaluate_proof(proof: dict[str, Any] | None, *, root: Path) -> tuple[str, str, dict[str, Any]]:
    if not proof:
        return "missing", "no machine-verifiable proof registered", {}
    if not isinstance(proof, dict):
        raise ValueError("proof must be an object")

    kind = proof.get("type")
    if kind == "all":
        children = proof.get("proofs", [])
        if not isinstance(children, list) or not children:
            return "unknown", "compound proof has no child proofs", {}
        evaluated = [evaluate_proof(child, root=root) for child in children]
        child_details = [
            {"status": status, "reason": reason, "details": details}
            for status, reason, details in evaluated
        ]
        statuses = [item[0] for item in evaluated]
        if "fail" in statuses:
            status = "fail"
        elif "unknown" in statuses:
            status = "unknown"
        elif "missing" in statuses:
            status = "missing"
        else:
            status = "pass"
        return status, f"{sum(value == 'pass' for value in statuses)}/{len(statuses)} child proofs pass", {
            "proofs": child_details
        }

    if kind in {"files", "tracked_files"}:
        declared_paths = proof.get("paths", [])
        if not isinstance(declared_paths, list) or not declared_paths:
            return "unknown", "file proof has no paths", {}
        paths = [_evidence_path(root, item) for item in declared_paths]
        missing = [str(path.relative_to(root.resolve())) for path in paths if not path.exists()]
        if missing:
            return "missing", f"missing evidence files: {', '.join(missing)}", {"missing": missing}
        non_files = [str(path.relative_to(root.resolve())) for path in paths if not path.is_file()]
        if non_files:
            return "unknown", f"evidence paths are not files: {', '.join(non_files)}", {"non_files": non_files}
        if kind == "tracked_files":
            command = ["git", "-C", str(root.resolve()), "ls-files", "--error-unmatch", "--", *declared_paths]
            try:
                tracked = subprocess.run(command, text=True, capture_output=True, check=False)
            except OSError as exc:
                return "unknown", f"cannot verify tracked evidence files: {exc}", {"error": str(exc)}
            if tracked.returncode != 0:
                return (
                    "missing",
                    "evidence files exist but are not all tracked by git",
                    {"paths": declared_paths, "git_stderr": tracked.stderr.strip()},
                )
            return "pass", "all required evidence files exist and are tracked by git", {"paths": declared_paths}
        return "pass", "all required evidence files exist", {"paths": declared_paths}

    if kind != "jsonl_query":
        return "unknown", f"unsupported proof type {kind!r}", {}

    declared_path = proof.get("path")
    path = _evidence_path(root, declared_path)
    if not path.exists():
        return "missing", f"evidence file is missing: {declared_path}", {}
    if not path.is_file():
        return "unknown", f"evidence path is not a file: {declared_path}", {}

    candidates = _load_jsonl(path)
    axis = proof.get("axis")
    if axis is not None:
        candidates = [row for row in candidates if row.get("axis") == axis]
    candidates = [row for row in candidates if _row_matches(row, _conditions(proof, "where"))]
    require = _conditions(proof, "require")
    rows = [row for row in candidates if _row_matches(row, require)]
    rejected = len(candidates) - len(rows)

    if proof.get("all_matches") and rejected:
        return (
            "fail",
            f"{rejected}/{len(candidates)} selected evidence rows failed required conditions",
            {
                "path": declared_path,
                "candidates": len(candidates),
                "matches": len(rows),
                "rejected_by_require": rejected,
            },
        )

    try:
        min_matches = int(proof.get("min_matches", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("min_matches must be an integer") from exc
    if min_matches < 0:
        raise ValueError("min_matches must be non-negative")
    if len(rows) < min_matches:
        return (
            "fail",
            f"only {len(rows)} matching rows; require at least {min_matches}",
            {
                "path": declared_path,
                "candidates": len(candidates),
                "matches": len(rows),
                "rejected_by_require": rejected,
            },
        )

    coverage_result: dict[str, Any] = {}
    coverage_items = proof.get("coverage", [])
    if not isinstance(coverage_items, list):
        raise ValueError("coverage must be a list")
    for coverage in coverage_items:
        if not isinstance(coverage, dict) or not isinstance(coverage.get("field"), str):
            raise ValueError("every coverage item must have a string field")
        field = coverage["field"]
        values = coverage.get("values", [])
        if not isinstance(values, list):
            raise ValueError("coverage values must be a list")
        expected = set(values)
        actual = {_field(row, field) for row in rows}
        missing = sorted(expected - actual, key=str)
        coverage_result[field] = {"expected": sorted(expected, key=str), "actual": sorted(actual, key=str)}
        if missing:
            return (
                "fail",
                f"coverage for {field} is missing: {missing}",
                {
                    "path": declared_path,
                    "candidates": len(candidates),
                    "matches": len(rows),
                    "coverage": coverage_result,
                },
            )
    return (
        "pass",
        f"proof matched {len(rows)} evidence rows",
        {
            "path": declared_path,
            "candidates": len(candidates),
            "matches": len(rows),
            "rejected_by_require": rejected,
            "coverage": coverage_result,
        },
    )


def evaluate_proof(proof: dict[str, Any] | None, *, root: Path) -> tuple[str, str, dict[str, Any]]:
    """Evaluate one proof without allowing malformed evidence to abort an audit."""

    try:
        return _evaluate_proof(proof, root=root)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return "unknown", f"invalid or unreadable proof: {exc}", {"error": str(exc)}


def audit(manifest: dict[str, Any], *, root: Path, category: str = "") -> list[dict[str, Any]]:
    gates = manifest.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError("manifest must contain a non-empty gates list")
    ids = [gate.get("id") if isinstance(gate, dict) else None for gate in gates]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("every gate id must be non-empty and unique")
    for gate in gates:
        missing_fields = [field for field in ("category", "title", "criterion") if not gate.get(field)]
        if missing_fields:
            raise ValueError(f"gate {gate.get('id')!r} is missing fields: {', '.join(missing_fields)}")
    results: list[dict[str, Any]] = []
    for gate in gates:
        if category and gate.get("category") != category:
            continue
        status, reason, details = evaluate_proof(gate.get("proof"), root=root)
        results.append(
            {
                "axis": "apple_production_gate",
                "status": status,
                "gate_id": gate["id"],
                "category": gate["category"],
                "title": gate["title"],
                "required": bool(gate.get("required", True)),
                "criterion": gate["criterion"],
                "reason": reason,
                "details": details,
            }
        )
    return results


def summarize(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    required = [row for row in rows if row["required"]]
    counts = {status: sum(row["status"] == status for row in required) for status in GATE_STATUSES}
    complete = bool(required) and counts["pass"] == len(required)
    return {
        "axis": "apple_production_acceptance_summary",
        "status": "pass" if complete else "fail",
        "complete": complete,
        "manifest_version": manifest.get("version"),
        "required_gates": len(required),
        "total_gates": len(rows),
        "counts": counts,
        "completion_ratio": round(counts["pass"] / len(required), 6) if required else 0.0,
        "incomplete_gate_ids": [row["gate_id"] for row in required if row["status"] != "pass"],
    }


def append_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--root", default="", help="repository root; defaults to the manifest's parent of bench/")
    parser.add_argument("--category", default="")
    parser.add_argument("--results", default="")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit non-zero unless every required gate passes")
    parser.add_argument("--fail-on-incomplete", action="store_true", help="deprecated alias for --strict")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.root:
        root = Path(args.root).resolve()
    elif manifest_path.parent.name == "bench":
        root = manifest_path.parents[1]
    else:
        root = manifest_path.parent
    rows = audit(manifest, root=root, category=args.category)
    summary = summarize(rows, manifest)
    if not args.summary_only:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))
    append_jsonl(args.results, [*rows, summary])
    strict = bool(args.strict or args.fail_on_incomplete)
    return 1 if strict and not summary["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
