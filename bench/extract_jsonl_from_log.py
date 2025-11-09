#!/usr/bin/env python3
"""Extract complete JSON-object lines from a retained acceptance log."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def extract_json_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or not isinstance(value.get("axis"), str):
            continue
        rows.append({**value, "source_log_line": int(line_number)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--fail-if-empty", action="store_true")
    args = parser.parse_args()

    log = Path(args.log)
    text = log.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    rows = extract_json_rows(text)
    enriched = [
        {
            **row,
            "source_log": str(log),
            "source_log_sha256": digest,
            "source_git_commit": args.source_commit or None,
        }
        for row in rows
    ]
    summary = {
        "axis": "extracted_acceptance_log_summary",
        "status": "pass" if rows else "fail",
        "source_log": str(log),
        "source_log_sha256": digest,
        "source_git_commit": args.source_commit or None,
        "rows": len(rows),
        "axes": sorted({str(row["axis"]) for row in rows}),
    }
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in [*enriched, summary]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if args.fail_if_empty and not rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
