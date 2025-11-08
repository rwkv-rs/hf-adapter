#!/usr/bin/env python3
"""Run the isolated wheel-install suite and emit machine-readable evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AXIS = "apple_clean_install_acceptance"


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def append_jsonl(path: str | Path | None, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def last_int(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return int(matches[-1]) if matches else None


def parse_test_counts(text: str) -> dict[str, int | None]:
    # The first pytest invocation is collect-only; the last summary belongs to
    # the actual profile. Keep both so collection cannot be inferred from a
    # passing subset.
    collected = last_int(r"(\d+)\s+(?:tests?\s+)?collected", text)
    return {
        "collected": collected,
        "passed": last_int(r"(\d+)\s+passed", text),
        "failed": last_int(r"(\d+)\s+failed", text) or 0,
        "skipped": last_int(r"(\d+)\s+skipped", text) or 0,
        "errors": last_int(r"(\d+)\s+errors?", text) or 0,
    }


def evidence_row(
    *,
    profile: str,
    command: list[str],
    returncode: int | None,
    output: str,
    elapsed_s: float,
    timed_out: bool,
    log_path: str,
) -> dict[str, Any]:
    counts = parse_test_counts(output)
    import_pass = "installed rwkv7-hf-adapter=" in output and "clean-install import leaked" not in output
    pip_check_pass = "No broken requirements found." in output
    collection_pass = bool(counts["collected"] and counts["errors"] == 0)
    status = (
        "pass"
        if returncode == 0 and not timed_out and import_pass and pip_check_pass and collection_pass
        else "fail"
    )
    return {
        "axis": AXIS,
        "status": status,
        "profile": profile,
        "command": command,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain", "--untracked-files=no")),
        "platform": platform.platform(),
        "python_launcher": platform.python_version(),
        "returncode": returncode,
        "timed_out": bool(timed_out),
        "elapsed_s": round(float(elapsed_s), 6),
        "clean_wheel_import_pass": bool(import_pass),
        "pip_check_pass": bool(pip_check_pass),
        "pytest_collection_pass": bool(collection_pass),
        **counts,
        "log_path": log_path,
        "log_sha256": hashlib.sha256(output.encode("utf-8", errors="replace")).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="full", choices=["smoke", "full", "apple"])
    parser.add_argument("--python", default="", help="Optional PYTHON_BIN for the isolated runner")
    parser.add_argument("--test-model", default="", help="Optional real converted model for Apple profile")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--results", default="")
    parser.add_argument("--log", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")

    command = [str(ROOT / "scripts" / "run_clean_install_tests.sh"), args.profile]
    if args.dry_run:
        row = {
            "axis": AXIS,
            "status": "plan",
            "profile": args.profile,
            "command": command,
            "git_commit": git_value("rev-parse", "HEAD"),
        }
        print(json.dumps(row, ensure_ascii=False))
        append_jsonl(args.results, row)
        return 0

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    if args.python:
        env["PYTHON_BIN"] = args.python
    if args.test_model:
        env["RWKV7_TEST_MODEL"] = str(Path(args.test_model).resolve())
    if args.profile == "apple":
        env["RWKV7_REQUIRE_APPLE"] = "1"

    started = time.perf_counter()
    timed_out = False
    returncode: int | None
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(args.timeout),
            check=False,
        )
        output = completed.stdout
        returncode = int(completed.returncode)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        raw = exc.stdout or ""
        output = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        output += f"\nTIMEOUT after {args.timeout}s\n"
    elapsed_s = time.perf_counter() - started

    log_path = args.log or f"bench/logs/clean_install_{args.profile}.log"
    log = Path(log_path)
    if not log.is_absolute():
        log = ROOT / log
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(output, encoding="utf-8")
    try:
        relative_log = str(log.relative_to(ROOT))
    except ValueError:
        relative_log = str(log)

    row = evidence_row(
        profile=args.profile,
        command=command,
        returncode=returncode,
        output=output,
        elapsed_s=elapsed_s,
        timed_out=timed_out,
        log_path=relative_log,
    )
    print(json.dumps(row, ensure_ascii=False))
    append_jsonl(args.results, row)
    return 1 if args.fail_on_gate and row["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
