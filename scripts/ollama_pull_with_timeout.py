#!/usr/bin/env python3
# coding=utf-8
"""Pull an Ollama model with bounded no-progress timeouts.

The stock ``ollama pull`` CLI prints a spinner while it waits.  That is fine for
interactive use, but it can hide a stuck Apple/Qwen3.5 acceptance run when the
registry keeps returning the same "pulling model" event without byte progress.
This helper talks to the local Ollama HTTP API directly, emits compact progress,
and appends a structured JSONL row that can live next to the benchmark evidence.
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AXIS = "qwen35_apple_ollama_pull"


def append_jsonl(path: str | None, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class PullProgress:
    model: str
    timeout_s: float
    idle_timeout_s: float
    start_s: float
    last_progress_s: float | None = None
    last_signature: tuple[Any, ...] | None = None
    last_completed: int | None = None
    last_event: dict[str, Any] | None = None

    def _base_row(self, *, now_s: float, status: str) -> dict[str, Any]:
        return {
            "axis": AXIS,
            "status": status,
            "engine": "ollama",
            "runtime": "ollama_mlx",
            "model": self.model,
            "elapsed_s": round(float(now_s - self.start_s), 6),
            "timeout_s": float(self.timeout_s),
            "idle_timeout_s": float(self.idle_timeout_s),
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
        }

    def fail_row(self, *, now_s: float, reason: str) -> dict[str, Any]:
        row = self._base_row(now_s=now_s, status="fail")
        row["reason"] = reason
        if self.last_event is not None:
            row["last_pull_event"] = self.last_event
        if self.last_completed is not None:
            row["last_completed"] = int(self.last_completed)
        return row

    def pass_row(self, *, now_s: float, event: dict[str, Any]) -> dict[str, Any]:
        row = self._base_row(now_s=now_s, status="pass")
        row["pull_status"] = event.get("status", "success")
        if self.last_completed is not None:
            row["last_completed"] = int(self.last_completed)
        total = _safe_int(event.get("total"))
        if total is not None:
            row["total"] = total
        return row

    def observe(self, event: dict[str, Any], *, now_s: float) -> dict[str, Any] | None:
        """Return a terminal row when the pull is complete or stuck."""
        self.last_event = dict(event)
        if now_s - self.start_s > self.timeout_s:
            return self.fail_row(now_s=now_s, reason="ollama pull exceeded global timeout")
        if event.get("error"):
            return self.fail_row(now_s=now_s, reason=f"ollama pull error: {event.get('error')}")

        status = str(event.get("status") or "")
        completed = _safe_int(event.get("completed"))
        total = _safe_int(event.get("total"))
        digest = event.get("digest")
        signature = (status, digest, total)

        progressed = False
        if self.last_progress_s is None:
            progressed = True
        elif signature != self.last_signature:
            progressed = True
        elif completed is not None and (self.last_completed is None or completed > self.last_completed):
            progressed = True

        if progressed:
            self.last_progress_s = now_s
            self.last_signature = signature
            if completed is not None:
                self.last_completed = completed

        if status in {"success", "pulled"} or event.get("done") is True:
            return self.pass_row(now_s=now_s, event=event)

        if self.last_progress_s is not None and now_s - self.last_progress_s > self.idle_timeout_s:
            return self.fail_row(
                now_s=now_s,
                reason="ollama pull made no byte/status progress before idle timeout",
            )
        return None


def format_event(event: dict[str, Any]) -> str:
    status = event.get("status", "unknown")
    digest = event.get("digest")
    completed = _safe_int(event.get("completed"))
    total = _safe_int(event.get("total"))
    if completed is not None and total:
        pct = 100.0 * float(completed) / float(total)
        return f"{status} {pct:.1f}% {completed}/{total} {digest or ''}".strip()
    if total:
        return f"{status} total={total} {digest or ''}".strip()
    return json.dumps(event, ensure_ascii=False)


def pull_model(
    *,
    model: str,
    host: str,
    timeout_s: float,
    idle_timeout_s: float,
    results: str | None,
    fail_on_timeout: bool,
) -> int:
    url = host.rstrip("/") + "/api/pull"
    body = json.dumps({"name": model, "stream": True}).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    start = time.monotonic()
    progress = PullProgress(model=model, timeout_s=timeout_s, idle_timeout_s=idle_timeout_s, start_s=start)
    print(f"ollama_pull_start model={model} host={host}", flush=True)
    socket_timeout = max(1.0, min(float(idle_timeout_s), 30.0))
    try:
        with urllib.request.urlopen(request, timeout=socket_timeout) as response:  # noqa: S310 - local Ollama endpoint
            for raw in response:
                now = time.monotonic()
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    row = progress.fail_row(now_s=now, reason=f"invalid Ollama pull JSON: {exc}")
                    append_jsonl(results, row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    return 2 if fail_on_timeout else 0
                print(format_event(event), flush=True)
                terminal = progress.observe(event, now_s=now)
                if terminal is not None:
                    append_jsonl(results, terminal)
                    print(json.dumps(terminal, ensure_ascii=False), flush=True)
                    if terminal["status"] == "pass":
                        return 0
                    return 2 if fail_on_timeout else 0
        now = time.monotonic()
        row = progress.fail_row(now_s=now, reason="ollama pull stream ended without success")
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        now = time.monotonic()
        if isinstance(exc, (TimeoutError, socket.timeout)) and progress.last_event is not None:
            row = progress.fail_row(
                now_s=now,
                reason=f"ollama pull made no byte/status progress before idle timeout: {type(exc).__name__}: {exc}",
            )
        else:
            row = progress.fail_row(now_s=now, reason=f"ollama pull request failed: {type(exc).__name__}: {exc}")
    append_jsonl(results, row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return 2 if fail_on_timeout else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="Ollama model tag to pull, for example qwen3.5:0.8b-mlx")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout-s", type=float, default=7200.0)
    ap.add_argument("--idle-timeout-s", type=float, default=120.0)
    ap.add_argument("--results", default="")
    ap.add_argument("--no-fail-on-timeout", action="store_true")
    args = ap.parse_args(argv)
    if args.timeout_s <= 0:
        ap.error("--timeout-s must be positive")
    if args.idle_timeout_s <= 0:
        ap.error("--idle-timeout-s must be positive")
    return pull_model(
        model=args.model,
        host=args.host,
        timeout_s=float(args.timeout_s),
        idle_timeout_s=float(args.idle_timeout_s),
        results=args.results or None,
        fail_on_timeout=not args.no_fail_on_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
