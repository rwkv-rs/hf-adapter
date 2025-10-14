#!/usr/bin/env python3
# coding=utf-8
"""Download a Hugging Face large file with bounded, resumable Range workers.

``huggingface_hub.snapshot_download`` is the preferred path for normal model
snapshots.  This helper is intentionally narrower: it is for public large files
that are reachable through the Hub ``/resolve/`` endpoint but stall during the
large-file/Xet phase.  It resolves the signed CDN URL with ``HEAD``, downloads
fixed byte ranges into ``*.parts/`` files, then combines them into the final
file and appends an optional JSONL evidence row.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

AXIS = "hf_parallel_download"
DEFAULT_ENDPOINT = "https://huggingface.co"

try:  # Keep import-time tests usable when requests is not installed.
    import requests
except Exception:  # pragma: no cover - exercised only in minimal envs
    requests = None  # type: ignore[assignment]


def human_bytes(value: float | int) -> str:
    n = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TiB"


def append_jsonl(path: str | None, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_resolve_url(*, repo_id: str, filename: str, revision: str = "main", endpoint: str = DEFAULT_ENDPOINT) -> str:
    repo = "/".join(quote(part, safe="") for part in repo_id.strip("/").split("/"))
    rev = quote(revision.strip("/"), safe="")
    path = "/".join(quote(part, safe="") for part in filename.strip("/").split("/"))
    return f"{endpoint.rstrip('/')}/{repo}/resolve/{rev}/{path}"


def make_ranges(size: int, chunk_bytes: int) -> list[tuple[int, int, int]]:
    if size <= 0:
        raise ValueError("size must be positive")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    return [(i, start, min(start + chunk_bytes - 1, size - 1)) for i, start in enumerate(range(0, size, chunk_bytes))]


def _base_row(*, url: str, output: Path, size: int | None, started_s: float, status: str) -> dict[str, Any]:
    return {
        "axis": AXIS,
        "status": status,
        "url": url,
        "output": str(output),
        "size_bytes": size,
        "elapsed_s": round(time.monotonic() - started_s, 6),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
    }


def _require_requests() -> Any:
    if requests is None:  # pragma: no cover - depends on environment
        raise RuntimeError("scripts/hf_parallel_download.py requires the 'requests' package")
    return requests


def resolve_download(url: str, *, timeout_s: float) -> tuple[str, int, dict[str, str]]:
    req = _require_requests()
    response = req.head(url, allow_redirects=True, timeout=timeout_s)
    response.raise_for_status()
    size_text = response.headers.get("content-length") or response.headers.get("x-linked-size")
    if not size_text:
        raise RuntimeError("could not determine content length from HEAD response")
    size = int(size_text)
    return response.url, size, dict(response.headers)


def download_range(
    *,
    final_url: str,
    part_path: Path,
    start: int,
    end: int,
    timeout_s: float,
    retries: int,
) -> int:
    req = _require_requests()
    expected = end - start + 1
    have = part_path.stat().st_size if part_path.exists() else 0
    if have == expected:
        return expected
    if have > expected:
        part_path.unlink()
        have = 0

    for attempt in range(retries):
        range_start = start + have
        headers = {"Range": f"bytes={range_start}-{end}"}
        try:
            with req.get(final_url, headers=headers, stream=True, timeout=timeout_s) as response:
                if response.status_code not in (200, 206):
                    body = response.text[:240] if getattr(response, "text", None) else ""
                    raise RuntimeError(f"HTTP {response.status_code}: {body}")
                with part_path.open("ab" if have else "wb") as f:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
            have = part_path.stat().st_size
            if have == expected:
                return expected
            if have > expected:
                part_path.unlink()
                have = 0
            raise RuntimeError(f"partial range got={have} expected={expected}")
        except Exception:
            if attempt + 1 >= retries:
                raise
            time.sleep(min(30.0, 1.0 + attempt * 0.5))
            have = part_path.stat().st_size if part_path.exists() else 0
            if have == expected:
                return expected
    return part_path.stat().st_size if part_path.exists() else 0


def combine_parts(*, output: Path, part_dir: Path, ranges: list[tuple[int, int, int]], keep_parts: bool) -> None:
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("wb") as writer:
        for idx, start, end in ranges:
            part = part_dir / f"{idx:06d}.part"
            expected = end - start + 1
            if not part.exists() or part.stat().st_size != expected:
                actual = part.stat().st_size if part.exists() else None
                raise RuntimeError(f"missing/incomplete part {part}: {actual} != {expected}")
            with part.open("rb") as reader:
                shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
    expected_size = ranges[-1][2] + 1
    if tmp.stat().st_size != expected_size:
        raise RuntimeError(f"combined size mismatch: {tmp.stat().st_size} != {expected_size}")
    tmp.replace(output)
    if not keep_parts:
        shutil.rmtree(part_dir, ignore_errors=True)


def download_file(
    *,
    url: str,
    output: str | Path,
    jobs: int,
    chunk_bytes: int,
    timeout_s: float,
    retries: int,
    results: str | None = None,
    keep_parts: bool = False,
) -> dict[str, Any]:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    started = time.monotonic()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        final_url, size, headers = resolve_download(url, timeout_s=timeout_s)
        if out.exists() and out.stat().st_size == size:
            row = _base_row(url=url, output=out, size=size, started_s=started, status="pass")
            row.update({"reason": "already_complete", "jobs": jobs, "chunk_bytes": chunk_bytes})
            append_jsonl(results, row)
            return row
        if out.exists() and out.stat().st_size not in (0, size):
            backup = out.with_name(f"{out.name}.single.partial.{int(time.time())}")
            out.rename(backup)
            print(f"moved incomplete single-file output to {backup}", flush=True)

        ranges = make_ranges(size, chunk_bytes)
        part_dir = out.with_name(out.name + ".parts")
        part_dir.mkdir(exist_ok=True)
        print(
            f"download start size={size} {human_bytes(size)} jobs={jobs} chunks={len(ranges)} output={out}",
            flush=True,
        )
        last_s = time.monotonic()
        last_done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(
                    download_range,
                    final_url=final_url,
                    part_path=part_dir / f"{idx:06d}.part",
                    start=start,
                    end=end,
                    timeout_s=timeout_s,
                    retries=retries,
                )
                for idx, start, end in ranges
            ]
            while True:
                done = 0
                complete = 0
                for idx, start, end in ranges:
                    part = part_dir / f"{idx:06d}.part"
                    if not part.exists():
                        continue
                    expected = end - start + 1
                    current = min(part.stat().st_size, expected)
                    done += current
                    if current == expected:
                        complete += 1
                now = time.monotonic()
                if now - last_s >= 5.0:
                    rate = (done - last_done) / max(now - last_s, 1e-9)
                    print(
                        f"progress {human_bytes(done)}/{human_bytes(size)} "
                        f"{done / size * 100.0:.2f}% rate={human_bytes(rate)}/s chunks={complete}/{len(ranges)}",
                        flush=True,
                    )
                    last_s = now
                    last_done = done
                if all(f.done() for f in futures):
                    for f in futures:
                        f.result()
                    break
                time.sleep(0.5)

        print("combining parts...", flush=True)
        combine_parts(output=out, part_dir=part_dir, ranges=ranges, keep_parts=keep_parts)
        row = _base_row(url=url, output=out, size=size, started_s=started, status="pass")
        row.update(
            {
                "jobs": jobs,
                "chunk_bytes": chunk_bytes,
                "chunk_count": len(ranges),
                "final_url_host": final_url.split("/", 3)[2] if "://" in final_url else None,
                "etag": headers.get("etag"),
                "output_size_bytes": out.stat().st_size,
            }
        )
        append_jsonl(results, row)
        return row
    except Exception as exc:
        row = _base_row(url=url, output=out, size=None, started_s=started, status="fail")
        row.update({"reason": f"{type(exc).__name__}: {exc}", "jobs": jobs, "chunk_bytes": chunk_bytes})
        append_jsonl(results, row)
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Direct Hub /resolve/ URL or signed CDN URL")
    source.add_argument("--repo-id", help="Hugging Face repo id, for example mlx-community/Qwen3.5-2B-MLX-4bit")
    ap.add_argument("--filename", help="File path inside the repo, required with --repo-id")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--chunk-mib", type=int, default=32)
    ap.add_argument("--timeout-s", type=float, default=90.0)
    ap.add_argument("--retries", type=int, default=50)
    ap.add_argument("--results", default="")
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args(argv)
    if args.repo_id and not args.filename:
        ap.error("--filename is required with --repo-id")
    url = args.url or build_resolve_url(
        repo_id=args.repo_id,
        filename=args.filename,
        revision=args.revision,
        endpoint=args.endpoint,
    )
    row = download_file(
        url=url,
        output=args.output,
        jobs=int(args.jobs),
        chunk_bytes=int(args.chunk_mib) * 1024 * 1024,
        timeout_s=float(args.timeout_s),
        retries=int(args.retries),
        results=args.results or None,
        keep_parts=bool(args.keep_parts),
    )
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0 if row.get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
