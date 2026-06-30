#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge parallel lm_eval matrix shards")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest: dict[str, dict] = {}
    provenance = args.output_dir / "shard_provenance"
    provenance.mkdir(exist_ok=True)

    for index, shard in enumerate(args.shard):
        rows = [
            json.loads(line)
            for line in (shard / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            unit = row["unit"]
            previous = latest.get(unit)
            if previous is not None and previous != row:
                raise SystemExit(f"conflicting duplicate unit across shards: {unit}")
            latest[unit] = row
            source = shard / unit
            if source.is_dir():
                shutil.copytree(source, args.output_dir / unit, dirs_exist_ok=True)
        shard_meta = provenance / f"{index:02d}-{shard.name}"
        shard_meta.mkdir(exist_ok=True)
        for name in ("environment.json", "models.json"):
            source = shard / name
            if source.is_file():
                shutil.copy2(source, shard_meta / name)

    manifest = args.output_dir / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(latest[unit], ensure_ascii=False) + "\n" for unit in sorted(latest)),
        encoding="utf-8",
    )
    report = {
        "status": "merged",
        "shards": [str(path) for path in args.shard],
        "unique_units": len(latest),
        "manifest": str(manifest),
    }
    (args.output_dir / "merge.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
