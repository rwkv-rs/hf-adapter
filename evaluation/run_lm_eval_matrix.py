#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_LM_EVAL = "0.4.9.1"
TASKS = (
    "wikitext",
    "lambada_openai",
    "piqa",
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
)
DEFAULT_MODELS = (
    "0.1b=wangyue114514/rwkv7-g1d-0.1b-hf@v0.9.0",
    "0.4b=wangyue114514/rwkv7-g1d-0.4b-hf@v0.9.0",
    "1.5b=wangyue114514/rwkv7-g1g-1.5b-hf@v0.9.0",
)


def parse_model(value: str):
    label, location = value.split("=", 1)
    if "@" in location:
        source, revision = location.rsplit("@", 1)
    else:
        source, revision = location, None
    return label, source, revision


def parse_args():
    parser = argparse.ArgumentParser(description="Run the formal 48-unit lm_eval matrix")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=None,
        help="PR smoke only; formal runs must omit this option",
    )
    parser.add_argument("--wandb-args", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    version = importlib.metadata.version("lm_eval")
    if version != EXPECTED_LM_EVAL:
        raise SystemExit(f"lm_eval=={EXPECTED_LM_EVAL} is required, found {version}")

    specs = [parse_model(value) for value in (args.model or DEFAULT_MODELS)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    failures = 0

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for label, source, revision in specs:
            for batch_size in (1, 8):
                for task in TASKS:
                    unit = f"{label}-b{batch_size}-{task}"
                    unit_dir = args.output_dir / unit
                    unit_dir.mkdir(parents=True, exist_ok=True)
                    model_parts = [f"pretrained={source}"]
                    if revision is not None:
                        model_parts.append(f"revision={revision}")
                    model_parts.extend(
                        [
                            f"max_length={args.max_length}",
                            "dtype=float16",
                            "trust_remote_code=True",
                        ]
                    )
                    model_args = ",".join(model_parts)
                    command = [
                        sys.executable,
                        "-m",
                        "lm_eval",
                        "--model",
                        "hf",
                        "--model_args",
                        model_args,
                        "--tasks",
                        task,
                        "--batch_size",
                        str(batch_size),
                        "--num_fewshot",
                        "0",
                        "--device",
                        args.device,
                        "--show_config",
                        "--log_samples",
                        "--output_path",
                        str(unit_dir),
                        "--random_seed",
                        str(args.seed),
                        "--numpy_random_seed",
                        str(args.seed),
                        "--torch_random_seed",
                        str(args.seed),
                        "--fewshot_random_seed",
                        str(args.seed),
                    ]
                    if args.smoke_limit is not None:
                        command += ["--limit", str(args.smoke_limit)]
                    if args.wandb_args:
                        command += ["--wandb_args", args.wandb_args]
                    stdout_path = unit_dir / "stdout.log"
                    stderr_path = unit_dir / "stderr.log"
                    started = datetime.now(timezone.utc).isoformat()
                    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                        result = subprocess.run(command, stdout=stdout, stderr=stderr)
                    row = {
                        "schema_version": 1,
                        "unit": unit,
                        "model": source,
                        "revision": revision,
                        "task": task,
                        "batch_size": batch_size,
                        "max_length": args.max_length,
                        "seed": args.seed,
                        "lm_eval": version,
                        "formal": args.smoke_limit is None,
                        "limit": args.smoke_limit,
                        "started_at": started,
                        "ended_at": datetime.now(timezone.utc).isoformat(),
                        "command": command,
                        "stdout": str(stdout_path),
                        "stderr": str(stderr_path),
                        "exit_code": result.returncode,
                    }
                    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                    manifest.flush()
                    failures += int(result.returncode != 0)
                    print(f"{unit}: {'PASS' if result.returncode == 0 else 'FAIL'}")

    expected = len(specs) * 2 * len(TASKS)
    print(json.dumps({"units": expected, "failures": failures, "manifest": str(manifest_path)}))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
