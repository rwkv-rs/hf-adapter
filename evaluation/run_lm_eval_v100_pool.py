#!/usr/bin/env python3
"""Run the formal lm_eval matrix as independent, resumable V100 units.

The reference recurrence consists of many small eager PyTorch operations, so a
single batch-one process leaves most of a V100 idle. This launcher executes
independent formal units concurrently without changing any unit's lm_eval
command, batch size, samples, or metrics. Batch-one and batch-eight phases use
different safe worker counts because their activation memory differs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_lm_eval_matrix import TASKS


MODELS = (
    ("1.5b", "rwkv7_15b_hf"),
    ("0.4b", "rwkv7_04b_hf"),
    ("0.1b", "rwkv7_01b_hf"),
)

# Start the longest tasks first so one HellaSwag process does not become a long
# tail after every other formal unit has finished.
TASK_ORDER = (
    "hellaswag",
    "lambada_openai",
    "arc_challenge",
    "arc_easy",
    "piqa",
    "winogrande",
    "openbookqa",
    "wikitext",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrent, resumable V100 launcher for the 48 lm_eval units"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--gpu", action="append", default=[])
    parser.add_argument("--batch1-workers-per-gpu", type=int, default=6)
    parser.add_argument("--batch8-workers-per-gpu", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_unit(
    *,
    repo_root: Path,
    output_dir: Path,
    model_root: Path,
    python: str,
    code_sha: str,
    gpu: str,
    label: str,
    folder: str,
    task: str,
    batch_size: int,
    force: bool,
) -> dict:
    unit = f"{label}-b{batch_size}-{task}"
    shard = output_dir / "shards" / unit
    shard.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        str(repo_root / "evaluation" / "run_lm_eval_matrix.py"),
        "--output-dir",
        str(shard),
        "--model",
        f"{label}={model_root / folder}",
        "--task",
        task,
        "--batch-size",
        str(batch_size),
        "--code-sha",
        code_sha,
    ]
    if force:
        command.append("--force")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("HF_DATASETS_OFFLINE", "1")
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (shard / "launcher.log").write_text(completed.stdout, encoding="utf-8")
    return {
        "unit": unit,
        "gpu": gpu,
        "exit_code": completed.returncode,
        "shard": str(shard),
    }


def run_phase(args: argparse.Namespace, batch_size: int, workers_per_gpu: int) -> list[dict]:
    repo_root = Path(__file__).resolve().parents[1]
    gpus = tuple(args.gpu or ("0", "1"))
    jobs = [
        (label, folder, task)
        for task in TASK_ORDER
        for label, folder in MODELS
    ]
    results: list[dict] = []

    # A separate executor per GPU is a hard concurrency limit; jobs are
    # distributed round-robin so both devices receive the same mix of sizes.
    executors = {
        gpu: ThreadPoolExecutor(max_workers=workers_per_gpu) for gpu in gpus
    }
    try:
        futures = []
        for index, (label, folder, task) in enumerate(jobs):
            gpu = gpus[index % len(gpus)]
            futures.append(
                executors[gpu].submit(
                    run_unit,
                    repo_root=repo_root,
                    output_dir=args.output_dir,
                    model_root=args.model_root,
                    python=args.python,
                    code_sha=args.code_sha,
                    gpu=gpu,
                    label=label,
                    folder=folder,
                    task=task,
                    batch_size=batch_size,
                    force=args.force,
                )
            )
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(json.dumps(row), flush=True)
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=False)
    return results


def main() -> None:
    args = parse_args()
    if args.batch1_workers_per_gpu < 1 or args.batch8_workers_per_gpu < 1:
        raise SystemExit("worker counts must be positive")
    if set(TASK_ORDER) != set(TASKS):
        raise SystemExit("pool task order is out of sync with the formal matrix")
    for _, folder in MODELS:
        if not (args.model_root / folder).is_dir():
            raise SystemExit(f"missing model folder: {args.model_root / folder}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(run_phase(args, batch_size=1, workers_per_gpu=args.batch1_workers_per_gpu))
    if any(row["exit_code"] for row in rows):
        raise SystemExit("at least one batch-one unit failed")
    rows.extend(run_phase(args, batch_size=8, workers_per_gpu=args.batch8_workers_per_gpu))
    (args.output_dir / "pool-status.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    if any(row["exit_code"] for row in rows):
        raise SystemExit("at least one batch-eight unit failed")

    merged = args.output_dir / "merged"
    if merged.exists():
        import shutil

        shutil.rmtree(merged)
    merge_command = [
        args.python,
        str(Path(__file__).with_name("merge_lm_eval_shards.py")),
        "--output-dir",
        str(merged),
    ]
    for row in sorted(rows, key=lambda item: item["unit"]):
        merge_command.extend(("--shard", row["shard"]))
    subprocess.run(merge_command, check=True)
    subprocess.run(
        [
            args.python,
            str(Path(__file__).with_name("validate_lm_eval_matrix.py")),
            "--result-dir",
            str(merged),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
