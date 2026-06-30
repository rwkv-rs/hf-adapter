#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
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
        "--code-sha",
        default=None,
        help="Source revision for rsync deployments without a .git directory",
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=None,
        help="PR smoke only; formal runs must omit this option",
    )
    parser.add_argument("--wandb-args", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun successful units instead of resuming from the manifest",
    )
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def source_sha(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model(source: str, revision: str | None) -> dict:
    path = Path(source).expanduser()
    if path.exists():
        names = {
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
        }
        files = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file()
            and (
                candidate.name in names
                or candidate.suffix in {".py", ".safetensors"}
                or candidate.name.endswith(".safetensors.index.json")
            )
        )
        hashes = {str(candidate.relative_to(path)): sha256_file(candidate) for candidate in files}
        aggregate = hashlib.sha256()
        for name, digest in hashes.items():
            aggregate.update(name.encode())
            aggregate.update(b"\0")
            aggregate.update(digest.encode())
            aggregate.update(b"\n")
        return {
            "kind": "local",
            "path": str(path.resolve()),
            "requested_revision": revision,
            "resolved_revision": aggregate.hexdigest(),
            "files": hashes,
        }

    result = {
        "kind": "hub",
        "repo_id": source,
        "requested_revision": revision,
        "resolved_revision": None,
    }
    try:
        from huggingface_hub import HfApi

        result["resolved_revision"] = HfApi().model_info(source, revision=revision).sha
    except Exception as error:  # Hub metadata is useful provenance, not an execution gate.
        result["resolution_error"] = f"{type(error).__name__}: {error}"
    return result


def environment() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "transformers",
                "accelerate",
                "datasets",
                "lm_eval",
                "huggingface_hub",
                "wandb",
            )
        },
    }


def read_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["unit"]] = row
    return latest


def find_result_json(unit_dir: Path) -> Path | None:
    candidates = [
        path
        for path in unit_dir.rglob("*.json")
        if "samples_" not in path.name and path.name != "manifest.json"
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def task_provenance(unit_dir: Path, task: str) -> dict:
    result_path = find_result_json(unit_dir)
    if result_path is None:
        return {"result_json": None}
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    config = payload.get("configs", {}).get(task, {})
    samples = payload.get("samples", {}).get(task, [])
    sample_files = sorted(unit_dir.rglob(f"samples_{task}_*.jsonl"))
    sample_digest = hashlib.sha256()
    sample_count = 0

    def add_sample(sample: dict) -> None:
        nonlocal sample_count
        hashes = {
            key: sample[key]
            for key in ("doc_hash", "prompt_hash", "target_hash")
            if key in sample
        }
        sample_digest.update(
            json.dumps(hashes, sort_keys=True, ensure_ascii=False).encode()
        )
        sample_digest.update(b"\n")
        sample_count += 1

    if sample_files:
        with sample_files[-1].open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    add_sample(json.loads(line))
    else:
        for sample in samples:
            add_sample(sample)
    sample_hash_fingerprint = sample_digest.hexdigest()
    provenance = {
        "result_json": str(result_path),
        "samples_jsonl": str(sample_files[-1]) if sample_files else None,
        "task_config": config,
        "dataset_revision": config.get("dataset_kwargs", {}).get("revision"),
        "sample_count": sample_count,
        "sample_hash_fingerprint": sample_hash_fingerprint,
    }
    # lm_eval's task config plus its document hashes are the stable dataset-input
    # fingerprint.  Hash them explicitly instead of depending on a private
    # datasets cache fingerprint that can vary across Arrow versions.
    canonical = json.dumps(
        {
            "task_config": config,
            "sample_count": sample_count,
            "sample_hash_fingerprint": sample_hash_fingerprint,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode()
    provenance["dataset_fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return provenance


def main():
    args = parse_args()
    version = importlib.metadata.version("lm_eval")
    if version != EXPECTED_LM_EVAL:
        raise SystemExit(f"lm_eval=={EXPECTED_LM_EVAL} is required, found {version}")

    specs = [parse_model(value) for value in (args.model or DEFAULT_MODELS)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    latest = read_manifest(manifest_path)
    code_sha = source_sha(args.code_sha)
    runtime = environment()
    model_provenance = {
        label: resolve_model(source, revision) for label, source, revision in specs
    }
    (args.output_dir / "environment.json").write_text(
        json.dumps(runtime, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "models.json").write_text(
        json.dumps(model_provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for label, source, revision in specs:
            for batch_size in (1, 8):
                for task in TASKS:
                    unit = f"{label}-b{batch_size}-{task}"
                    unit_dir = args.output_dir / unit
                    unit_dir.mkdir(parents=True, exist_ok=True)
                    previous = latest.get(unit)
                    if (
                        not args.force
                        and previous is not None
                        and previous.get("exit_code") == 0
                        and previous.get("formal") == (args.smoke_limit is None)
                        and find_result_json(unit_dir) is not None
                    ):
                        print(f"{unit}: SKIP (already passed)")
                        continue
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
                        "--trust_remote_code",
                        "--output_path",
                        str(unit_dir),
                        "--seed",
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
                        "schema_version": 2,
                        "unit": unit,
                        "model_label": label,
                        "model": source,
                        "revision": revision,
                        "model_provenance": model_provenance[label],
                        "code_sha": code_sha,
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
                    if result.returncode == 0:
                        row["task_provenance"] = task_provenance(unit_dir, task)
                    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                    manifest.flush()
                    latest[unit] = row
                    print(f"{unit}: {'PASS' if result.returncode == 0 else 'FAIL'}")

    expected = len(specs) * 2 * len(TASKS)
    expected_units = {
        f"{label}-b{batch_size}-{task}"
        for label, _, _ in specs
        for batch_size in (1, 8)
        for task in TASKS
    }
    failures = sum(
        1 for unit in expected_units if latest.get(unit, {}).get("exit_code") != 0
    )
    print(json.dumps({"units": expected, "failures": failures, "manifest": str(manifest_path)}))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
