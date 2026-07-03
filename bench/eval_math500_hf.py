#!/usr/bin/env python3
# coding=utf-8
"""MATH500 evaluator for the RWKV-7 HF adapter.

The output schema mirrors the key fields from BlinkDL/Albatross
`faster3a_2605/eval_math500.py`: rollout accuracy, pass@rollout accuracy,
sample/sec, token/sec, and generation/verification artifacts.  This script is
intended for acceptance testing of the HF adapter; full acceptance should use
`--rollout 64 --max-new-tokens 1500` on the complete MATH500 file.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
from math_verify import parse, verify
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


@dataclass(frozen=True)
class Task:
    index: int
    problem: str
    answer: str
    subject: str = ""
    level: str = ""
    unique_id: str = ""


def load_tasks(dataset: str, limit: int = 0) -> list[Task]:
    tasks: list[Task] = []
    with open(dataset, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            tasks.append(
                Task(
                    index=index,
                    problem=str(item["problem"]),
                    answer=str(item["answer"]),
                    subject=str(item.get("subject", "")),
                    level=str(item.get("level", "")),
                    unique_id=str(item.get("unique_id", index)),
                )
            )
            if limit > 0 and len(tasks) >= limit:
                break
    return tasks


def build_prompt(problem: str, prompt_style: str) -> str:
    problem = problem.strip().replace("\r\n", "\n")
    if prompt_style == "fake_think":
        return f"User: {problem}\n\nAssistant: <think></think"
    if prompt_style == "plain":
        return f"User: {problem}\n\nAssistant:"
    raise ValueError(f"unknown prompt style: {prompt_style}")


def verify_completion(answer: str, completion: str) -> tuple[bool, str]:
    try:
        gold = parse(f"$\\boxed{{{answer}}}$")
        pred = parse(str(completion))
        return bool(pred and verify(gold, pred, strict=False)), ""
    except Exception as exc:  # pragma: no cover - depends on math_verify parsers
        return False, f"{type(exc).__name__}: {exc}"


def trim_completion(text: str) -> str:
    text = text.split("\nUser:", 1)[0]
    if text.startswith(">"):
        text = text[1:]
    return text.strip()


def generate_one(args: argparse.Namespace, model, tokenizer, task: Task, sample_id: int) -> dict[str, Any]:
    prompt = build_prompt(task.problem, args.prompt_style)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_tokens = int(enc.input_ids.shape[1])
    if prompt_tokens + args.max_new_tokens > args.ctx_limit:
        keep = max(1, args.ctx_limit - args.max_new_tokens)
        enc["input_ids"] = enc.input_ids[:, -keep:]
        if "attention_mask" in enc:
            enc["attention_mask"] = enc.attention_mask[:, -keep:]
        prompt_tokens = int(enc.input_ids.shape[1])
    enc = {k: v.to(args.device) for k, v in enc.items()}
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "use_cache": True,
        "pad_token_id": args.pad_token_id,
    }
    if args.temperature > 0:
        gen_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    started = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**enc, **gen_kwargs)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    new_ids = out[0, prompt_tokens:]
    completion = trim_completion(tokenizer.decode(new_ids.detach().cpu().tolist(), skip_special_tokens=False))
    correct, verify_error = verify_completion(task.answer, completion)
    generated_tokens = int(new_ids.numel())
    ended_eod = bool(generated_tokens > 0 and int(new_ids[-1].detach().cpu()) == args.eos_token_id)
    ended_user_stop = "\nUser:" in tokenizer.decode(new_ids.detach().cpu().tolist(), skip_special_tokens=False)
    return {
        "task_index": task.index,
        "local_task_index": task.index,
        "sample_id": sample_id,
        "problem": task.problem,
        "answer": task.answer,
        "subject": task.subject,
        "level": task.level,
        "unique_id": task.unique_id,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "tokens_including_eod": generated_tokens,
        "tokens_including_stop": generated_tokens,
        "ended_eod": ended_eod,
        "ended_user_stop": ended_user_stop,
        "stop_reason": "eod" if ended_eod else "user_stop" if ended_user_stop else "max_tokens",
        "truncated": generated_tokens >= args.max_new_tokens and not ended_eod and not ended_user_stop,
        "completion": completion,
        "correct": correct,
        "verify_error": verify_error,
        "generate_sec": elapsed,
        "generate_tokps": generated_tokens / max(elapsed, 1e-9),
    }


def summarize(args: argparse.Namespace, rows: list[dict[str, Any]], started: float, out_dir: Path) -> dict[str, Any]:
    by_task: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(int(row["task_index"]), []).append(row)
    total = len(rows)
    correct_generations = sum(int(row["correct"]) for row in rows)
    pass_tasks = sum(1 for task_rows in by_task.values() if any(row["correct"] for row in task_rows))
    elapsed = time.perf_counter() - started
    tokens = sum(int(row.get("tokens_including_stop", row["generated_tokens"])) for row in rows)
    return {
        "axis": "math500_hf_adapter",
        "backend": "hf_adapter",
        "status": "pass",
        "model": args.hf_dir,
        "dataset": args.dataset,
        "num_tasks": len(by_task),
        "rollout": args.rollout,
        "total_generations": total,
        "correct_generations": correct_generations,
        "rollout_accuracy": correct_generations / max(total, 1),
        "pass_at_rollout_accuracy": pass_tasks / max(len(by_task), 1),
        "ended_eod": sum(int(row["ended_eod"]) for row in rows),
        "ended_user_stop": sum(int(row["ended_user_stop"]) for row in rows),
        "truncated": sum(int(row["truncated"]) for row in rows),
        "truncated_rate": sum(int(row["truncated"]) for row in rows) / max(total, 1),
        "mean_generated_tokens": sum(int(row["generated_tokens"]) for row in rows) / max(total, 1),
        "mean_tokens_including_stop": tokens / max(total, 1),
        "elapsed_sec": elapsed,
        "sample_per_sec": total / max(elapsed, 1e-9),
        "token_per_sec": tokens / max(elapsed, 1e-9),
        "generations_jsonl": str(out_dir / "generations.jsonl"),
        "config": {
            "hf_dir": args.hf_dir,
            "dataset": args.dataset,
            "rollout": args.rollout,
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
            "ctx_limit": args.ctx_limit,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "prompt_style": args.prompt_style,
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", default="math500_hf_runs")
    ap.add_argument("--rollout", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1500)
    ap.add_argument("--ctx-limit", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.28)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-style", choices=("fake_think", "plain"), default="fake_think")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pad-token-id", type=int, default=0)
    ap.add_argument("--eos-token-id", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    tasks = load_tasks(args.dataset, args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=DTYPES[args.dtype],
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    work_total = len(tasks) * args.rollout
    for task in tasks:
        for sample_id in range(args.rollout):
            row = generate_one(args, model, tokenizer, task, sample_id)
            rows.append(row)
            if args.progress_every > 0 and len(rows) % args.progress_every == 0:
                tokens = sum(int(r["generated_tokens"]) for r in rows)
                elapsed = time.perf_counter() - started
                print(
                    f"math500_hf progress rows={len(rows)}/{work_total} "
                    f"tokens={tokens} tps={tokens / max(elapsed, 1e-9):.1f}",
                    flush=True,
                )

    rows.sort(key=lambda x: (x["task_index"], x["sample_id"]))
    with (out_dir / "generations.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(args, rows, started, out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("MATH500_HF_ADAPTER_RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
