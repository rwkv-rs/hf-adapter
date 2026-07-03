#!/usr/bin/env python3
# coding=utf-8
"""Probe HF-adapter vs Albatross logits parity on selected MATH500 tasks.

The goal is to localize the MATH500 avg@64 accuracy gap after the benchmark
comparison has shown matching prompt lengths and clean verifier output.  This
script runs both implementations in one process and compares:

- prompt/prefill next-token logits,
- optional HF native prefill logits,
- teacher-forced decode logits along a fixed continuation.

It intentionally writes compact JSON/Markdown reports rather than full logits.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
import torch.nn.functional as F
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


def load_tasks(dataset: str) -> list[Task]:
    rows: list[Task] = []
    with open(dataset, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                Task(
                    index=index,
                    problem=str(item["problem"]),
                    answer=str(item["answer"]),
                    subject=str(item.get("subject", "")),
                    level=str(item.get("level", "")),
                    unique_id=str(item.get("unique_id", index)),
                )
            )
    return rows


def build_prompt(problem: str, prompt_style: str) -> str:
    problem = problem.strip().replace("\r\n", "\n")
    if prompt_style == "fake_think":
        return f"User: {problem}\n\nAssistant: <think></think"
    if prompt_style == "plain":
        return f"User: {problem}\n\nAssistant:"
    raise ValueError(f"unknown prompt style: {prompt_style}")


def load_generation_completions(path: str | Path, sample_id: int) -> dict[int, str]:
    out: dict[int, str] = {}
    if not path:
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("sample_id", -1)) == sample_id:
                task_index = int(row["task_index"])
                out.setdefault(task_index, str(row.get("completion", "")))
    return out


def encode_hf_prompt(tokenizer: Any, problem: str, prompt_style: str, *, add_bos: bool, bos_token_id: int) -> list[int]:
    ids = tokenizer(build_prompt(problem, prompt_style), add_special_tokens=False).input_ids
    if add_bos:
        ids = [int(bos_token_id)] + [int(x) for x in ids]
    return [int(x) for x in ids]


def encode_albatross_prompt(tokenizer: Any, problem: str, prompt_style: str, *, add_bos: bool, bos_token_id: int) -> list[int]:
    ids = [int(x) for x in tokenizer.encode(build_prompt(problem, prompt_style))]
    if add_bos:
        ids = [int(bos_token_id)] + ids
    return ids


def decode_token_safe(hf_tokenizer: Any, token_id: int) -> str:
    try:
        return hf_tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return ""


def top_tokens(logits: torch.Tensor, hf_tokenizer: Any, k: int) -> list[dict[str, Any]]:
    vals, ids = torch.topk(logits.float(), k=min(k, logits.numel()))
    vals_cpu = vals.detach().cpu().tolist()
    ids_cpu = ids.detach().cpu().tolist()
    return [
        {"rank": i + 1, "token_id": int(tok), "logit": float(val), "text": decode_token_safe(hf_tokenizer, int(tok))}
        for i, (tok, val) in enumerate(zip(ids_cpu, vals_cpu))
    ]


def compare_logits(
    hf_logits: torch.Tensor,
    alb_logits: torch.Tensor,
    *,
    target_token: int | None,
    top_k: int,
    tokenizer: Any,
) -> dict[str, Any]:
    hf = hf_logits.detach().float().view(-1)
    alb = alb_logits.detach().float().view(-1)
    if hf.numel() != alb.numel():
        raise ValueError(f"logit vocab mismatch: {hf.numel()} vs {alb.numel()}")
    diff = hf - alb
    hf_top = torch.topk(hf, k=min(top_k, hf.numel())).indices
    alb_top = torch.topk(alb, k=min(top_k, alb.numel())).indices
    hf_top_set = set(int(x) for x in hf_top.detach().cpu().tolist())
    alb_top_set = set(int(x) for x in alb_top.detach().cpu().tolist())
    target_metrics: dict[str, Any] = {}
    if target_token is not None and 0 <= int(target_token) < hf.numel():
        tok = int(target_token)
        hf_lp = torch.log_softmax(hf, dim=-1)[tok]
        alb_lp = torch.log_softmax(alb, dim=-1)[tok]
        target_metrics = {
            "target_token": tok,
            "target_text": decode_token_safe(tokenizer, tok),
            "hf_target_logprob": float(hf_lp.detach().cpu()),
            "albatross_target_logprob": float(alb_lp.detach().cpu()),
            "target_logprob_delta_hf_minus_albatross": float((hf_lp - alb_lp).detach().cpu()),
            "target_nll_delta_hf_minus_albatross": float((-hf_lp + alb_lp).detach().cpu()),
        }
    return {
        "max_abs": float(diff.abs().max().detach().cpu()),
        "mean_abs": float(diff.abs().mean().detach().cpu()),
        "rms": float(torch.sqrt((diff * diff).mean()).detach().cpu()),
        "cosine": float(F.cosine_similarity(hf, alb, dim=0).detach().cpu()),
        "hf_argmax": int(torch.argmax(hf).detach().cpu()),
        "albatross_argmax": int(torch.argmax(alb).detach().cpu()),
        "argmax_match": bool(torch.argmax(hf).item() == torch.argmax(alb).item()),
        "top_k_overlap": len(hf_top_set & alb_top_set),
        "top_k": int(min(top_k, hf.numel())),
        "hf_top": top_tokens(hf, tokenizer, min(top_k, 8)),
        "albatross_top": top_tokens(alb, tokenizer, min(top_k, 8)),
        **target_metrics,
    }


def summarize_step_metrics(steps: list[dict[str, Any]]) -> dict[str, Any]:
    if not steps:
        return {"steps": 0}
    def vals(key: str) -> list[float]:
        return [float(s[key]) for s in steps if key in s and s[key] is not None]
    target_deltas = vals("target_logprob_delta_hf_minus_albatross")
    nll_deltas = vals("target_nll_delta_hf_minus_albatross")
    return {
        "steps": len(steps),
        "argmax_match_rate": sum(int(s.get("argmax_match", False)) for s in steps) / len(steps),
        "top_k_overlap_mean": sum(float(s.get("top_k_overlap", 0)) for s in steps) / len(steps),
        "max_abs_max": max(vals("max_abs")),
        "max_abs_mean": sum(vals("max_abs")) / len(vals("max_abs")),
        "mean_abs_mean": sum(vals("mean_abs")) / len(vals("mean_abs")),
        "rms_mean": sum(vals("rms")) / len(vals("rms")),
        "cosine_min": min(vals("cosine")),
        "cosine_mean": sum(vals("cosine")) / len(vals("cosine")),
        "target_logprob_delta_mean_hf_minus_albatross": (sum(target_deltas) / len(target_deltas)) if target_deltas else None,
        "target_nll_delta_sum_hf_minus_albatross": sum(nll_deltas) if nll_deltas else None,
        "target_nll_delta_mean_hf_minus_albatross": (sum(nll_deltas) / len(nll_deltas)) if nll_deltas else None,
    }


def load_hf(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=DTYPES[args.dtype],
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    return model, tokenizer


def load_albatross(args: argparse.Namespace):
    sys.path.insert(0, args.albatross_dir)
    from rwkv.utils import PIPELINE

    v3a = importlib.import_module(args.albatross_module)
    if args.chdir_albatross:
        os.chdir(args.albatross_dir)
    v3a.MODEL_PATH = args.albatross_model
    v3a.WKV_MODE = args.albatross_wkv
    v3a.EMB_DEVICE = args.albatross_emb
    v3a.RKV_MODE = args.albatross_batched_rkv
    v3a.CMIX_SPARSE = args.albatross_cmix_sparse
    v3a.LOWRANK_WEIGHT = args.albatross_lowrank_weight
    v3a.ORIG_LINEAR_GROUPS = v3a.parse_orig_linear_groups(args.albatross_orig_linear_groups)
    torch.set_grad_enabled(False)
    v3a.load_extensions(v3a.WKV_MODE)
    model = v3a.RWKV7()
    tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")
    return model, tokenizer, v3a


def hf_prefill(model: Any, ids: list[int], device: str, *, native: bool) -> tuple[torch.Tensor, Any]:
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        if native and hasattr(model, "rwkv7_prefill_native"):
            out = model.rwkv7_prefill_native(input_ids, logits_to_keep=1, return_dict=True)
        else:
            out = model(input_ids, use_cache=True, logits_to_keep=1, return_dict=True)
    logits = out.logits[:, -1, :].reshape(-1).detach()
    return logits, out.past_key_values


def hf_forward_all(model: Any, ids: list[int], device: str) -> torch.Tensor:
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids, use_cache=False, return_dict=True)
    return out.logits[0].detach()


def albatross_prefill(model: Any, ids: list[int], token_device: str) -> tuple[torch.Tensor, Any]:
    state = model.zero_state(1)
    tokens = torch.tensor(ids, dtype=torch.long, device=token_device)
    with torch.inference_mode():
        logits = model.forward(tokens, state).view(-1).detach()
    return logits, state


def albatross_forward_all(model: Any, ids: list[int], token_device: str) -> torch.Tensor:
    state = model.zero_state(1)
    tokens = torch.tensor(ids, dtype=torch.long, device=token_device)
    with torch.inference_mode():
        logits = model.forward_all_logits(tokens, state)[0].detach()
    return logits


def hf_step_decode(model: Any, token: int, cache: Any, device: str) -> tuple[torch.Tensor, Any]:
    input_ids = torch.tensor([[int(token)]], dtype=torch.long, device=device)
    with torch.inference_mode():
        if hasattr(model, "rwkv7_forward_token"):
            out = model.rwkv7_forward_token(input_ids, past_key_values=cache, return_dict=True)
        else:
            out = model(input_ids, past_key_values=cache, use_cache=True, return_dict=True)
    return out.logits[:, -1, :].reshape(-1).detach(), out.past_key_values


def albatross_step_decode(model: Any, token: int, state: Any, token_device: str) -> torch.Tensor:
    tokens = torch.tensor([int(token)], dtype=torch.long, device=token_device)
    with torch.inference_mode():
        return model.forward(tokens, state).view(-1).detach()


def analyze_task(
    task: Task,
    *,
    args: argparse.Namespace,
    hf_model: Any,
    hf_tokenizer: Any,
    alb_model: Any,
    alb_tokenizer: Any,
    token_device: str,
    continuation_text: str,
) -> dict[str, Any]:
    hf_prompt_ids = encode_hf_prompt(hf_tokenizer, task.problem, args.prompt_style, add_bos=args.add_bos, bos_token_id=args.bos_token_id)
    alb_prompt_ids = encode_albatross_prompt(alb_tokenizer, task.problem, args.prompt_style, add_bos=args.add_bos, bos_token_id=args.bos_token_id)
    prompt_ids_match = hf_prompt_ids == alb_prompt_ids

    hf_cont_ids = [int(x) for x in hf_tokenizer(continuation_text, add_special_tokens=False).input_ids]
    alb_cont_ids = [int(x) for x in alb_tokenizer.encode(continuation_text)]
    cont_ids_match = hf_cont_ids == alb_cont_ids
    cont_ids = hf_cont_ids[: max(0, args.max_steps)]

    result: dict[str, Any] = {
        "task_index": task.index,
        "problem": task.problem,
        "answer": task.answer,
        "prompt_tokens_hf": len(hf_prompt_ids),
        "prompt_tokens_albatross": len(alb_prompt_ids),
        "prompt_ids_match": prompt_ids_match,
        "continuation_source_sample_id": args.sample_id,
        "continuation_chars": len(continuation_text),
        "continuation_tokens_hf": len(hf_cont_ids),
        "continuation_tokens_albatross": len(alb_cont_ids),
        "continuation_ids_match": cont_ids_match,
        "teacher_forced_steps_requested": args.max_steps,
        "teacher_forced_steps_used": len(cont_ids),
    }
    if not prompt_ids_match:
        result["prompt_first_mismatch"] = first_mismatch(hf_prompt_ids, alb_prompt_ids)
    if not cont_ids_match:
        result["continuation_first_mismatch"] = first_mismatch(hf_cont_ids, alb_cont_ids)

    if not prompt_ids_match:
        result["status"] = "prompt_token_mismatch"
        return result

    alb_prompt_logits, alb_state = albatross_prefill(alb_model, alb_prompt_ids, token_device)
    hf_forward_logits, hf_forward_cache = hf_prefill(hf_model, hf_prompt_ids, args.device, native=False)
    result["prefill_forward_vs_albatross"] = compare_logits(
        hf_forward_logits, alb_prompt_logits, target_token=(cont_ids[0] if cont_ids else None), top_k=args.top_k, tokenizer=hf_tokenizer
    )
    hf_native_cache = None
    if hasattr(hf_model, "rwkv7_prefill_native"):
        hf_native_logits, hf_native_cache = hf_prefill(hf_model, hf_prompt_ids, args.device, native=True)
        result["prefill_native_vs_albatross"] = compare_logits(
            hf_native_logits, alb_prompt_logits, target_token=(cont_ids[0] if cont_ids else None), top_k=args.top_k, tokenizer=hf_tokenizer
        )
        result["prefill_native_vs_forward"] = compare_logits(
            hf_native_logits, hf_forward_logits, target_token=(cont_ids[0] if cont_ids else None), top_k=args.top_k, tokenizer=hf_tokenizer
        )

    # Teacher-forced all-logits comparison uses identical token IDs and checks each next-token position.
    if cont_ids:
        seq_ids = hf_prompt_ids + cont_ids
        hf_all = hf_forward_all(hf_model, seq_ids, args.device)
        alb_all = albatross_forward_all(alb_model, alb_prompt_ids + cont_ids, token_device)
        all_steps: list[dict[str, Any]] = []
        # Step j predicts cont_ids[j] from prefix prompt + cont_ids[:j].
        for j, tok in enumerate(cont_ids):
            pos = len(hf_prompt_ids) - 1 + j
            cmp = compare_logits(hf_all[pos], alb_all[pos], target_token=tok, top_k=args.top_k, tokenizer=hf_tokenizer)
            cmp["step"] = j
            if j < args.keep_step_examples:
                cmp["position"] = pos
            else:
                # Drop verbose top lists for compactness after examples.
                cmp.pop("hf_top", None)
                cmp.pop("albatross_top", None)
            all_steps.append(cmp)
        result["teacher_forced_all_logits"] = summarize_step_metrics(all_steps)
        result["teacher_forced_all_logits_examples"] = all_steps[: args.keep_step_examples]

        # Dynamic-path step comparison: native/forward prefill cache -> one-token recurrent updates.
        hf_cache = hf_native_cache if hf_native_cache is not None else hf_forward_cache
        alb_dyn_state = alb_state
        dyn_steps: list[dict[str, Any]] = []
        for j, tok in enumerate(cont_ids):
            hf_next_logits, hf_cache = hf_step_decode(hf_model, tok, hf_cache, args.device)
            alb_next_logits = albatross_step_decode(alb_model, tok, alb_dyn_state, token_device)
            target = cont_ids[j + 1] if j + 1 < len(cont_ids) else None
            cmp = compare_logits(hf_next_logits, alb_next_logits, target_token=target, top_k=args.top_k, tokenizer=hf_tokenizer)
            cmp["step_after_consuming"] = j
            cmp["consumed_token"] = tok
            if j >= args.keep_step_examples:
                cmp.pop("hf_top", None)
                cmp.pop("albatross_top", None)
            dyn_steps.append(cmp)
        result["teacher_forced_dynamic_path"] = summarize_step_metrics(dyn_steps)
        result["teacher_forced_dynamic_path_examples"] = dyn_steps[: args.keep_step_examples]
    result["status"] = "pass"
    return result


def first_mismatch(a: list[int], b: list[int]) -> dict[str, Any] | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if int(x) != int(y):
            return {"index": i, "hf": int(x), "albatross": int(y)}
    if len(a) != len(b):
        return {"index": min(len(a), len(b)), "hf_len": len(a), "albatross_len": len(b)}
    return None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "tasks": len(rows),
        "statuses": {},
        "prompt_id_mismatches": sum(int(not r.get("prompt_ids_match", False)) for r in rows),
        "continuation_id_mismatches": sum(int(not r.get("continuation_ids_match", False)) for r in rows),
    }
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[str(r.get("status", "unknown"))] = statuses.get(str(r.get("status", "unknown")), 0) + 1
    out["statuses"] = statuses
    for key in ("prefill_forward_vs_albatross", "prefill_native_vs_albatross", "prefill_native_vs_forward"):
        metrics = [r[key] for r in rows if key in r]
        if metrics:
            out[key] = aggregate_logit_metrics(metrics)
    for key in ("teacher_forced_all_logits", "teacher_forced_dynamic_path"):
        metrics = [r[key] for r in rows if key in r]
        if metrics:
            out[key] = aggregate_summary_metrics(metrics)
    return out


def aggregate_logit_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(k: str) -> list[float]:
        return [float(m[k]) for m in metrics if k in m and m[k] is not None]
    return {
        "count": len(metrics),
        "argmax_match_rate": sum(int(m.get("argmax_match", False)) for m in metrics) / len(metrics),
        "max_abs_max": max(vals("max_abs")),
        "max_abs_mean": sum(vals("max_abs")) / len(vals("max_abs")),
        "mean_abs_mean": sum(vals("mean_abs")) / len(vals("mean_abs")),
        "rms_mean": sum(vals("rms")) / len(vals("rms")),
        "cosine_min": min(vals("cosine")),
        "cosine_mean": sum(vals("cosine")) / len(vals("cosine")),
        "target_logprob_delta_mean_hf_minus_albatross": mean_or_none(vals("target_logprob_delta_hf_minus_albatross")),
        "target_nll_delta_mean_hf_minus_albatross": mean_or_none(vals("target_nll_delta_hf_minus_albatross")),
    }


def aggregate_summary_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(k: str) -> list[float]:
        return [float(m[k]) for m in metrics if k in m and m[k] is not None]
    keys = [
        "steps",
        "argmax_match_rate",
        "top_k_overlap_mean",
        "max_abs_max",
        "max_abs_mean",
        "mean_abs_mean",
        "rms_mean",
        "cosine_min",
        "cosine_mean",
        "target_logprob_delta_mean_hf_minus_albatross",
        "target_nll_delta_sum_hf_minus_albatross",
        "target_nll_delta_mean_hf_minus_albatross",
    ]
    out = {"count": len(metrics)}
    for k in keys:
        vs = vals(k)
        if vs:
            out[k + "_mean"] = sum(vs) / len(vs)
            out[k + "_min"] = min(vs)
            out[k + "_max"] = max(vs)
    return out


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def render_markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]
    agg = report["aggregate"]
    lines: list[str] = []
    lines.append("# HF vs Albatross logits parity probe")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append(f"- Tasks: `{cfg['task_indices']}`")
    lines.append(f"- Continuation source: `{cfg['continuations_jsonl']}`, sample `{cfg['sample_id']}`")
    lines.append(f"- Max teacher-forced steps: `{cfg['max_steps']}`")
    lines.append(f"- HF dir: `{cfg['hf_dir']}`")
    lines.append(f"- Albatross module/model: `{cfg['albatross_module']}` / `{cfg['albatross_model']}`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- Statuses: `{agg.get('statuses')}`")
    lines.append(f"- Prompt ID mismatches: `{agg.get('prompt_id_mismatches')}`")
    lines.append(f"- Continuation ID mismatches: `{agg.get('continuation_id_mismatches')}`")
    for key in ("prefill_forward_vs_albatross", "prefill_native_vs_albatross", "prefill_native_vs_forward"):
        if key in agg:
            m = agg[key]
            lines.append("")
            lines.append(f"### {key}")
            lines.append(f"- argmax_match_rate: `{m.get('argmax_match_rate')}`")
            lines.append(f"- cosine_mean/min: `{m.get('cosine_mean')}` / `{m.get('cosine_min')}`")
            lines.append(f"- mean_abs_mean: `{m.get('mean_abs_mean')}`")
            lines.append(f"- max_abs_max: `{m.get('max_abs_max')}`")
            lines.append(f"- target NLL delta mean (HF - Albatross): `{m.get('target_nll_delta_mean_hf_minus_albatross')}`")
    for key in ("teacher_forced_all_logits", "teacher_forced_dynamic_path"):
        if key in agg:
            m = agg[key]
            lines.append("")
            lines.append(f"### {key}")
            lines.append(f"- steps mean/min/max: `{m.get('steps_mean')}` / `{m.get('steps_min')}` / `{m.get('steps_max')}`")
            lines.append(f"- argmax_match_rate mean: `{m.get('argmax_match_rate_mean')}`")
            lines.append(f"- cosine mean/min aggregate: `{m.get('cosine_mean_mean')}` / `{m.get('cosine_min_min')}`")
            lines.append(f"- mean_abs_mean: `{m.get('mean_abs_mean_mean')}`")
            lines.append(f"- max_abs_max: `{m.get('max_abs_max_max')}`")
            lines.append(f"- target NLL delta sum mean (HF - Albatross): `{m.get('target_nll_delta_sum_hf_minus_albatross_mean')}`")
    lines.append("")
    lines.append("## Per-task summary")
    lines.append("")
    lines.append("| Task | Status | Prompt IDs | Cont IDs | Prefill argmax fwd/alb | Prefill argmax native/alb | TF all argmax | TF dynamic argmax |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in report["tasks"]:
        fwd = row.get("prefill_forward_vs_albatross", {})
        nat = row.get("prefill_native_vs_albatross", {})
        tf = row.get("teacher_forced_all_logits", {})
        dyn = row.get("teacher_forced_dynamic_path", {})
        lines.append(
            f"| {row['task_index']} | {row.get('status')} | {row.get('prompt_ids_match')} | {row.get('continuation_ids_match')} | "
            f"{fwd.get('argmax_match')} | {nat.get('argmax_match')} | {tf.get('argmax_match_rate')} | {dyn.get('argmax_match_rate')} |"
        )
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- If `prefill_forward_vs_albatross` is already far from parity, inspect HF weight/layout/math vs Albatross before sampler work.")
    lines.append("- If prefill is close but `teacher_forced_dynamic_path` diverges, inspect recurrent state update / fast-token cache path.")
    lines.append("- If logits are close but sampled generations still diverge, inspect sampler RNG and dynamic refill order.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--albatross-dir", required=True)
    ap.add_argument("--albatross-model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--continuations-jsonl", required=True, help="Generation JSONL used as fixed continuation source")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", default="73,160,116,67,277,374,383,319,72")
    ap.add_argument("--sample-id", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=64)
    ap.add_argument("--keep-step-examples", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--prompt-style", choices=("fake_think", "plain"), default="fake_think")
    ap.add_argument("--add-bos", action="store_true", default=True)
    ap.add_argument("--no-add-bos", dest="add_bos", action="store_false")
    ap.add_argument("--bos-token-id", type=int, default=0)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--albatross-module", default="rwkv7_fast_v3a")
    ap.add_argument("--albatross-wkv", default="fp32io16")
    ap.add_argument("--albatross-emb", default="cpu")
    ap.add_argument("--albatross-batched-rkv", default="off")
    ap.add_argument("--albatross-cmix-sparse", default="no-fc")
    ap.add_argument("--albatross-lowrank-weight", default="both")
    ap.add_argument("--albatross-orig-linear-groups", default="att_c2c,ffn_key,head")
    ap.add_argument("--chdir-albatross", action="store_true", default=True)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    tasks = {t.index: t for t in load_tasks(args.dataset)}
    task_indices = [int(x) for x in args.tasks.split(",") if x.strip()]
    continuations = load_generation_completions(args.continuations_jsonl, args.sample_id)

    t0 = time.perf_counter()
    hf_model, hf_tokenizer = load_hf(args)
    alb_model, alb_tokenizer, _ = load_albatross(args)
    token_device = "cpu" if getattr(alb_model, "emb_cpu", False) else args.device

    rows: list[dict[str, Any]] = []
    for task_index in task_indices:
        if task_index not in tasks:
            raise KeyError(f"task {task_index} not found in dataset")
        cont = continuations.get(task_index, "")
        if not cont:
            raise KeyError(f"no continuation for task {task_index} sample_id={args.sample_id} in {args.continuations_jsonl}")
        print(f"logits_parity task={task_index} continuation_chars={len(cont)}", flush=True)
        rows.append(
            analyze_task(
                tasks[task_index],
                args=args,
                hf_model=hf_model,
                hf_tokenizer=hf_tokenizer,
                alb_model=alb_model,
                alb_tokenizer=alb_tokenizer,
                token_device=token_device,
                continuation_text=cont,
            )
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    report = {
        "axis": "math500_logits_parity",
        "status": "pass",
        "elapsed_sec": time.perf_counter() - t0,
        "config": {
            "hf_dir": args.hf_dir,
            "albatross_dir": args.albatross_dir,
            "albatross_model": args.albatross_model,
            "albatross_module": args.albatross_module,
            "dataset": args.dataset,
            "continuations_jsonl": args.continuations_jsonl,
            "task_indices": task_indices,
            "sample_id": args.sample_id,
            "max_steps": args.max_steps,
            "top_k": args.top_k,
            "prompt_style": args.prompt_style,
            "add_bos": args.add_bos,
            "dtype": args.dtype,
            "device": args.device,
            "albatross_wkv": args.albatross_wkv,
            "albatross_emb": args.albatross_emb,
            "albatross_batched_rkv": args.albatross_batched_rkv,
            "albatross_cmix_sparse": args.albatross_cmix_sparse,
            "albatross_lowrank_weight": args.albatross_lowrank_weight,
            "albatross_orig_linear_groups": args.albatross_orig_linear_groups,
        },
        "aggregate": aggregate(rows),
        "tasks": rows,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logits_parity_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {out_dir / 'logits_parity_report.json'}", flush=True)


if __name__ == "__main__":
    main()
