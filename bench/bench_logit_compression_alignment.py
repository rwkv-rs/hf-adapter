#!/usr/bin/env python3
# coding=utf-8
"""Uncheatable logit-alignment benchmark using compression/NLL.

This is stricter than max-diff / cosine / greedy checks.  It scores the
probability assigned by two inference paths to fixed, external target tokens
and reports:

* bits/token for the reference and candidate paths;
* candidate/reference compression ratio;
* compression ratio vs token-position bins.

The target tokens come from input text / JSONL fields, not from either model's
sampled generations, so the metric cannot be gamed by matching only sampled
outputs.  The default JSONL template is MATH500-oriented and uses the problem
plus the gold answer as an external teacher-forced sequence.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
DEFAULT_TEXT_TEMPLATE = "User: {problem}\n\nAssistant: The final answer is \\boxed{{{answer}}}."
FALSE_VALUES = {"", "0", "false", "False", "no", "off", "none"}


@dataclass(frozen=True)
class TextItem:
    index: int
    text: str
    metadata: dict[str, Any]


@dataclass
class ScoreAggregate:
    name: str
    total_bits: float = 0.0
    total_tokens: int = 0
    by_position: dict[str, dict[str, float]] | None = None
    per_text: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.by_position is None:
            self.by_position = {}
        if self.per_text is None:
            self.per_text = []

    @property
    def bits_per_token(self) -> float | None:
        if self.total_tokens <= 0:
            return None
        return self.total_bits / self.total_tokens


def env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw not in FALSE_VALUES


def infer_model_size_label(path: str, explicit: str = "") -> str | None:
    if explicit:
        return explicit.lower()
    match = re.search(r"(\d+(?:\.\d+)?b)", Path(path).name.lower())
    return match.group(1) if match else None


def cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_texts(args: argparse.Namespace) -> list[TextItem]:
    path = Path(args.dataset)
    if not path.exists():
        raise FileNotFoundError(f"dataset does not exist: {path}")
    rows: list[TextItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if args.text_field:
                text = str(obj.get(args.text_field, ""))
            else:
                safe = {str(k): str(v) for k, v in obj.items()}
                text = args.text_template.format_map(_FormatDefault(safe))
            text = text.strip()
            if not text:
                continue
            rows.append(
                TextItem(
                    index=int(obj.get("index", len(rows))),
                    text=text,
                    metadata={
                        "line_no": line_no,
                        "problem": str(obj.get("problem", "")),
                        "answer": str(obj.get("answer", "")),
                        "subject": str(obj.get("subject", "")),
                        "level": str(obj.get("level", "")),
                        "unique_id": str(obj.get("unique_id", obj.get("index", len(rows)))),
                    },
                )
            )
            if args.limit > 0 and len(rows) >= args.limit:
                break
    if not rows:
        raise RuntimeError(f"no scoring texts loaded from {path}")
    return rows


class _FormatDefault(dict):
    def __missing__(self, key: str) -> str:
        return ""


def load_common_tokenizer(args: argparse.Namespace):
    candidates = [
        args.tokenizer_dir,
        args.reference_hf_dir if args.reference_kind == "hf" else "",
        args.candidate_hf_dir if args.candidate_kind == "hf" else "",
    ]
    for path in candidates:
        if path:
            return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    raise ValueError("a HF tokenizer source is required; pass --tokenizer-dir or at least one HF model dir")


def encode_text(tokenizer: Any, text: str, *, add_bos: bool, bos_token_id: int, max_tokens: int) -> list[int]:
    ids = [int(x) for x in tokenizer(text, add_special_tokens=False).input_ids]
    if add_bos:
        ids = [int(bos_token_id)] + ids
    if max_tokens > 0 and len(ids) > max_tokens:
        ids = ids[:max_tokens]
    return ids


def parse_position_bins(raw: str) -> list[tuple[int, int, str]]:
    bounds = [int(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]
    bounds = sorted({b for b in bounds if b > 0})
    if not bounds:
        bounds = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    bins: list[tuple[int, int, str]] = []
    start = 1
    for bound in bounds:
        if bound < start:
            continue
        label = str(bound) if start == bound else f"{start}-{bound}"
        bins.append((start, bound, label))
        start = bound + 1
    bins.append((start, 10**18, f">{bounds[-1]}"))
    return bins


def position_bin(position: int, bins: list[tuple[int, int, str]]) -> str:
    for lo, hi, label in bins:
        if lo <= position <= hi:
            return label
    return bins[-1][2]


class HFScorer:
    def __init__(self, args: argparse.Namespace, prefix: str) -> None:
        self.args = args
        self.prefix = prefix
        hf_dir = getattr(args, f"{prefix}_hf_dir")
        if not hf_dir:
            raise ValueError(f"--{prefix.replace('_', '-')}-hf-dir is required for HF scorer")
        dtype = DTYPES[getattr(args, f"{prefix}_dtype")]
        self.quantization = getattr(args, f"{prefix}_quantization")
        quant_policy = str(getattr(args, f"{prefix}_quant_policy"))
        os.environ["RWKV7_FAST_TOKEN_BACKEND"] = getattr(args, f"{prefix}_fast_token_backend")
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "device_map": args.device if args.device.startswith("cuda") else None,
        }
        if self.quantization in {"bnb8", "bnb8_a8w8_head"}:
            os.environ["RWKV7_BNB_SKIP_POLICY"] = quant_policy
            threshold = float(os.environ.get("RWKV7_BNB_INT8_THRESHOLD", "6.0"))
            if threshold < 0.0:
                raise ValueError("RWKV7_BNB_INT8_THRESHOLD must be non-negative")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=threshold,
            )
        elif self.quantization == "bnb4":
            os.environ["RWKV7_BNB_SKIP_POLICY"] = quant_policy
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=False,
            )
        self.model = AutoModelForCausalLM.from_pretrained(hf_dir, **load_kwargs).eval()
        self.device = args.device
        self.hf_dir = hf_dir
        if self.quantization in {
            "a8w8",
            "mm8",
            "mm4",
            "torchao_w8",
            "torchao_w4",
            "bnb8_a8w8_head",
        }:
            native_quantization = (
                "a8w8" if self.quantization == "bnb8_a8w8_head" else self.quantization
            )
            self._quantize(
                native_quantization,
                min_params=int(getattr(args, f"{prefix}_quant_min_params")),
                policy="speed" if self.quantization == "bnb8_a8w8_head" else quant_policy,
                group_size=int(getattr(args, f"{prefix}_quant_group_size")),
            )
        threshold_tag = (
            f":threshold={float(os.environ.get('RWKV7_BNB_INT8_THRESHOLD', '6.0')):g}"
            if self.quantization in {"bnb8", "bnb8_a8w8_head"}
            else ""
        )
        self.name = f"hf:{Path(hf_dir).name}:{self.quantization}{threshold_tag}"

    def _quantize(
        self,
        quantization: str,
        *,
        min_params: int,
        policy: str,
        group_size: int,
    ) -> None:
        if quantization == "a8w8":
            from rwkv7_hf.native_quant_a8w8 import quantize_model_a8w8

            quantize_model_a8w8(self.model, min_params=min_params, policy=policy)
        elif quantization == "mm8":
            from rwkv7_hf.native_quant_mm8 import quantize_model_mm8

            quantize_model_mm8(self.model, min_params=min_params, fused=True, policy=policy)
        elif quantization == "mm4":
            from rwkv7_hf.native_quant_mm4 import quantize_model_mm4

            quantize_model_mm4(self.model, min_params=min_params, policy=policy)
        elif quantization in {"torchao_w8", "torchao_w4"}:
            from rwkv7_hf.native_quant_torchao import quantize_model_torchao

            quantize_model_torchao(
                self.model,
                quantization,
                min_params=min_params,
                policy=policy,
                group_size=group_size,
            )
        else:  # pragma: no cover
            raise ValueError(f"unsupported quantization: {quantization}")

    def forward_all_logits(self, ids: list[int]) -> torch.Tensor:
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            out = self.model(input_ids, use_cache=False, return_dict=True)
        logits = out.logits[0].detach()
        return logits

    def close(self) -> None:
        del self.model


class AlbatrossScorer:
    def __init__(self, args: argparse.Namespace, prefix: str) -> None:
        self.args = args
        self.prefix = prefix
        alb_dir = getattr(args, f"{prefix}_albatross_dir")
        alb_model = getattr(args, f"{prefix}_albatross_model")
        if not alb_dir or not alb_model:
            raise ValueError(
                f"--{prefix.replace('_', '-')}-albatross-dir and --{prefix.replace('_', '-')}-albatross-model are required"
            )
        sys.path.insert(0, alb_dir)
        if getattr(args, f"{prefix}_chdir_albatross"):
            os.chdir(alb_dir)
        module_name = getattr(args, f"{prefix}_albatross_module")
        self.v3a = importlib.import_module(module_name)
        self.v3a.MODEL_PATH = alb_model
        self.v3a.WKV_MODE = getattr(args, f"{prefix}_albatross_wkv")
        self.v3a.EMB_DEVICE = getattr(args, f"{prefix}_albatross_emb")
        self.v3a.RKV_MODE = getattr(args, f"{prefix}_albatross_batched_rkv")
        self.v3a.CMIX_SPARSE = getattr(args, f"{prefix}_albatross_cmix_sparse")
        self.v3a.LOWRANK_WEIGHT = getattr(args, f"{prefix}_albatross_lowrank_weight")
        self.v3a.ORIG_LINEAR_GROUPS = self.v3a.parse_orig_linear_groups(getattr(args, f"{prefix}_albatross_orig_linear_groups"))
        torch.set_grad_enabled(False)
        self.v3a.load_extensions(self.v3a.WKV_MODE)
        self.model = self.v3a.RWKV7()
        self.token_device = "cpu" if getattr(self.model, "emb_cpu", False) else args.device
        self.name = f"albatross:{Path(alb_model).name}"

    def forward_all_logits(self, ids: list[int]) -> torch.Tensor:
        if not hasattr(self.model, "forward_all_logits"):
            raise RuntimeError("Albatross module does not expose forward_all_logits; cannot run compression benchmark")
        state = self.model.zero_state(1)
        tokens = torch.tensor(ids, dtype=torch.long, device=self.token_device)
        with torch.inference_mode():
            logits = self.model.forward_all_logits(tokens, state)[0].detach()
        return logits

    def close(self) -> None:
        del self.model


def build_scorer(args: argparse.Namespace, prefix: str):
    kind = getattr(args, f"{prefix}_kind")
    if kind == "hf":
        return HFScorer(args, prefix)
    if kind == "albatross":
        return AlbatrossScorer(args, prefix)
    raise ValueError(f"unknown scorer kind: {kind}")


def nll_bits_for_ids(
    logits: torch.Tensor,
    ids: list[int],
    *,
    logit_chunk_size: int,
) -> torch.Tensor:
    if logits.dim() != 2:
        logits = logits.reshape(-1, logits.shape[-1])
    if logits.shape[0] < len(ids) - 1:
        raise ValueError(f"logits too short: {tuple(logits.shape)} for ids={len(ids)}")
    targets = torch.tensor(ids[1:], dtype=torch.long, device=logits.device)
    pred_logits = logits[: len(ids) - 1]
    chunks: list[torch.Tensor] = []
    for start in range(0, targets.numel(), max(1, logit_chunk_size)):
        end = min(targets.numel(), start + max(1, logit_chunk_size))
        sub = pred_logits[start:end].float()
        tgt = targets[start:end]
        nll = F.cross_entropy(sub, tgt, reduction="none") / math.log(2.0)
        chunks.append(nll.detach().cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0)


def score_model(
    args: argparse.Namespace,
    scorer: Any,
    encoded: list[tuple[TextItem, list[int]]],
    bins: list[tuple[int, int, str]],
) -> ScoreAggregate:
    aggregate = ScoreAggregate(name=scorer.name)
    assert aggregate.by_position is not None
    assert aggregate.per_text is not None
    started = time.perf_counter()
    for idx, (item, ids) in enumerate(encoded, 1):
        if len(ids) < 2:
            continue
        logits = scorer.forward_all_logits(ids)
        bits = nll_bits_for_ids(logits, ids, logit_chunk_size=args.logit_chunk_size)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        token_count = int(bits.numel())
        total_bits = float(bits.sum().item())
        aggregate.total_bits += total_bits
        aggregate.total_tokens += token_count
        aggregate.per_text.append(
            {
                "index": item.index,
                "line_no": item.metadata.get("line_no"),
                "unique_id": item.metadata.get("unique_id"),
                "tokens_scored": token_count,
                "bits": total_bits,
                "bits_per_token": total_bits / max(token_count, 1),
                "subject": item.metadata.get("subject", ""),
                "level": item.metadata.get("level", ""),
            }
        )
        for pos0, value in enumerate(bits.tolist(), start=1):
            label = position_bin(pos0, bins)
            bucket = aggregate.by_position.setdefault(label, {"bits": 0.0, "tokens": 0})
            bucket["bits"] += float(value)
            bucket["tokens"] += 1
        if args.progress_every > 0 and idx % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            bpt = aggregate.bits_per_token
            bpt_text = f"{bpt:.6f}" if bpt is not None else "n/a"
            print(
                f"compression score {scorer.name} texts={idx}/{len(encoded)} tokens={aggregate.total_tokens} "
                f"bits_per_token={bpt_text} elapsed_s={elapsed:.3f}",
                flush=True,
            )
    return aggregate


def compact_score(score: ScoreAggregate) -> dict[str, Any]:
    assert score.by_position is not None
    return {
        "name": score.name,
        "total_bits": score.total_bits,
        "total_tokens": score.total_tokens,
        "bits_per_token": score.bits_per_token,
        "by_position": {
            label: {
                "bits": bucket["bits"],
                "tokens": int(bucket["tokens"]),
                "bits_per_token": bucket["bits"] / max(int(bucket["tokens"]), 1),
            }
            for label, bucket in score.by_position.items()
        },
        "per_text": score.per_text,
    }


def compare_scores(reference: ScoreAggregate, candidate: ScoreAggregate) -> dict[str, Any]:
    ref_bpt = reference.bits_per_token
    cand_bpt = candidate.bits_per_token
    by_pos: dict[str, Any] = {}
    assert reference.by_position is not None and candidate.by_position is not None
    for label in reference.by_position:
        ref = reference.by_position.get(label, {"bits": 0.0, "tokens": 0})
        cand = candidate.by_position.get(label, {"bits": 0.0, "tokens": 0})
        ref_tokens = int(ref.get("tokens", 0))
        cand_tokens = int(cand.get("tokens", 0))
        ref_pos_bpt = float(ref.get("bits", 0.0)) / max(ref_tokens, 1)
        cand_pos_bpt = float(cand.get("bits", 0.0)) / max(cand_tokens, 1)
        by_pos[label] = {
            "reference_bits_per_token": ref_pos_bpt,
            "candidate_bits_per_token": cand_pos_bpt,
            "candidate_over_reference_bits_ratio": cand_pos_bpt / ref_pos_bpt if ref_pos_bpt > 0 else None,
            "tokens": min(ref_tokens, cand_tokens),
        }
    per_text_cmp: list[dict[str, Any]] = []
    ref_rows = {int(row["index"]): row for row in (reference.per_text or [])}
    cand_rows = {int(row["index"]): row for row in (candidate.per_text or [])}
    for key in sorted(set(ref_rows) & set(cand_rows)):
        ref = ref_rows[key]
        cand = cand_rows[key]
        ref_text_bpt = float(ref["bits_per_token"])
        cand_text_bpt = float(cand["bits_per_token"])
        per_text_cmp.append(
            {
                "index": key,
                "line_no": ref.get("line_no"),
                "unique_id": ref.get("unique_id"),
                "tokens": min(int(ref.get("tokens_scored", 0)), int(cand.get("tokens_scored", 0))),
                "reference_bits_per_token": ref_text_bpt,
                "candidate_bits_per_token": cand_text_bpt,
                "candidate_over_reference_bits_ratio": cand_text_bpt / ref_text_bpt if ref_text_bpt > 0 else None,
                "candidate_minus_reference_bits_per_token": cand_text_bpt - ref_text_bpt,
                "subject": ref.get("subject", ""),
                "level": ref.get("level", ""),
            }
        )
    worst = sorted(
        per_text_cmp,
        key=lambda row: row["candidate_over_reference_bits_ratio"] if row["candidate_over_reference_bits_ratio"] is not None else -1,
        reverse=True,
    )[:20]
    return {
        "reference_bits_per_token": ref_bpt,
        "candidate_bits_per_token": cand_bpt,
        "candidate_over_reference_bits_ratio": (cand_bpt / ref_bpt if ref_bpt and cand_bpt is not None else None),
        "candidate_minus_reference_bits_per_token": (cand_bpt - ref_bpt if ref_bpt is not None and cand_bpt is not None else None),
        "tokens": min(reference.total_tokens, candidate.total_tokens),
        "by_position": by_pos,
        "worst_texts_by_ratio": worst,
    }


def evaluate_quality_gates(
    comparison: dict[str, Any],
    *,
    max_bits_ratio: float | None,
    min_scored_tokens: int,
) -> dict[str, Any]:
    """Evaluate explicit, machine-readable quality acceptance gates."""

    ratio = comparison.get("candidate_over_reference_bits_ratio")
    tokens = int(comparison.get("tokens") or 0)
    finite_ratio = ratio is not None and math.isfinite(float(ratio))
    ratio_pass = bool(
        finite_ratio
        and (max_bits_ratio is None or float(ratio) <= float(max_bits_ratio))
    )
    token_pass = tokens >= int(min_scored_tokens)
    return {
        "max_candidate_over_reference_bits_ratio": {
            "required": max_bits_ratio is not None,
            "maximum": max_bits_ratio,
            "actual": ratio,
            "passed": ratio_pass,
        },
        "minimum_scored_tokens": {
            "required": True,
            "minimum": int(min_scored_tokens),
            "actual": tokens,
            "passed": token_pass,
        },
        "overall_pass": bool(ratio_pass and token_pass),
    }


def write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    cmp = report["comparison"]
    lines: list[str] = []
    lines.append("# Uncheatable logit compression alignment")
    lines.append("")
    lines.append("Teacher-forced external-token NLL benchmark. Lower bits/token is better; candidate/reference ratio near `1.0` means the candidate compresses the fixed external text like the reference.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| reference bits/token | `{_fmt(cmp.get('reference_bits_per_token'))}` |")
    lines.append(f"| candidate bits/token | `{_fmt(cmp.get('candidate_bits_per_token'))}` |")
    lines.append(f"| candidate/reference bits ratio | `{_fmt(cmp.get('candidate_over_reference_bits_ratio'))}` |")
    lines.append(f"| candidate-reference bits/token | `{_fmt(cmp.get('candidate_minus_reference_bits_per_token'))}` |")
    lines.append(f"| tokens scored | `{cmp.get('tokens')}` |")
    gates = report.get("gates", {})
    lines.append(f"| quality gate | `{'PASS' if gates.get('overall_pass') else 'FAIL'}` |")
    lines.append("")
    lines.append("## Compression ratio vs token position")
    lines.append("")
    lines.append("| token position bin | reference bpt | candidate bpt | cand/ref ratio | tokens |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, row in cmp.get("by_position", {}).items():
        lines.append(
            f"| `{label}` | `{_fmt(row.get('reference_bits_per_token'))}` | `{_fmt(row.get('candidate_bits_per_token'))}` | "
            f"`{_fmt(row.get('candidate_over_reference_bits_ratio'))}` | `{row.get('tokens')}` |"
        )
    lines.append("")
    lines.append("## Worst texts by candidate/reference ratio")
    lines.append("")
    lines.append("| index | unique_id | tokens | ref bpt | cand bpt | ratio | delta bpt |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for row in cmp.get("worst_texts_by_ratio", [])[:20]:
        lines.append(
            f"| `{row.get('index')}` | `{row.get('unique_id')}` | `{row.get('tokens')}` | "
            f"`{_fmt(row.get('reference_bits_per_token'))}` | `{_fmt(row.get('candidate_bits_per_token'))}` | "
            f"`{_fmt(row.get('candidate_over_reference_bits_ratio'))}` | `{_fmt(row.get('candidate_minus_reference_bits_per_token'))}` |"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)


def add_scorer_args(ap: argparse.ArgumentParser, prefix: str) -> None:
    label = prefix.replace("_", "-")
    ap.add_argument(f"--{label}-kind", choices=("hf", "albatross"), default="hf")
    ap.add_argument(f"--{label}-hf-dir", default="")
    ap.add_argument(f"--{label}-dtype", choices=sorted(DTYPES), default="fp16")
    ap.add_argument(f"--{label}-fast-token-backend", default="native_graph")
    ap.add_argument(
        f"--{label}-quantization",
        choices=(
            "none",
            "bnb8",
            "bnb4",
            "bnb8_a8w8_head",
            "a8w8",
            "mm8",
            "mm4",
            "torchao_w8",
            "torchao_w4",
        ),
        default="none",
    )
    ap.add_argument(f"--{label}-quant-min-params", type=int, default=8_000_000)
    ap.add_argument(f"--{label}-quant-group-size", type=int, default=128)
    ap.add_argument(
        f"--{label}-quant-policy",
        choices=("memory", "decode_hot", "prefill_hot", "balanced", "speed", "dense"),
        default="speed",
    )
    ap.add_argument(f"--{label}-albatross-dir", default="")
    ap.add_argument(f"--{label}-albatross-model", default="")
    ap.add_argument(f"--{label}-albatross-module", default="rwkv7_fast_v3a")
    ap.add_argument(f"--{label}-albatross-wkv", default="fp32io16")
    ap.add_argument(f"--{label}-albatross-emb", default="cpu")
    ap.add_argument(f"--{label}-albatross-batched-rkv", default="off")
    ap.add_argument(f"--{label}-albatross-cmix-sparse", default="no-fc")
    ap.add_argument(f"--{label}-albatross-lowrank-weight", default="both")
    ap.add_argument(f"--{label}-albatross-orig-linear-groups", default="att_c2c,ffn_key,head")
    ap.add_argument(f"--{label}-chdir-albatross", action="store_true")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="JSONL dataset with external text or MATH500 problem/answer fields")
    ap.add_argument("--tokenizer-dir", default="", help="HF tokenizer directory. Defaults to a HF reference/candidate dir.")
    ap.add_argument("--text-field", default="", help="Use one JSONL field as the external text")
    ap.add_argument("--text-template", default=DEFAULT_TEXT_TEMPLATE, help="Python format template for JSONL rows")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens-per-text", type=int, default=1024)
    ap.add_argument("--add-bos", action="store_true")
    ap.add_argument("--bos-token-id", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--position-bins", default="1,8,16,32,64,128,256,512,1024,2048,4096,8192")
    ap.add_argument("--logit-chunk-size", type=int, default=128)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument(
        "--max-candidate-over-reference-bits-ratio",
        type=float,
        default=None,
        help="Fail when candidate compression bits/token exceeds this ratio versus reference.",
    )
    ap.add_argument("--min-scored-tokens", type=int, default=1)
    ap.add_argument("--out-json", default="compression_alignment.json")
    ap.add_argument("--out-md", default="compression_alignment.md")
    add_scorer_args(ap, "reference")
    add_scorer_args(ap, "candidate")
    args = ap.parse_args()
    if args.max_candidate_over_reference_bits_ratio is not None and args.max_candidate_over_reference_bits_ratio <= 0:
        ap.error("--max-candidate-over-reference-bits-ratio must be positive")
    if args.min_scored_tokens < 1:
        ap.error("--min-scored-tokens must be at least 1")

    started = time.perf_counter()
    texts = load_texts(args)
    tokenizer = load_common_tokenizer(args)
    encoded = [
        (item, encode_text(tokenizer, item.text, add_bos=args.add_bos, bos_token_id=args.bos_token_id, max_tokens=args.max_tokens_per_text))
        for item in texts
    ]
    encoded = [(item, ids) for item, ids in encoded if len(ids) >= 2]
    if not encoded:
        raise RuntimeError("no texts had at least two tokens after encoding")
    bins = parse_position_bins(args.position_bins)

    print(f"compression benchmark texts={len(encoded)} device={args.device}", flush=True)
    reference = build_scorer(args, "reference")
    ref_score = score_model(args, reference, encoded, bins)
    cuda_sync(args.device)
    reference.close()
    del reference
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    candidate = build_scorer(args, "candidate")
    cand_score = score_model(args, candidate, encoded, bins)
    cuda_sync(args.device)
    candidate.close()
    del candidate
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    comparison = compare_scores(ref_score, cand_score)
    gates = evaluate_quality_gates(
        comparison,
        max_bits_ratio=args.max_candidate_over_reference_bits_ratio,
        min_scored_tokens=args.min_scored_tokens,
    )
    report = {
        "axis": "uncheatable_logit_compression_alignment",
        "status": "pass" if gates["overall_pass"] else "fail",
        "dataset": args.dataset,
        "num_texts": len(encoded),
        "max_tokens_per_text": args.max_tokens_per_text,
        "add_bos": args.add_bos,
        "text_field": args.text_field,
        "text_template": args.text_template if not args.text_field else None,
        "reference": compact_score(ref_score),
        "candidate": compact_score(cand_score),
        "comparison": comparison,
        "gates": gates,
        "elapsed_sec": time.perf_counter() - started,
        "config": vars(args),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(args.out_md, report)
    print("LOGIT_COMPRESSION_ALIGNMENT_RESULT " + json.dumps(report["comparison"], ensure_ascii=False), flush=True)
    return 0 if gates["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
