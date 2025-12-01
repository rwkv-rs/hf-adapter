#!/usr/bin/env python3
"""Validate the production BN/TN contract over real RWKV FFN row counts.

The guarded call enters ``production_bn_tn=True`` and therefore fails closed
inside the CUDA launcher if Marlin's selected CTA output width is not the
declared BN.  The unguarded call uses the same packed weights with only the
physical TN check enabled; exact equality proves that the guard does not alter
the computed result.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from rwkv7_hf.native_quant_marlin import MarlinW4Linear


DEFAULT_ROWS = (
    1,
    2,
    4,
    8,
    16,
    17,
    24,
    32,
    64,
    96,
    128,
    256,
    512,
    1024,
)
DEFAULT_SHAPES = ((4096, 16384), (16384, 4096))


def parse_shape(raw: str) -> tuple[int, int]:
    k, n = (int(value) for value in raw.lower().split("x", 1))
    return k, n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", type=parse_shape, default=DEFAULT_SHAPES)
    parser.add_argument("--rows", nargs="+", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if any(rows <= 0 for rows in args.rows):
        raise SystemExit("all row counts must be positive")

    torch.manual_seed(args.seed)
    records: list[dict[str, object]] = []
    for k, n in args.shapes:
        dense = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=torch.bfloat16)
        packed = MarlinW4Linear(
            dense,
            group_size=128,
            fp32_reduce=False,
            production_bn_tn=True,
            fuse_relu2=n > k,
        )
        del dense
        torch.cuda.empty_cache()
        for rows in args.rows:
            x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
            record: dict[str, object] = {
                "rows": rows,
                "k": k,
                "n": n,
                "declared_plan": packed.effective_bn_tn_plan(rows).as_dict(),
                "device": torch.cuda.get_device_name(),
                "compute_capability": list(torch.cuda.get_device_capability()),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            }
            try:
                guarded = packed(x)
                unguarded = packed._apply_marlin(x, expected_bn_tn=(-1, 8))
                torch.cuda.synchronize()
                wrong_bn_rejected = False
                wrong_bn_error = ""
                try:
                    # BN=64 is emitted by Marlin generally but is never valid
                    # for this exact 5090 FFN contract. It must fail closed.
                    packed._apply_marlin(x, expected_bn_tn=(64, 8))
                    torch.cuda.synchronize()
                except RuntimeError as exc:
                    wrong_bn_rejected = True
                    wrong_bn_error = str(exc).splitlines()[0]
                record.update(
                    status="pass",
                    output_shape=list(guarded.shape),
                    finite=bool(torch.isfinite(guarded).all().item()),
                    exact_vs_unguarded=bool(torch.equal(guarded, unguarded)),
                    max_abs_vs_unguarded=float(
                        (guarded.float() - unguarded.float()).abs().max().item()
                    ),
                    wrong_bn_rejected=wrong_bn_rejected,
                    wrong_bn_error=wrong_bn_error,
                )
                if not wrong_bn_rejected:
                    record.update(status="fail", error="wrong BN was not rejected")
                del guarded, unguarded
            except Exception as exc:
                record.update(status="fail", error=str(exc).splitlines()[0])
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            del x
        del packed
        gc.collect()
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    passed = sum(record["status"] == "pass" for record in records)
    exact = sum(record.get("exact_vs_unguarded") is True for record in records)
    fail_closed = sum(record.get("wrong_bn_rejected") is True for record in records)
    summary = {
        "pass": passed,
        "exact": exact,
        "fail_closed": fail_closed,
        "fail": len(records) - passed,
        "total": len(records),
        "output": str(args.output),
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    return 0 if passed == exact == fail_closed == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
