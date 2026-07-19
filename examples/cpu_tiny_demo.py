#!/usr/bin/env python3
"""Run a download-free RWKV-7 Native inference and training demo on CPU."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Allow the documented source-checkout command to run before an editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


@contextmanager
def native_eager_backend():
    key = "RWKV7_NATIVE_MODEL_BACKEND"
    previous = os.environ.get(key)
    os.environ[key] = "eager"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "infer", "train"), default="all")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional directory that keeps the trained tiny checkpoint.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("steps", "batch_size", "length", "max_new_tokens", "threads"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.length < 2:
        raise ValueError("--length must be at least 2")


def build_tiny_model(seed: int) -> NativeRWKV7ForCausalLM:
    torch.manual_seed(seed)
    config = NativeRWKV7Config(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        head_dim=4,
        intermediate_size=32,
        decay_low_rank_dim=4,
        gate_low_rank_dim=4,
        a_low_rank_dim=4,
        v_low_rank_dim=4,
        use_cache=True,
    )
    return NativeRWKV7ForCausalLM(config).to(device="cpu", dtype=torch.float32)


def make_training_batch(model: NativeRWKV7ForCausalLM, batch_size: int, length: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.long).view(1, -1)
    offsets = torch.arange(batch_size, dtype=torch.long).view(-1, 1) * 3
    return (positions + offsets + 1) % int(model.config.vocab_size)


def infer_tokens(
    model: NativeRWKV7ForCausalLM,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    model.eval()
    model.config.use_cache = True
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
        )
    elapsed = time.perf_counter() - started
    generated = output[0, prompt.shape[1] :].tolist()
    if len(generated) != max_new_tokens:
        raise AssertionError(
            f"expected {max_new_tokens} generated tokens, got {len(generated)}"
        )
    return {
        "prompt_token_ids": prompt[0].tolist(),
        "generated_token_ids": generated,
        "elapsed_s": round(elapsed, 6),
    }


def _parameter_snapshot(model: NativeRWKV7ForCausalLM) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def train_tiny_model(
    model: NativeRWKV7ForCausalLM,
    *,
    steps: int,
    batch_size: int,
    length: int,
    learning_rate: float,
) -> dict[str, Any]:
    model.train()
    model.config.use_cache = False
    input_ids = make_training_batch(model, batch_size, length)
    labels = input_ids.clone()
    before = _parameter_snapshot(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    with torch.no_grad():
        initial_loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
    if initial_loss is None or not bool(torch.isfinite(initial_loss).item()):
        raise AssertionError(f"non-finite initial loss: {initial_loss}")

    losses: list[float] = []
    max_grad_l1 = 0.0
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = output.loss
        if loss is None or not bool(torch.isfinite(loss.detach()).item()):
            raise AssertionError(f"non-finite training loss: {loss}")
        loss.backward()
        grad_l1 = sum(
            float(parameter.grad.detach().abs().sum())
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        if grad_l1 <= 0:
            raise AssertionError("expected a non-zero CPU training gradient")
        max_grad_l1 = max(max_grad_l1, grad_l1)
        optimizer.step()
        losses.append(float(loss.detach()))
    elapsed = time.perf_counter() - started

    model.eval()
    with torch.no_grad():
        final_loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
    if final_loss is None or not bool(torch.isfinite(final_loss).item()):
        raise AssertionError(f"non-finite final loss: {final_loss}")

    changed_l1 = sum(
        float((parameter.detach() - before[name]).abs().sum())
        for name, parameter in model.named_parameters()
        if name in before
    )
    if changed_l1 <= 0:
        raise AssertionError("expected CPU training to update model parameters")
    if not float(final_loss) < float(initial_loss):
        raise AssertionError(
            f"expected final loss < initial loss, got {float(final_loss)} >= {float(initial_loss)}"
        )

    return {
        "steps": steps,
        "batch_size": batch_size,
        "length": length,
        "initial_loss": round(float(initial_loss), 6),
        "last_step_loss": round(losses[-1], 6),
        "final_loss": round(float(final_loss), 6),
        "max_grad_l1": round(max_grad_l1, 6),
        "parameter_changed_l1": round(changed_l1, 6),
        "elapsed_s": round(elapsed, 6),
    }


def save_reload_check(
    model: NativeRWKV7ForCausalLM,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    probe = torch.tensor([[2, 4, 6, 8]], dtype=torch.long)
    with torch.inference_mode():
        expected = model(input_ids=probe, use_cache=False).logits
    model.save_pretrained(output_dir, safe_serialization=True)
    reloaded = NativeRWKV7ForCausalLM.from_pretrained(
        output_dir,
        dtype=torch.float32,
    ).eval()
    with torch.inference_mode():
        actual = reloaded(input_ids=probe, use_cache=False).logits
    max_abs = float((actual - expected).abs().max())
    if max_abs != 0.0:
        raise AssertionError(f"save/reload logits changed: max_abs={max_abs}")
    return {
        "checkpoint_dir": str(output_dir.resolve()),
        "max_abs": max_abs,
        "files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    torch.set_num_threads(args.threads)
    with native_eager_backend():
        model = build_tiny_model(args.seed)
        result: dict[str, Any] = {
            "status": "pass",
            "backend": "native_eager",
            "device": "cpu",
            "dtype": "float32",
            "torch": torch.__version__,
            "threads": args.threads,
            "seed": args.seed,
            "mode": args.mode,
        }

        if args.mode in {"all", "infer"}:
            result["inference_before_training"] = infer_tokens(
                model,
                max_new_tokens=args.max_new_tokens,
            )
            print("CPU INFERENCE PASS")

        if args.mode in {"all", "train"}:
            result["training"] = train_tiny_model(
                model,
                steps=args.steps,
                batch_size=args.batch_size,
                length=args.length,
                learning_rate=args.learning_rate,
            )
            print("CPU TRAINING PASS")
            result["inference_after_training"] = infer_tokens(
                model,
                max_new_tokens=args.max_new_tokens,
            )

        if args.output_dir:
            result["save_reload"] = save_reload_check(model, Path(args.output_dir))
        else:
            with tempfile.TemporaryDirectory(prefix="rwkv7-cpu-tiny-") as tmp:
                result["save_reload"] = save_reload_check(model, Path(tmp))
                result["save_reload"]["checkpoint_dir"] = "temporary"
        print("CPU SAVE/RELOAD PASS")
        return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_demo(args)
    print("CPU_DEMO_RESULT=" + json.dumps(result, ensure_ascii=False, sort_keys=True))
    print("CPU DEMO PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
