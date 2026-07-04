#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))

from common import (  # noqa: E402
    environment,
    git_revision,
    model_fingerprint,
    sha256_file,
    write_bundle,
)


EXPECTED_FLA_COMMIT = "80e494f6c588e091fc8316b612870df29375c5b8"


def reference_backend_context():
    """Force the clean side to PyTorch when an optional backend is installed.

    The v0.9 reference package has no dispatcher, while the optional v0.10
    package exposes ``use_rwkv7_backend``. A no-op context keeps this harness
    shared by both lines. An inherited ``RWKV7_BACKEND=optimized`` must not
    turn clean-vs-FLA parity into optimized-vs-FLA or break gradient cases
    because the optional protocol is inference-only.
    """

    try:
        from rwkv7_hf.kernel_bridge import use_rwkv7_backend
    except ModuleNotFoundError as exc:
        if exc.name != "rwkv7_hf.kernel_bridge":
            raise
        return nullcontext()
    return use_rwkv7_backend("reference")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Record a non-blocking numerical diagnostic between the readable "
            "RWKV-7 reference model and a pinned optimized FLA backend"
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", default="1,4")
    parser.add_argument("--lengths", default="1,17,128")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-unverified-fla", action="store_true")
    parser.add_argument(
        "--require-thresholds",
        action="store_true",
        help="Return non-zero when diagnostic thresholds are missed",
    )
    parser.add_argument(
        "--fla-source",
        type=Path,
        help="Pinned FLA source checkout/archive directory to prepend to sys.path",
    )
    parser.add_argument(
        "--fla-archive",
        type=Path,
        help="Optional downloaded commit archive whose SHA256 is recorded",
    )
    parser.add_argument(
        "--code-sha",
        help="Source commit for rsync deployments that intentionally omit .git",
    )
    return parser.parse_args()


def direct_url_commit(distribution: str) -> str | None:
    try:
        dist = importlib.metadata.distribution(distribution)
        text = dist.read_text("direct_url.json")
        if text:
            return json.loads(text).get("vcs_info", {}).get("commit_id")
    except Exception:
        pass
    return None


def prepare_fla_source(source: Path | None) -> tuple[str | None, str | None]:
    """Select and verify the exact FLA source used by this process."""

    if source is None:
        return direct_url_commit("flash-linear-attention"), None
    source = source.expanduser().resolve()
    if not (source / "fla" / "__init__.py").is_file():
        raise SystemExit(f"not an FLA source tree: {source}")
    marker = source / ".fla-upstream-commit"
    if (source / ".git").exists():
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    elif marker.is_file():
        commit = marker.read_text(encoding="utf-8").strip()
    else:
        commit = None
    sys.path.insert(0, str(source))
    return commit, str(source)


def metrics(left: torch.Tensor, right: torch.Tensor) -> dict:
    a = left.float().reshape(-1)
    b = right.float().reshape(-1)
    # A single FP32 reduction over full-vocabulary logits can accumulate enough
    # error to report an impossible cosine greater than one. Keep the tensors
    # in FP32, but accumulate the three scalar reductions in FP64.
    dot = (a * b).sum(dtype=torch.float64)
    left_norm = (a * a).sum(dtype=torch.float64).sqrt()
    right_norm = (b * b).sum(dtype=torch.float64).sqrt()
    denominator = left_norm * right_norm
    if denominator == 0:
        cosine = 1.0 if torch.equal(a, b) else 0.0
    else:
        cosine = float((dot / denominator).clamp(-1.0, 1.0))
    return {
        "cosine": cosine,
        "max_abs": float((a - b).abs().max()),
        "mean_abs": float((a - b).abs().mean()),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "argmax_same": bool(torch.equal(left.argmax(-1), right.argmax(-1))),
        "fp32_allclose": bool(torch.allclose(a, b, rtol=1e-4, atol=1e-5)),
    }


def cached_teacher(model, prompt_ids, continuation_ids):
    """Prefill once, then teacher-force the continuation through the cache."""

    output = model(input_ids=prompt_ids, use_cache=True)
    cache = output.past_key_values
    pieces = [output.logits.detach().cpu()]
    for token_idx in range(continuation_ids.shape[1]):
        output = model(
            input_ids=continuation_ids[:, token_idx : token_idx + 1],
            past_key_values=cache,
            use_cache=True,
        )
        cache = output.past_key_values
        pieces.append(output.logits.detach().cpu())
    return torch.cat(pieces, dim=1), cache


def greedy(model, prefix, count: int):
    output = model(input_ids=prefix, use_cache=True)
    cache = output.past_key_values
    token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [token.cpu()]
    for _ in range(count - 1):
        output = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(token.cpu())
    return torch.cat(generated, dim=1)


def clean_states(cache):
    return [value.detach().float().cpu() for value in cache.recurrent_state]


def fla_states(cache):
    states = []
    for layer_idx in range(len(cache)):
        value = cache[layer_idx].get("recurrent_state")
        if isinstance(value, (tuple, list)):
            value = value[0]
        states.append(value.detach().float().cpu())
    return states


def state_metrics(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError(
            f"cache layer count mismatch: reference={len(reference)}, "
            f"candidate={len(candidate)}"
        )
    rows = []
    for clean, fla in zip(reference, candidate):
        direct = (
            (clean - fla).abs().max()
            if clean.shape == fla.shape
            else torch.tensor(float("inf"))
        )
        transposed = (
            (clean - fla.transpose(-1, -2)).abs().max()
            if clean.shape == fla.transpose(-1, -2).shape
            else torch.tensor(float("inf"))
        )
        if transposed < direct:
            row = metrics(clean, fla.transpose(-1, -2))
            row["layout"] = "fla_transposed"
        else:
            row = metrics(clean, fla)
            row["layout"] = "direct"
        rows.append(row)
    return rows


def thresholds(dtype_name: str, comparison: dict, *, logits: bool) -> bool:
    if not comparison["finite"]:
        return False
    if dtype_name == "fp32":
        return comparison["fp32_allclose"]
    if comparison["cosine"] < 0.9999:
        return False
    # The plan's 0.15 max-absolute bound is specifically an FP16 logit gate.
    # BF16 and recurrent states use the cosine/finite requirements.
    return not (dtype_name == "fp16" and logits and comparison["max_abs"] > 0.15)


def operator_parity(dtype: torch.dtype, dtype_name: str, device: torch.device, batches, lengths):
    """Compare the public PyTorch operator, final state, and backward gradients."""

    from fla.ops.rwkv7 import chunk_rwkv7
    from rwkv7_hf.ops_rwkv7 import rwkv7_recurrent

    rows = {}
    for batch in batches:
        for length in lengths:
            name = f"b{batch}_t{length}"
            generator = torch.Generator(device=device).manual_seed(
                10_000 + batch * 1000 + length
            )
            shapes = (batch, length, 2, 64)
            base = {
                "r": torch.randn(shapes, generator=generator, device=device, dtype=dtype) * 0.1,
                "w": -(torch.rand(shapes, generator=generator, device=device, dtype=dtype) * 0.5 + 0.1),
                "k": torch.randn(shapes, generator=generator, device=device, dtype=dtype) * 0.1,
                "v": torch.randn(shapes, generator=generator, device=device, dtype=dtype) * 0.1,
                "a": torch.randn(shapes, generator=generator, device=device, dtype=dtype) * 0.1,
                "b": torch.randn(shapes, generator=generator, device=device, dtype=dtype) * 0.1,
                "state": torch.randn(
                    (batch, 2, 64, 64), generator=generator, device=device, dtype=dtype
                ) * 0.01,
            }

            clean_inputs = {
                key: value.detach().clone().requires_grad_(True)
                for key, value in base.items()
            }
            clean_out, clean_state = rwkv7_recurrent(
                clean_inputs["r"],
                clean_inputs["w"].exp(),
                clean_inputs["k"],
                clean_inputs["v"],
                clean_inputs["a"],
                clean_inputs["b"],
                clean_inputs["state"],
            )
            clean_loss = clean_out.float().square().mean() + clean_state.float().square().mean()
            clean_loss.backward()
            clean_grads = {key: value.grad.detach().cpu() for key, value in clean_inputs.items()}

            fla_inputs = {
                key: value.detach().clone().requires_grad_(True)
                for key, value in base.items()
            }
            fla_out, fla_state = chunk_rwkv7(
                r=fla_inputs["r"],
                w=fla_inputs["w"],
                k=fla_inputs["k"],
                v=fla_inputs["v"],
                a=fla_inputs["a"],
                b=fla_inputs["b"],
                initial_state=fla_inputs["state"],
                output_final_state=True,
            )
            fla_loss = fla_out.float().square().mean() + fla_state.float().square().mean()
            fla_loss.backward()

            output_row = metrics(clean_out.detach().cpu(), fla_out.detach().cpu())
            state_row = metrics(clean_state.detach().cpu(), fla_state.detach().cpu())
            gradient_rows = {
                key: metrics(clean_grads[key], value.grad.detach().cpu())
                for key, value in fla_inputs.items()
            }
            if dtype_name == "fp32":
                passed = torch.allclose(clean_out, fla_out, rtol=1e-4, atol=1e-5)
                passed = passed and torch.allclose(clean_state, fla_state, rtol=1e-4, atol=1e-5)
                passed = passed and all(
                    torch.allclose(
                        clean_grads[key].to(device), value.grad,
                        rtol=5e-4, atol=5e-5,
                    )
                    for key, value in fla_inputs.items()
                )
            else:
                # FP16/BF16 gradients can be tiny enough that cosine is
                # meaningless even when every element differs by <1e-6. The
                # strict rtol/atol gradient gate is therefore run in FP32;
                # low precision requires finite gradients and output/state
                # cosine parity.
                passed = all(
                    row["finite"] and row["cosine"] >= 0.9999
                    for row in (output_row, state_row)
                ) and all(row["finite"] for row in gradient_rows.values())
            rows[name] = {
                "passed": bool(passed),
                "output": output_row,
                "state": state_row,
                "gradients": gradient_rows,
            }
            del base, clean_inputs, fla_inputs, clean_out, clean_state, fla_out, fla_state
    return rows


def main():
    args = parse_args()
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    device = torch.device(args.device)
    batches = [int(value) for value in args.batches.split(",")]
    lengths = [int(value) for value in args.lengths.split(",")]

    installed_commit, fla_source = prepare_fla_source(args.fla_source)
    if installed_commit != EXPECTED_FLA_COMMIT and not args.allow_unverified_fla:
        raise SystemExit(
            "FLA must be installed from commit "
            f"{EXPECTED_FLA_COMMIT}; detected {installed_commit!r}. "
            "Use --allow-unverified-fla only for an exploratory benchmark."
        )

    torch.manual_seed(args.seed)
    cases = {
        f"b{batch}_t{length}": torch.randint(
            1, 8192, (batch, length), device=device
        )
        for batch in batches
        for length in lengths
    }
    decode_prompt = torch.randint(1, 8192, (1, 17), device=device)
    decode_continuation = torch.randint(1, 8192, (1, 16), device=device)
    decode_ids = torch.cat((decode_prompt, decode_continuation), dim=1)

    from rwkv7_hf.configuration_rwkv7 import RWKV7Config
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    with reference_backend_context():
        operator = operator_parity(dtype, args.dtype, device, batches, lengths)

        clean_config = RWKV7Config.from_pretrained(args.model)
        clean = RWKV7ForCausalLM.from_pretrained(
            args.model, config=clean_config, dtype=dtype
        ).to(device).eval()
        clean_logits = {}
        with torch.inference_mode():
            for name, ids in cases.items():
                clean_logits[name] = clean(input_ids=ids, use_cache=False).logits.cpu()
            cached_logits, clean_cache = cached_teacher(
                clean, decode_prompt, decode_continuation
            )
            clean_full_decode = clean(input_ids=decode_ids, use_cache=False).logits.cpu()
            clean_state = clean_states(clean_cache)
            clean_tokens = greedy(clean, decode_ids, args.decode_tokens)
        del clean
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config as FLAConfig
    from fla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM as FLAModel

    fla_config = FLAConfig.from_pretrained(args.model)
    fla = FLAModel.from_pretrained(
        args.model, config=fla_config, dtype=dtype
    ).to(device).eval()
    comparisons = {}
    with torch.inference_mode():
        for name, ids in cases.items():
            candidate = fla(input_ids=ids, use_cache=False).logits.cpu()
            comparisons[name] = metrics(clean_logits[name], candidate)
        fla_cached_logits, fla_cache = cached_teacher(
            fla, decode_prompt, decode_continuation
        )
        fla_full_decode = fla(input_ids=decode_ids, use_cache=False).logits.cpu()
        comparisons["cached_teacher_clean_vs_fla"] = metrics(
            cached_logits, fla_cached_logits
        )
        comparisons["clean_cached_vs_clean_full"] = metrics(
            cached_logits, clean_full_decode
        )
        comparisons["fla_cached_vs_fla_full"] = metrics(
            fla_cached_logits, fla_full_decode
        )
        fla_tokens = greedy(fla, decode_ids, args.decode_tokens)
        states = state_metrics(clean_state, fla_states(fla_cache))

    greedy_equal = bool(torch.equal(clean_tokens, fla_tokens))
    logits_passed = all(
        thresholds(args.dtype, row, logits=True) for row in comparisons.values()
    )
    states_passed = all(
        thresholds(args.dtype, row, logits=False) for row in states
    )
    operator_passed = all(row["passed"] for row in operator.values())
    within_thresholds = (
        logits_passed and states_passed and greedy_equal and operator_passed
    )

    report = {
        "schema_version": 1,
        "status": "within_thresholds" if within_thresholds else "outside_thresholds",
        "blocking": False,
        "code_sha": args.code_sha or git_revision(ROOT),
        "fla_commit": installed_commit or "unverified",
        "fla_source": fla_source,
        "fla_archive_sha256": (
            sha256_file(args.fla_archive.expanduser().resolve())
            if args.fla_archive is not None
            else None
        ),
        "required_fla_commit": EXPECTED_FLA_COMMIT,
        "model": model_fingerprint(args.model),
        "dtype": args.dtype,
        "environment": environment(),
        "comparisons": comparisons,
        "operator": operator,
        "state": states,
        "greedy": {
            "tokens": args.decode_tokens,
            "equal": greedy_equal,
            "clean": clean_tokens.tolist(),
            "fla": fla_tokens.tolist(),
        },
        "gates": {
            "logits": logits_passed,
            "state": states_passed,
            "operator": operator_passed,
            "greedy": greedy_equal,
        },
    }
    name = f"clean-vs-fla-{args.model.name}-{args.dtype}"
    paths = write_bundle(args.output_dir, name, report)
    print(json.dumps({"status": report["status"], "artifacts": [str(p) for p in paths]}))
    raise SystemExit(
        1 if args.require_thresholds and not within_thresholds else 0
    )


if __name__ == "__main__":
    main()
