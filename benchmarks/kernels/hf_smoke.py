#!/usr/bin/env python3
"""Exercise package-free HF loading, generation, reload, and train fallback."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import platform
from importlib import metadata
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="label=path")
    parser.add_argument("--implementation", choices=("graph", "triton"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reload-dir", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_models(values: list[str]) -> list[tuple[str, Path]]:
    rows = []
    for value in values:
        label, separator, path = value.partition("=")
        if not separator:
            raise ValueError("--model must be label=path")
        rows.append((label, Path(path).expanduser().resolve()))
    return rows


def dynamic_route(model) -> dict | None:
    module_name = model.__class__.__module__.rsplit(".", 1)[0] + ".ops_rwkv7"
    module = importlib.import_module(module_name)
    return module.get_last_recurrent_route()


def main() -> int:
    args = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    os.environ["RWKV7_BACKEND"] = "optimized"
    os.environ["RWKV7_KERNEL_IMPL"] = args.implementation
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    models = parse_models(args.model)

    report = {
        "schema": "rwkv7-hf-optional-kernel-smoke-v1",
        "code_sha": args.code_sha,
        "implementation": args.implementation,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "transformers": package_version("transformers"),
            "triton": package_version("triton"),
            "rwkv7_hf": package_version("rwkv7-hf"),
            "rwkv7_kernels": package_version("rwkv7-kernels"),
        },
        "models": {},
    }

    for index, (label, path) in enumerate(models):
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        encoded = tokenizer("RWKV", return_tensors="pt")["input_ids"].to(device)
        encoded = encoded[:, : min(8, encoded.shape[1])]

        backbone = AutoModel.from_pretrained(
            path, trust_remote_code=True, dtype=torch.float16
        ).to(device).eval()
        with torch.inference_mode():
            hidden = backbone(input_ids=encoded, use_cache=True)
        backbone_route = dynamic_route(backbone)
        backbone_shape = list(hidden.last_hidden_state.shape)
        del hidden, backbone
        gc.collect()
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, dtype=torch.float16
        ).to(device).eval()
        with torch.inference_mode():
            forward = model(input_ids=encoded, use_cache=True)
            greedy = model.generate(
                encoded,
                max_new_tokens=4,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
            )
            beam = model.generate(
                encoded,
                max_new_tokens=2,
                num_beams=2,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
            )
        causal_route = dynamic_route(model)
        expected_impl = {
            "graph": "torch-cuda-graph-reference-v1",
            "triton": "native-triton-rank1-scan-v1",
        }[args.implementation]
        route_passed = bool(
            causal_route
            and causal_route["selected"] == "optimized"
            and causal_route["implementation"] == expected_impl
        )
        finite = bool(torch.isfinite(forward.logits).all())
        row = {
            "path": str(path),
            "config_model_type": config.model_type,
            "weight_sha256": sha256(path / "model.safetensors"),
            "tokenizer_class": tokenizer.__class__.__name__,
            "backbone_shape": backbone_shape,
            "backbone_route": backbone_route,
            "causal_route": causal_route,
            "logits_finite": finite,
            "greedy_shape": list(greedy.shape),
            "beam_shape": list(beam.shape),
            "passed": bool(
                config.model_type == "rwkv7"
                and finite
                and route_passed
                and greedy.shape[1] == encoded.shape[1] + 4
                and beam.shape[1] == encoded.shape[1] + 2
            ),
        }

        if index == 0:
            args.reload_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.reload_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.reload_dir)
            reloaded = AutoModelForCausalLM.from_pretrained(
                args.reload_dir, trust_remote_code=True, dtype=torch.float16
            ).to(device).eval()
            with torch.inference_mode():
                reload_logits = reloaded(input_ids=encoded, use_cache=False).logits
                original_logits = model(input_ids=encoded, use_cache=False).logits
            row["save_reload_equal"] = bool(
                torch.equal(reload_logits, original_logits)
            )
            row["passed"] = row["passed"] and row["save_reload_equal"]
            del reloaded, reload_logits, original_logits

        del forward, greedy, beam, model
        gc.collect()
        torch.cuda.empty_cache()
        report["models"][label] = row

    # Model-level training must not enter the inference-only optional kernel.
    os.environ["RWKV7_BACKEND"] = "auto"
    train_label, train_path = models[0]
    train_model = AutoModelForCausalLM.from_pretrained(
        train_path, trust_remote_code=True, dtype=torch.float32
    ).to(device).train()
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    ids = torch.randint(
        1, int(train_model.config.vocab_size), (1, 8), generator=generator, device=device
    )
    output = train_model(input_ids=ids, labels=ids, use_cache=False)
    output.loss.backward()
    route = dynamic_route(train_model)
    gradients = [
        parameter.grad
        for parameter in train_model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    train_passed = bool(
        torch.isfinite(output.loss)
        and gradients
        and all(torch.isfinite(gradient).all() for gradient in gradients)
        and any(torch.count_nonzero(gradient) > 0 for gradient in gradients)
        and route
        and route["selected"] == "reference"
    )
    report["training_fallback"] = {
        "model": train_label,
        "loss": float(output.loss.detach()),
        "gradient_tensors": len(gradients),
        "route": route,
        "passed": train_passed,
    }
    report["passed"] = bool(
        all(row["passed"] for row in report["models"].values()) and train_passed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
