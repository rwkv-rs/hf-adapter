#!/usr/bin/env python3
"""Run the pinned train_temp shape through the canonical Native HF backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv7_hf.train_temp_alignment import build_train_temp_param_groups
from scripts.run_train_temp_official_recipe import (
    DEFAULT_RECIPE,
    build_official_runtime_env,
    load_recipe,
)


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def ensure_single_process_distributed_env() -> None:
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    os.environ.setdefault("DS_IGNORE_CUDA_DETECTION", "1")


def build_deepspeed_config(recipe: dict[str, Any]) -> dict[str, Any]:
    run = recipe["run"]
    batch_size = int(run["micro_bsz"])
    return {
        "zero_allow_untested_optimizer": True,
        "train_micro_batch_size_per_gpu": batch_size,
        "train_batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": float(run["grad_clip"]),
        "bf16": {"enabled": str(run["precision"]).lower() == "bf16"},
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 200_000_000,
            "reduce_bucket_size": 200_000_000,
            "sub_group_size": 1_000_000_000_000,
        },
        "activation_checkpointing": {
            "partition_activations": False,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": False,
            "synchronize_checkpoint_boundary": False,
        },
    }


def validate_model_config(config: Any, recipe: dict[str, Any]) -> None:
    expected = recipe["model"]
    actual = {
        "model_type": str(config.model_type),
        "n_layer": int(config.num_hidden_layers),
        "n_embd": int(config.hidden_size),
        "effective_dim_ffn": int(config.intermediate_size),
        "head_size": int(config.head_dim),
        "vocab_size": int(config.vocab_size),
    }
    wanted = {
        "model_type": "rwkv7_native",
        "n_layer": int(expected["n_layer"]),
        "n_embd": int(expected["n_embd"]),
        "effective_dim_ffn": int(expected["effective_dim_ffn"]),
        "head_size": int(expected["head_size"]),
        "vocab_size": int(expected["vocab_size"]),
    }
    if actual != wanted:
        raise ValueError(f"Native model does not match pinned recipe: {actual} != {wanted}")


def validate_batch(batch: dict[str, torch.Tensor], recipe: dict[str, Any]) -> None:
    input_ids = batch.get("input_ids")
    targets = batch.get("targets")
    if input_ids is None or targets is None or not torch.equal(input_ids[:, 1:], targets[:, :-1]):
        raise ValueError("batch must contain matching shifted input_ids and targets")
    expected_shape = (
        int(recipe["run"]["micro_bsz"]),
        int(recipe["model"]["ctx_len"]),
    )
    if tuple(input_ids.shape) != expected_shape or tuple(targets.shape) != expected_shape:
        raise ValueError(
            f"batch shape must be {expected_shape}, got {tuple(input_ids.shape)} and {tuple(targets.shape)}"
        )
    vocab_size = int(recipe["model"]["vocab_size"])
    if int(input_ids.min()) < 0 or int(targets.min()) < 0:
        raise ValueError("batch contains negative token ids")
    if int(input_ids.max()) >= vocab_size or int(targets.max()) >= vocab_size:
        raise ValueError("batch token id exceeds pinned vocabulary")


def streaming_model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        value = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
        del value
    return digest.hexdigest()


def progress(stage: str) -> None:
    print(f"[native-train-temp] {stage}", flush=True)


def inspect_deepspeed_gradients(engine: Any) -> tuple[str, int, int]:
    """Inspect gradients before step, including ZeRO-2 partitioned storage."""
    averaged_gradients = getattr(getattr(engine, "optimizer", None), "averaged_gradients", None)
    if isinstance(averaged_gradients, dict):
        gradients = []
        for group in averaged_gradients.values():
            if group is None:
                continue
            if isinstance(group, torch.Tensor):
                gradients.append(group)
            else:
                gradients.extend(gradient for gradient in group if gradient is not None)
        if gradients:
            finite = sum(int(bool(torch.isfinite(gradient).all().item())) for gradient in gradients)
            return "deepspeed_zero_partition", len(gradients), finite

    gradients = [
        parameter.grad
        for parameter in engine.module.parameters()
        if parameter.grad is not None
    ]
    finite = sum(int(bool(torch.isfinite(gradient).all().item())) for gradient in gradients)
    return "parameter_grad", len(gradients), finite


def run(args: argparse.Namespace) -> dict[str, Any]:
    if sys.platform != "linux" or not torch.cuda.is_available():
        raise RuntimeError("Native train_temp recipe requires Linux/WSL and CUDA")
    recipe = load_recipe(args.recipe)
    if int(args.max_steps) != 1:
        raise ValueError("this bounded acceptance runner currently requires --max-steps 1")
    batch = load_file(args.batch)
    validate_batch(batch, recipe)

    runtime_env, build_environment = build_official_runtime_env(
        Path(args.output).resolve().parent / "torch_extensions"
    )
    os.environ.update(runtime_env)
    ensure_single_process_distributed_env()
    progress("runtime environment ready")

    from rwkv7_hf.train_temp_cuda import (
        enable_train_temp_cuda_backend,
        load_train_temp_cuda_extension,
        train_temp_fused_cross_entropy,
    )

    progress("loading CUDA extensions")
    load_train_temp_cuda_extension()
    progress("CUDA extensions ready")

    import deepspeed
    from deepspeed.ops.adam import FusedAdam
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    progress("model weights loaded")
    validate_model_config(model.config, recipe)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    backend_metadata = enable_train_temp_cuda_backend(model)
    model = model.to(device="cuda", dtype=torch.bfloat16)
    progress("Native backend bound and model moved to CUDA")

    run_config = recipe["run"]
    named_parameters = sorted(model.named_parameters())
    groups = build_train_temp_param_groups(
        named_parameters,
        weight_decay=float(run_config["weight_decay"]),
        naming="hf",
        sort_key=lambda name: name,
    )
    first_step_lr = float(run_config["lr_init"]) * 0.01
    for group in groups:
        group["lr"] = first_step_lr * float(group["my_lr_scale"])
    optimizer = FusedAdam(
        groups,
        lr=first_step_lr,
        betas=(float(run_config["beta1"]), float(run_config["beta2"])),
        eps=float(run_config["adam_eps"]),
        weight_decay=0.0,
        bias_correction=True,
        adam_w_mode=True,
        amsgrad=False,
    )
    deepspeed_config = build_deepspeed_config(recipe)
    progress("FusedAdam ready; initializing DeepSpeed ZeRO-2")
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=deepspeed_config,
    )
    progress("DeepSpeed ZeRO-2 ready")

    input_ids = batch["input_ids"].to(device=engine.device, dtype=torch.long)
    targets = batch["targets"].to(device=engine.device, dtype=torch.long)
    before_sha256 = streaming_model_sha256(engine.module)
    progress("pre-step model hash ready")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()

    engine.train()
    outputs = engine(input_ids)
    progress("forward complete")
    loss = train_temp_fused_cross_entropy(outputs.logits, targets)
    engine.backward(loss)
    progress("backward complete")
    gradient_observation, gradient_tensors, finite_gradient_tensors = inspect_deepspeed_gradients(engine)
    engine.step()
    progress("optimizer step complete")
    torch.cuda.synchronize()
    runtime_s = time.perf_counter() - started
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    after_sha256 = streaming_model_sha256(engine.module)
    loss_value = float(loss.detach().float().item())
    status = (
        "pass"
        if math.isfinite(loss_value)
        and gradient_tensors > 0
        and finite_gradient_tensors == gradient_tensors
        and before_sha256 != after_sha256
        else "fail"
    )
    return {
        "schema_version": 1,
        "axis": "train_temp_native_official_recipe",
        "status": status,
        "backend": "native_train_temp_cuda",
        "model": str(Path(args.model).resolve()),
        "batch": str(Path(args.batch).resolve()),
        "recipe": str(Path(args.recipe).resolve()),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "deepspeed_version": deepspeed.__version__,
        "batch_size": int(input_ids.shape[0]),
        "sequence_length": int(input_ids.shape[1]),
        "dtype": "bf16",
        "gradient_checkpointing": True,
        "zero_stage": 2,
        "optimizer": "DeepSpeed FusedAdam",
        "learning_rate": first_step_lr,
        "loss": loss_value,
        "gradient_observation": gradient_observation,
        "gradient_tensors": gradient_tensors,
        "finite_gradient_tensors": finite_gradient_tensors,
        "model_sha256_before": before_sha256,
        "model_sha256_after": after_sha256,
        "model_updated": before_sha256 != after_sha256,
        "peak_memory_mb": peak_memory_mb,
        "runtime_s": runtime_s,
        "backend_metadata": backend_metadata,
        "deepspeed_config": deepspeed_config,
        "build_environment": build_environment,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recipe", default=str(DEFAULT_RECIPE))
    parser.add_argument("--max-steps", type=int, default=1)
    args = parser.parse_args()
    output = Path(args.output)
    try:
        result = run(args)
    except Exception as exc:
        result = {
            "schema_version": 1,
            "axis": "train_temp_native_official_recipe",
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json_atomic(output, result)
        raise
    write_json_atomic(output, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
