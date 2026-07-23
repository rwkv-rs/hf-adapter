#!/usr/bin/env python3
"""Exercise clean RWKV-7 tensor leaves through HF, Accelerate, PEFT, and TRL.

Training always executes the readable ``modeling_rwkv7.py`` layer loop.  The
adaptive BF16 lane independently selects recurrent, linear, and explicit-shift
Mix6 leaves; PEFT/TRL adapters remain ordinary modules around those stateless
boundaries.  The reference lane keeps every training operation in PyTorch.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import torch

from common import environment, git_revision, model_fingerprint, sha256_file


REFERENCE_MODEL = "torch-reference-model-v1"
MATRIX_RECURRENT = "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
FACTORIZED_RECURRENT = "native-nvidia-rwkv7-factorized-recurrent-training-v1"
FLATTENED_LINEAR = "torch-cuda-rwkv7-flattened-linear-training-v1"
MIX6 = "native-nvidia-rwkv7-mix6-training-v1"


def canonical_training_mode(value: str) -> str:
    aliases = {
        "adaptive": "adaptive",
        "native": "adaptive",
        "reference": "reference",
        "reference-fallback": "reference",
        "auto": "reference",
    }
    try:
        return aliases[value]
    except KeyError as exc:  # pragma: no cover - argparse validates the CLI
        raise ValueError(f"unknown training mode: {value}") from exc


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--training-mode",
        choices=("adaptive", "reference", "native", "reference-fallback"),
        default="adaptive",
        help=(
            "adaptive selects independent clean-loop tensor leaves; reference "
            "keeps pure PyTorch training. native/reference-fallback are "
            "deprecated aliases."
        ),
    )
    parser.add_argument("--training-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--code-sha")
    parser.add_argument("--hf-wheel", type=Path)
    parser.add_argument("--kernel-wheel", type=Path)
    return parser.parse_args()


def inference_backend_environment() -> None:
    os.environ["RWKV7_BACKEND"] = "optimized"
    os.environ["RWKV7_MODEL_KERNEL_IMPL"] = "native"
    os.environ["RWKV7_KERNEL_IMPL"] = "auto"
    os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = "auto"


def training_backend_environment(training_mode: str) -> None:
    training_mode = canonical_training_mode(training_mode)
    os.environ["RWKV7_KERNEL_IMPL"] = "auto"
    os.environ["RWKV7_MODEL_KERNEL_IMPL"] = "auto"
    if training_mode == "adaptive":
        os.environ["RWKV7_BACKEND"] = "auto"
        os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = "adaptive"
    else:
        os.environ["RWKV7_BACKEND"] = "reference"
        os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = "auto"


def last_route() -> dict[str, Any] | None:
    from rwkv7_hf.ops_rwkv7 import get_last_model_route

    return get_last_model_route()


def last_training_routes() -> dict[str, Any]:
    from rwkv7_hf.ops_rwkv7 import (
        get_last_linear_route,
        get_last_mix6_route,
        get_last_model_route,
        get_last_recurrent_route,
    )

    return {
        "model": get_last_model_route(),
        "recurrent": get_last_recurrent_route(),
        "linear": get_last_linear_route(),
        "mix6": get_last_mix6_route(),
    }


def expected_dense_training_route(
    routes: dict[str, Any] | None,
    training_mode: str,
    *,
    batch: int = 1,
    tokens: int = 16,
) -> bool:
    training_mode = canonical_training_mode(training_mode)
    routes = routes or {}
    model = routes.get("model") or {}
    recurrent = routes.get("recurrent") or {}
    linear = routes.get("linear") or {}
    mix6 = routes.get("mix6") or {}
    if not (
        model.get("selected") == "reference"
        and model.get("phase") == "training"
        and model.get("implementation") == REFERENCE_MODEL
    ):
        return False
    if training_mode == "reference":
        return bool(
            recurrent.get("selected") == "reference"
            and recurrent.get("implementation") == "torch-reference-v1"
            and linear.get("selected") == "reference"
            and linear.get("implementation") == "torch-reference-linear-v1"
            and mix6.get("selected") == "reference"
            and mix6.get("implementation") == "torch-reference-mix6-v1"
        )
    aligned = tokens > 0 and tokens % 16 == 0
    expected_recurrent = FACTORIZED_RECURRENT if aligned else MATRIX_RECURRENT
    linear_optimized = aligned and batch * tokens >= 128
    linear_passed = (
        linear.get("selected") == "optimized"
        and linear.get("implementation") == FLATTENED_LINEAR
        if linear_optimized
        else linear.get("selected") == "reference"
        and linear.get("implementation") == "torch-reference-linear-v1"
    )
    return bool(
        recurrent.get("selected") == "optimized"
        and recurrent.get("implementation") == expected_recurrent
        and linear_passed
        and mix6.get("selected") == "optimized"
        and mix6.get("implementation") == MIX6
    )


def training_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def training_parameter_dtype(name: str) -> torch.dtype:
    # Native BF16 training operates directly on BF16 checkpoint parameters.
    # Conventional HF FP16 training instead keeps the optimizer's master
    # parameters in FP32 and obtains FP16 compute through Accelerate/Trainer
    # autocast. Loading FP16 parameters and then enabling GradScaler makes
    # both libraries correctly reject the run while unscaling gradients.
    return torch.bfloat16 if name == "bf16" else torch.float32


def finite_nonzero_gradients(model) -> tuple[bool, int]:
    rows = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return bool(
        rows
        and all(bool(torch.isfinite(value).all()) for value in rows)
        and any(bool(value.detach().abs().max() > 0) for value in rows)
    ), len(rows)


def token_batch(
    model, *, batch: int, tokens: int, seed: int
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(
        1,
        int(model.config.vocab_size),
        (batch, tokens),
        generator=generator,
        device="cuda",
    )
    labels = ids.clone()
    labels[0, tokens // 2] = -100
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": labels,
    }


def release(*values: Any) -> None:
    del values
    gc.collect()
    torch.cuda.empty_cache()


def run_auto_model(path: Path, seed: int) -> dict[str, Any]:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    with tempfile.TemporaryDirectory(prefix="rwkv7-backend-v2-route-") as trace_dir:
        trace_path = Path(trace_dir) / "kernel-route.json"
        previous_trace = os.environ.get("RWKV7_KERNEL_TRACE_PATH")
        os.environ["RWKV7_KERNEL_TRACE_PATH"] = str(trace_path)
        try:
            config = AutoConfig.from_pretrained(path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            base = (
                AutoModel.from_pretrained(
                    path, torch_dtype=torch.float16, trust_remote_code=True
                )
                .cuda()
                .eval()
            )
            ids = torch.randint(
                1,
                int(config.vocab_size),
                (1, 17),
                generator=torch.Generator(device="cuda").manual_seed(seed),
                device="cuda",
            )
            with torch.inference_mode():
                base_output = base(input_ids=ids, use_cache=True)
            base_route = last_route()
            base_ok = bool(
                torch.isfinite(base_output.last_hidden_state).all()
                and base_output.past_key_values is not None
            )
            release(base, base_output)

            model = (
                AutoModelForCausalLM.from_pretrained(
                    path, torch_dtype=torch.float16, trust_remote_code=True
                )
                .cuda()
                .eval()
            )
            with torch.inference_mode():
                original = model(input_ids=ids, use_cache=True).logits
                greedy = model.generate(
                    ids[:, :8],
                    max_new_tokens=4,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=0,
                    eos_token_id=None,
                )
                beam = model.generate(
                    ids[:, :8],
                    max_new_tokens=4,
                    num_beams=2,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=0,
                    eos_token_id=None,
                )
            generation_route = last_route()
            with tempfile.TemporaryDirectory(
                prefix="rwkv7-backend-v2-save-"
            ) as directory:
                model.save_pretrained(directory, safe_serialization=True)
                tokenizer.save_pretrained(directory)
                reloaded = (
                    AutoModelForCausalLM.from_pretrained(
                        directory,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                    )
                    .cuda()
                    .eval()
                )
                with torch.inference_mode():
                    restored = reloaded(input_ids=ids, use_cache=False).logits
                reload_equal = bool(torch.equal(original, restored))
                saved_files = sorted(item.name for item in Path(directory).iterdir())
                release(reloaded, restored)
            from rwkv7_kernels.trace import write_trace

            write_trace()
            actual_trace = json.loads(trace_path.read_text())
        finally:
            if previous_trace is None:
                os.environ.pop("RWKV7_KERNEL_TRACE_PATH", None)
            else:
                os.environ["RWKV7_KERNEL_TRACE_PATH"] = previous_trace
    actual_model_calls = actual_trace.get("actual_model_calls", {})
    native_prefill_calls = sum(
        int(count)
        for implementation, count in actual_model_calls.items()
        if str(implementation).startswith("native-nvidia-prefill-v2[")
    )
    native_decode_calls = sum(
        int(count)
        for implementation, count in actual_model_calls.items()
        if str(implementation).startswith("native-nvidia-fused-decode-v2[")
    )
    passed = bool(
        base_ok
        and torch.isfinite(original).all()
        and greedy.shape == beam.shape == (1, 12)
        and reload_equal
        and native_prefill_calls > 0
        and native_decode_calls > 0
    )
    row = {
        "passed": passed,
        "config_class": type(config).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "base_route": base_route,
        "generation_route": generation_route,
        "actual_route_trace": actual_trace,
        "greedy_shape": list(greedy.shape),
        "beam_shape": list(beam.shape),
        "save_reload_logits_equal": reload_equal,
        "saved_files": saved_files,
    }
    release(model, original, greedy, beam)
    return row


def run_accelerate(
    path: Path, seed: int, dtype_name: str, training_mode: str
) -> dict[str, Any]:
    from accelerate import Accelerator
    from accelerate.utils import GradScalerKwargs
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    model = RWKV7ForCausalLM.from_pretrained(
        path, torch_dtype=training_parameter_dtype(dtype_name)
    ).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-5)
    scaler_kwargs = (
        [GradScalerKwargs(init_scale=128.0, growth_interval=2000)]
        if dtype_name == "fp16"
        else []
    )
    accelerator = Accelerator(
        mixed_precision=dtype_name,
        kwargs_handlers=scaler_kwargs,
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    batch = token_batch(model, batch=1, tokens=16, seed=seed)
    optimizer.zero_grad(set_to_none=True)
    with accelerator.autocast():
        output = model(**batch, use_cache=False, logits_to_keep=0)
    accelerator.backward(output.loss)
    accelerator.unscale_gradients(optimizer)
    gradients_finite, gradient_count = finite_nonzero_gradients(model)
    tracked_parameter = None
    tracked_index = None
    tracked_before = None
    for parameter in model.parameters():
        if parameter.grad is None or not bool(parameter.grad.detach().abs().max() > 0):
            continue
        tracked_parameter = parameter
        tracked_index = int(parameter.grad.detach().abs().reshape(-1).argmax())
        tracked_before = parameter.detach().reshape(-1)[tracked_index].clone()
        break
    optimizer.step()
    parameter_changed = bool(
        tracked_parameter is not None
        and tracked_before is not None
        and not torch.equal(
            tracked_before,
            tracked_parameter.detach().reshape(-1)[tracked_index],
        )
    )
    route = last_training_routes()
    passed = bool(
        torch.isfinite(output.loss)
        and gradients_finite
        and parameter_changed
        and expected_dense_training_route(route, training_mode)
    )
    row = {
        "passed": passed,
        "loss": float(output.loss.detach()),
        "finite_nonzero_gradients": gradients_finite,
        "gradient_tensor_count": gradient_count,
        "parameters_changed": parameter_changed,
        "route": route,
        "device": str(accelerator.device),
    }
    release(model, optimizer, output, accelerator)
    return row


def synthetic_dataset(vocab: int, seed: int):
    from datasets import Dataset

    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, vocab, (2, 16), generator=generator).tolist()
    return Dataset.from_dict(
        {
            "input_ids": ids,
            "attention_mask": [[1] * 16 for _ in ids],
            "labels": ids,
        }
    )


def run_trainer(
    path: Path, seed: int, dtype_name: str, training_mode: str
) -> dict[str, Any]:
    from transformers import (
        DefaultDataCollator,
        Trainer,
        TrainingArguments,
    )
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    model = RWKV7ForCausalLM.from_pretrained(
        path, torch_dtype=training_parameter_dtype(dtype_name)
    )
    dataset = synthetic_dataset(int(model.config.vocab_size), seed)
    with tempfile.TemporaryDirectory(prefix="rwkv7-backend-v2-trainer-") as directory:
        training_args = TrainingArguments(
            output_dir=directory,
            max_steps=1,
            per_device_train_batch_size=1,
            learning_rate=1.0e-5,
            bf16=dtype_name == "bf16",
            fp16=dtype_name == "fp16",
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            disable_tqdm=True,
            seed=seed,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DefaultDataCollator(),
        )
        result = trainer.train()
    route = last_training_routes()
    loss = float(result.training_loss)
    passed = bool(
        result.global_step == 1
        and torch.isfinite(torch.tensor(loss))
        and expected_dense_training_route(route, training_mode)
    )
    row = {
        "passed": passed,
        "global_step": int(result.global_step),
        "training_loss": loss,
        "route": route,
    }
    release(trainer, model, dataset, result)
    return row


def lora_config():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"],
    )


def run_peft(path: Path, seed: int, dtype_name: str) -> dict[str, Any]:
    from peft import PeftModel, get_peft_model
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    dtype = training_parameter_dtype(dtype_name)
    model = RWKV7ForCausalLM.from_pretrained(path, torch_dtype=dtype).cuda()
    model = get_peft_model(model, lora_config()).train()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1.0e-3)
    batch = token_batch(model, batch=1, tokens=16, seed=seed)
    optimizer.zero_grad(set_to_none=True)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if dtype_name == "fp16"
        else torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    )
    with autocast:
        output = model(**batch, use_cache=False, logits_to_keep=0)
    output.loss.backward()
    gradients_finite, gradient_count = finite_nonzero_gradients(model)
    optimizer.step()
    changed = any(
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(before, trainable)
    )
    route = last_training_routes()
    probe_ids = batch["input_ids"][:, :8]
    model.eval()
    with torch.inference_mode():
        expected = model(input_ids=probe_ids, use_cache=False).logits
    with tempfile.TemporaryDirectory(prefix="rwkv7-backend-v2-peft-") as directory:
        model.save_pretrained(directory)
        base = RWKV7ForCausalLM.from_pretrained(path, torch_dtype=dtype).cuda()
        reloaded = PeftModel.from_pretrained(base, directory).eval()
        with torch.inference_mode():
            actual = reloaded(input_ids=probe_ids, use_cache=False).logits
        reload_equal = bool(torch.equal(expected, actual))
        release(reloaded, base, actual)
    passed = bool(
        torch.isfinite(output.loss)
        and gradients_finite
        and changed
        and expected_dense_training_route(
            route,
            canonical_training_mode(
                os.environ.get("RWKV7_TRAINING_KERNEL_IMPL", "reference")
            ),
        )
        and reload_equal
    )
    row = {
        "passed": passed,
        "loss": float(output.loss.detach()),
        "finite_nonzero_gradients": gradients_finite,
        "gradient_tensor_count": gradient_count,
        "trainable_parameter_tensors": len(trainable),
        "parameters_changed": changed,
        "adapter_save_reload_logits_equal": reload_equal,
        "route": route,
    }
    release(model, optimizer, output, expected, before, trainable)
    return row


def run_trl_sft(path: Path, seed: int, dtype_name: str) -> dict[str, Any]:
    from transformers import AutoTokenizer, DefaultDataCollator
    from trl import SFTConfig, SFTTrainer
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = RWKV7ForCausalLM.from_pretrained(
        path, torch_dtype=training_parameter_dtype(dtype_name)
    )
    dataset = synthetic_dataset(int(model.config.vocab_size), seed)
    max_steps = 8 if dtype_name == "fp16" else 1
    with tempfile.TemporaryDirectory(prefix="rwkv7-backend-v2-trl-") as directory:
        config = SFTConfig(
            output_dir=directory,
            # A fresh GradScaler intentionally starts conservatively high.
            # A short FP16 smoke allows it to back off from any skipped warmup
            # updates and still proves that LoRA parameters really change.
            max_steps=max_steps,
            per_device_train_batch_size=1,
            learning_rate=1.0e-3,
            bf16=dtype_name == "bf16",
            fp16=dtype_name == "fp16",
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            disable_tqdm=True,
            gradient_checkpointing=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            seed=seed,
        )
        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=DefaultDataCollator(),
            peft_config=lora_config(),
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in trainer.model.named_parameters()
            if parameter.requires_grad
        }
        result = trainer.train()
        changed = any(
            name in before and not torch.equal(before[name], parameter.detach())
            for name, parameter in trainer.model.named_parameters()
            if parameter.requires_grad
        )
    route = last_training_routes()
    loss = float(result.training_loss)
    passed = bool(
        result.global_step == max_steps
        and torch.isfinite(torch.tensor(loss))
        and changed
        and expected_dense_training_route(
            route,
            canonical_training_mode(
                os.environ.get("RWKV7_TRAINING_KERNEL_IMPL", "reference")
            ),
        )
    )
    row = {
        "passed": passed,
        "global_step": int(result.global_step),
        "training_loss": loss,
        "parameters_changed": changed,
        "route": route,
    }
    release(trainer, model, dataset, tokenizer, result, before)
    return row


def execute(name: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        row = function()
    except Exception as exc:
        row = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"name": name, **row}


def main() -> int:
    args = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    training_mode = canonical_training_mode(args.training_mode)
    if training_mode == "adaptive" and args.training_dtype != "bf16":
        raise ValueError("adaptive clean-leaf ecosystem acceptance requires BF16")
    torch.manual_seed(args.seed)
    path = args.model.expanduser().resolve()
    inference_backend_environment()
    stages = [
        execute("auto-model-save-generation", lambda: run_auto_model(path, args.seed))
    ]
    training_backend_environment(training_mode)
    stages.extend(
        [
            execute(
                f"accelerate-{training_mode}",
                lambda: run_accelerate(
                    path,
                    args.seed + 1,
                    args.training_dtype,
                    training_mode,
                ),
            ),
            execute(
                f"trainer-{training_mode}",
                lambda: run_trainer(
                    path,
                    args.seed + 2,
                    args.training_dtype,
                    training_mode,
                ),
            ),
            execute(
                "peft-lora-fallback",
                lambda: run_peft(path, args.seed + 3, args.training_dtype),
            ),
            execute(
                "trl-sft-lora-fallback",
                lambda: run_trl_sft(path, args.seed + 4, args.training_dtype),
            ),
        ]
    )
    wheels = {}
    for name, wheel in (
        ("rwkv7_hf", args.hf_wheel),
        ("rwkv7_kernels", args.kernel_wheel),
    ):
        if wheel is not None:
            wheels[name] = {
                "path": str(wheel),
                "sha256": sha256_file(wheel),
            }
    passed = all(row.get("passed") for row in stages)
    report = {
        "schema": "rwkv7-backend-v2-hf-ecosystem-v2",
        "status": "passed" if passed else "failed",
        "code_sha": args.code_sha or git_revision(Path(__file__).resolve().parents[1]),
        "environment": environment(),
        "model": model_fingerprint(path),
        "wheels": wheels,
        "training_expectation": {
            "mode": training_mode,
            "requested_mode": args.training_mode,
            "dtype": args.training_dtype,
            "parameter_dtype": str(training_parameter_dtype(args.training_dtype)),
            "optimized_leaves_requested": training_mode == "adaptive",
        },
        "backend_environment": {
            name: os.environ.get(name)
            for name in (
                "RWKV7_BACKEND",
                "RWKV7_MODEL_KERNEL_IMPL",
                "RWKV7_KERNEL_IMPL",
                "RWKV7_TRAINING_KERNEL_IMPL",
            )
        },
        "stages": stages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
