from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import torch
from transformers import TrainerCallback, set_seed


TARGET_MODULES = ["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"]
TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
READABLE_MODEL_IMPLEMENTATION = "torch-reference-model-v1"
MATRIX_RECURRENT_IMPLEMENTATION = (
    "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
)
FACTORIZED_RECURRENT_IMPLEMENTATION = (
    "native-nvidia-rwkv7-factorized-recurrent-training-v1"
)
FLATTENED_LINEAR_IMPLEMENTATION = "torch-cuda-rwkv7-flattened-linear-training-v1"
MIX6_IMPLEMENTATION = "native-nvidia-rwkv7-mix6-training-v1"
ADAPTIVE_TRAINING_PROGRAM_IMPLEMENTATION = (
    "native-nvidia-rwkv7-adaptive-training-program-v1"
)
HISTORICAL_WHOLE_MODEL_IMPLEMENTATION = "native-nvidia-official-training-autograd-v2"


def _output_dir_from_argv() -> Path | None:
    for index, value in enumerate(sys.argv[1:]):
        if value == "--output-dir" and index + 2 <= len(sys.argv) - 1:
            return Path(sys.argv[index + 2]).expanduser().resolve()
        if value.startswith("--output-dir="):
            return Path(value.split("=", 1)[1]).expanduser().resolve()
    return None


def run_captured(main) -> int:
    """Run an example under a tiny supervisor that always records its exit."""

    marker = "RWKV7_HF_FINETUNE_CHILD"
    if os.environ.get(marker) == "1":
        main()
        return 0
    output = _output_dir_from_argv()
    if output is None or any(value in {"-h", "--help"} for value in sys.argv[1:]):
        main()
        return 0
    output.mkdir(parents=True, exist_ok=True)
    stdout_path = output / "stdout.log"
    stderr_path = output / "stderr.log"
    child_env = dict(os.environ)
    child_env[marker] = "1"
    child_env.setdefault(
        "RWKV7_KERNEL_TRACE_PATH",
        str(output / "kernel_route_trace.json"),
    )
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.run(
            [sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
            env=child_env,
            stdout=stdout,
            stderr=stderr,
            check=False,
            text=True,
        )
    reconcile_kernel_trace_checks(output)
    status = {
        "returncode": int(process.returncode),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    (output / "exit_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False))
    return int(process.returncode)


def reconcile_kernel_trace_checks(output: Path) -> None:
    """Merge process-wide optional-leaf evidence into training checks.

    Preference trainers may run a differentiable policy pass followed by a
    no-grad reference pass. The latter is correctly the last ContextVar route,
    but it must not erase the earlier optimized execution. The optional package
    writes an actual-call counter at child-process exit; the supervisor merges
    that immutable evidence after the child has terminated.
    """

    checks_path = output / "training_checks.json"
    trace_path = output / "kernel_route_trace.json"
    if not checks_path.is_file() or not trace_path.is_file():
        return
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    model = trace.get("actual_model_calls", {})
    recurrent = trace.get("actual_recurrent_calls", {})
    linear = trace.get("actual_linear_calls", {})
    mix6 = trace.get("actual_mix6_calls", {})
    if not all(isinstance(value, dict) for value in (model, recurrent, linear, mix6)):
        raise RuntimeError("kernel route trace contains invalid call counters")

    checks["matrix_recurrent_leaf"] = bool(
        checks.get("matrix_recurrent_leaf")
        or int(recurrent.get(MATRIX_RECURRENT_IMPLEMENTATION, 0))
    )
    checks["factorized_recurrent_leaf"] = bool(
        checks.get("factorized_recurrent_leaf")
        or int(recurrent.get(FACTORIZED_RECURRENT_IMPLEMENTATION, 0))
    )
    checks["flattened_linear_leaf"] = bool(
        checks.get("flattened_linear_leaf")
        or int(linear.get(FLATTENED_LINEAR_IMPLEMENTATION, 0))
    )
    checks["mix6_leaf"] = bool(
        checks.get("mix6_leaf") or int(mix6.get(MIX6_IMPLEMENTATION, 0))
    )
    checks["historical_whole_model_diagnostic"] = bool(
        checks.get("historical_whole_model_diagnostic")
        or int(model.get(HISTORICAL_WHOLE_MODEL_IMPLEMENTATION, 0))
    )
    checks["clean_leaf_training"] = bool(
        checks.get("readable_model_loop")
        and checks.get("factorized_recurrent_leaf")
        and checks.get("flattened_linear_leaf")
        and checks.get("mix6_leaf")
    )
    checks["kernel_trace_schema"] = trace.get("schema")
    checks["kernel_trace_actual_model_calls"] = model
    checks["kernel_trace_actual_recurrent_calls"] = recurrent
    checks["kernel_trace_actual_linear_calls"] = linear
    checks["kernel_trace_actual_mix6_calls"] = mix6
    checks_path.write_text(
        json.dumps(checks, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def optional_artifact(path: str | None) -> dict | None:
    if not path:
        return None
    artifact = Path(path).expanduser().resolve()
    if not artifact.is_file():
        raise RuntimeError(f"artifact does not exist: {artifact}")
    return {
        "path": str(artifact),
        "bytes": artifact.stat().st_size,
        "sha256": sha256(artifact),
    }


def model_provenance(model: str, requested_revision: str) -> dict:
    path = Path(model).expanduser()
    if path.exists():
        files = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file()
            and (
                candidate.suffix
                in {".json", ".jinja", ".model", ".py", ".safetensors", ".txt"}
                or candidate.name.endswith(".safetensors.index.json")
            )
        )
        hashes = {
            str(candidate.relative_to(path)): sha256(candidate) for candidate in files
        }
        aggregate = hashlib.sha256()
        for name, digest in hashes.items():
            aggregate.update(f"{name}\0{digest}\n".encode())
        return {
            "kind": "local",
            "path": str(path.resolve()),
            "requested_revision": requested_revision,
            "resolved_revision": aggregate.hexdigest(),
            "files": hashes,
        }
    result = {
        "kind": "hub",
        "repo_id": model,
        "requested_revision": requested_revision,
        "resolved_revision": None,
    }
    try:
        from huggingface_hub import HfApi

        result["resolved_revision"] = (
            HfApi().model_info(model, revision=requested_revision).sha
        )
    except Exception as error:
        result["resolution_error"] = f"{type(error).__name__}: {error}"
    return result


def model_load_kwargs(args) -> dict:
    """Return explicit, reproducible Hugging Face model-loading arguments."""

    kwargs = {
        "revision": args.model_revision,
        "trust_remote_code": True,
    }
    if args.torch_dtype != "auto":
        # The model is loaded directly in the requested execution dtype.
        # Trainer AMP remains disabled so CUDA autocast cannot silently promote
        # LayerNorm outputs and break the optional leaf's BF16 contract.
        kwargs["dtype"] = TORCH_DTYPES[args.torch_dtype]
    return kwargs


def trainer_precision_flags() -> dict[str, bool]:
    """Disable a second AMP layer after explicit model-dtype selection.

    TRL configuration defaults vary by release and accelerator. The examples
    load the model directly in ``--torch-dtype``, so an additional autocast
    context would change LayerNorm outputs back to FP32 and silently select a
    different optional training route.
    """

    return {"bf16": False, "fp16": False}


def gradient_checkpointing_kwargs() -> dict[str, bool]:
    """Keep checkpoint recomputation in the ordinary autograd context.

    The non-reentrant PyTorch implementation records the forward autograd
    graph while still discarding saved activations. This lets an optional leaf
    inspect the real differentiable request instead of the no-grad probe pass
    used by legacy reentrant checkpointing.
    """

    return {"use_reentrant": False}


def attach_lora_adapters(model, args):
    """Attach LoRA once, with an explicit and reproducible adapter dtype.

    PEFT normally promotes FP16/BF16 adapters to FP32. That stability-first
    default is retained unless ``--lora-dtype model`` is requested. Matching
    the model dtype keeps projection outputs in the BF16 contract required by
    the optional factorized training leaf; the readable HF layer loop and
    ordinary PEFT parameter ownership remain unchanged.
    """

    from peft import get_peft_model

    return get_peft_model(
        model,
        lora_config(),
        autocast_adapter_dtype=args.lora_dtype == "float32",
    )


def prepare_run(args, dataset_name: str, dataset_revision: str) -> Path:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    resolved = dict(vars(args))
    resolved.update(
        {
            "dataset_name": dataset_name,
            "dataset_revision": dataset_revision,
            "target_modules": TARGET_MODULES,
            "command": sys.argv,
            "source_revision": args.code_sha
            or revision(Path(__file__).resolve().parents[2])
            or "unknown",
        }
    )
    (output / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": package_version("transformers"),
        "trl": package_version("trl"),
        "peft": package_version("peft"),
        "datasets": package_version("datasets"),
        "wandb": package_version("wandb"),
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    artifacts = {
        name: row
        for name, row in (
            ("rwkv7_hf", optional_artifact(args.hf_wheel)),
            ("rwkv7_kernels", optional_artifact(args.kernel_wheel)),
        )
        if row is not None
    }
    (output / "artifact_provenance.json").write_text(
        json.dumps(artifacts, indent=2) + "\n", encoding="utf-8"
    )
    (output / "model_provenance.json").write_text(
        json.dumps(
            model_provenance(args.model, args.model_revision),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def deterministic_subset(dataset, count: int, seed: int, output: Path, name: str):
    count = min(int(count), len(dataset))
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected = sorted(indices[:count])
    selected_dataset = dataset.select(selected)
    (output / f"{name}_indices.json").write_text(json.dumps(selected) + "\n")
    fingerprint_path = output / "dataset_fingerprints.json"
    fingerprints = (
        json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if fingerprint_path.is_file()
        else {}
    )
    fingerprints[name] = {
        "source": getattr(dataset, "_fingerprint", None),
        "selected": getattr(selected_dataset, "_fingerprint", None),
    }
    fingerprint_path.write_text(
        json.dumps(fingerprints, indent=2) + "\n", encoding="utf-8"
    )
    return selected_dataset


def validate_resume(resume_from_checkpoint: str | None, global_step: int, output: Path):
    result = {
        "requested": resume_from_checkpoint,
        "prior_global_step": None,
        "final_global_step": int(global_step),
        "advanced": True,
    }
    if resume_from_checkpoint:
        state_path = Path(resume_from_checkpoint) / "trainer_state.json"
        if not state_path.is_file():
            raise RuntimeError(
                f"resume checkpoint has no trainer_state.json: {state_path}"
            )
        prior = int(json.loads(state_path.read_text())["global_step"])
        result["prior_global_step"] = prior
        result["advanced"] = int(global_step) > prior
    (output / "resume_check.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if not result["advanced"]:
        raise RuntimeError(f"resumed run did not advance global_step: {result}")


def checkpoint_inventory(output: Path) -> list[dict]:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and (
            path.suffix in {".json", ".safetensors"}
            or path.name.startswith("trainer_state")
        ):
            rows.append(
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    (output / "checkpoint_inventory.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def snapshot_trainable(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def validate_parameter_change(model, before: dict[str, torch.Tensor], output: Path):
    changed = []
    for name, parameter in model.named_parameters():
        if name in before and not torch.equal(before[name], parameter.detach().cpu()):
            changed.append(name)
    (output / "changed_parameters.json").write_text(
        json.dumps(changed, indent=2) + "\n"
    )
    if not changed:
        raise RuntimeError("no trainable parameter changed")


def lora_parameter_dtype(model) -> torch.dtype:
    """Return the single dtype used by the active LoRA matrices."""

    dtypes = {
        parameter.dtype
        for name, parameter in model.named_parameters()
        if {"lora_A", "lora_B"}.intersection(name.split("."))
    }
    if not dtypes:
        raise RuntimeError("adapter validation found no LoRA parameters")
    if len(dtypes) != 1:
        names = ", ".join(sorted(str(dtype) for dtype in dtypes))
        raise RuntimeError(f"LoRA parameters use mixed dtypes: {names}")
    return next(iter(dtypes))


def validate_adapter_reload(
    trained_model,
    *,
    model_id: str,
    model_revision: str,
    adapter_dir: Path,
    output: Path,
):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    device = next(trained_model.parameters()).device
    sample = torch.tensor([[1, 2, 3]], device=device)
    # Gradient checkpointing is a training-only execution mode.  Disable it on
    # both sides so adapter serialization is compared under the same canonical
    # inference configuration.
    trained_model.gradient_checkpointing_disable()
    trained_model.eval()
    with torch.inference_mode():
        expected = trained_model(input_ids=sample, use_cache=False).logits.float().cpu()
    trained_adapter_dtype = lora_parameter_dtype(trained_model)
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        trust_remote_code=True,
        dtype=next(trained_model.parameters()).dtype,
    ).to(device)
    reloaded = PeftModel.from_pretrained(
        base,
        adapter_dir,
        # PEFT promotes adapter matrices to FP32 while restoring a Trainer
        # checkpoint. Match the actual trained runtime instead of forcing the
        # fresh-run dtype and then comparing two different GEMM programs.
        autocast_adapter_dtype=trained_adapter_dtype == torch.float32,
    ).eval()
    reloaded.gradient_checkpointing_disable()
    reloaded.config.use_cache = trained_model.config.use_cache
    reloaded.config.bos_token_id = trained_model.config.bos_token_id
    with torch.inference_mode():
        actual = reloaded(input_ids=sample, use_cache=False).logits.float().cpu()
        expected_repeat = (
            trained_model(input_ids=sample, use_cache=False).logits.float().cpu()
        )
        actual_repeat = reloaded(input_ids=sample, use_cache=False).logits.float().cpu()
    max_abs = float((expected - actual).abs().max())
    expected_parameters = dict(trained_model.named_parameters())
    parameter_differences = []
    for name, parameter in reloaded.named_parameters():
        reference = expected_parameters.get(name)
        if reference is None:
            parameter_differences.append({"name": name, "missing": True})
            continue
        difference = float(
            (reference.detach().float().cpu() - parameter.detach().float().cpu())
            .abs()
            .max()
        )
        if difference or reference.dtype != parameter.dtype:
            parameter_differences.append(
                {
                    "name": name,
                    "max_abs": difference,
                    "trained_dtype": str(reference.dtype),
                    "reloaded_dtype": str(parameter.dtype),
                }
            )
    expected_buffers = dict(trained_model.named_buffers())
    buffer_differences = []
    for name, buffer in reloaded.named_buffers():
        reference = expected_buffers.get(name)
        if reference is None or not torch.equal(
            reference.detach().cpu(), buffer.detach().cpu()
        ):
            buffer_differences.append(name)

    def adapter_runtime(value):
        for module in value.modules():
            if hasattr(module, "lora_A") and hasattr(module, "scaling"):
                return {
                    "active_adapters": list(getattr(module, "active_adapters", [])),
                    "disable_adapters": bool(
                        getattr(module, "disable_adapters", False)
                    ),
                    "merged": bool(getattr(module, "merged", False)),
                    "scaling": dict(getattr(module, "scaling", {})),
                }
        return {}

    result = {
        "max_abs": max_abs,
        "close": bool(torch.allclose(expected, actual, rtol=1e-5, atol=1e-5)),
        "trained_dtype": str(next(trained_model.parameters()).dtype),
        "reloaded_dtype": str(next(reloaded.parameters()).dtype),
        "trained_adapter_dtype": str(trained_adapter_dtype),
        "reloaded_adapter_dtype": str(lora_parameter_dtype(reloaded)),
        "parameter_differences": parameter_differences[:50],
        "buffer_differences": buffer_differences[:50],
        "trained_adapter_runtime": adapter_runtime(trained_model),
        "reloaded_adapter_runtime": adapter_runtime(reloaded),
        "trained_repeat_max_abs": float((expected - expected_repeat).abs().max()),
        "reloaded_repeat_max_abs": float((actual - actual_repeat).abs().max()),
        "trained_gradient_checkpointing": bool(
            getattr(
                trained_model.base_model.model.model, "gradient_checkpointing", False
            )
        ),
        "reloaded_gradient_checkpointing": bool(
            getattr(reloaded.base_model.model.model, "gradient_checkpointing", False)
        ),
        "config_differences": {
            key: [
                trained_model.config.to_dict().get(key),
                reloaded.config.to_dict().get(key),
            ]
            for key in sorted(
                trained_model.config.to_dict().keys() | reloaded.config.to_dict().keys()
            )
            if trained_model.config.to_dict().get(key)
            != reloaded.config.to_dict().get(key)
        },
    }
    (output / "adapter_reload.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["close"]:
        raise RuntimeError(f"adapter reload changed logits: {result}")


class ReproCallback(TrainerCallback):
    def __init__(self, output: Path):
        self.output = output
        self.metrics = output / "metrics.jsonl"
        self.saw_finite_loss = False
        self.saw_nonzero_gradient = False
        self.backend_routes = []

    @staticmethod
    def _last_backend_routes(model=None):
        # A local/Hub model loaded through AutoModel uses Transformers' remote
        # module namespace, so its ops module owns a different ContextVar from
        # the installed package module. Resolve both public route accessors
        # from the actual model class first; retain the installed package for
        # direct imports.
        routes = {}
        if model is not None:
            get_base_model = getattr(model, "get_base_model", None)
            base_model = get_base_model() if callable(get_base_model) else model
            modeling_module = sys.modules.get(type(base_model).__module__)
            maybe_forward = getattr(modeling_module, "maybe_model_forward", None)
            ops_module = sys.modules.get(getattr(maybe_forward, "__module__", ""))
            for boundary, name in (
                ("model", "get_last_model_route"),
                ("recurrent", "get_last_recurrent_route"),
                ("linear", "get_last_linear_route"),
                ("mix6", "get_last_mix6_route"),
                ("program", "get_last_training_program_route"),
            ):
                getter = getattr(ops_module, name, None)
                if callable(getter):
                    route = getter()
                    if isinstance(route, dict):
                        routes[boundary] = route
        try:
            from rwkv7_hf.ops_rwkv7 import (
                get_last_linear_route,
                get_last_mix6_route,
                get_last_model_route,
                get_last_recurrent_route,
                get_last_training_program_route,
            )

            for boundary, getter in (
                ("model", get_last_model_route),
                ("recurrent", get_last_recurrent_route),
                ("linear", get_last_linear_route),
                ("mix6", get_last_mix6_route),
                ("program", get_last_training_program_route),
            ):
                route = getter()
                if boundary not in routes and isinstance(route, dict):
                    routes[boundary] = route
        except Exception:
            pass
        return routes

    def _capture_backend_route(self, event: str, model=None):
        for boundary, route in self._last_backend_routes(model).items():
            row = {"event": event, "boundary": boundary, **route}
            if row not in self.backend_routes:
                self.backend_routes.append(row)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._capture_backend_route("log", kwargs.get("model"))
        logs = dict(logs or {})
        loss = logs.get("loss")
        if loss is not None and torch.isfinite(torch.tensor(float(loss))):
            self.saw_finite_loss = True
        row = {"step": state.global_step, **logs}
        with self.metrics.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        self._capture_backend_route("pre_optimizer_step", model)
        if model is None:
            return
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                if (
                    torch.isfinite(parameter.grad).all()
                    and float(parameter.grad.abs().sum()) > 0
                ):
                    self.saw_nonzero_gradient = True
                    break

    def write_status(self, global_step: int):
        historical_whole_model_diagnostic = any(
            route.get("event") == "pre_optimizer_step"
            and route.get("boundary") == "model"
            and route.get("phase") == "training"
            and route.get("implementation") == HISTORICAL_WHOLE_MODEL_IMPLEMENTATION
            for route in self.backend_routes
        )
        readable_model_loop = any(
            route.get("event") == "pre_optimizer_step"
            and route.get("boundary") == "model"
            and route.get("selected") == "reference"
            and route.get("phase") == "training"
            and route.get("implementation") == READABLE_MODEL_IMPLEMENTATION
            for route in self.backend_routes
        )
        matrix_recurrent_leaf = any(
            route.get("event") == "pre_optimizer_step"
            and route.get("boundary") == "recurrent"
            and route.get("selected") == "optimized"
            and route.get("implementation") == MATRIX_RECURRENT_IMPLEMENTATION
            for route in self.backend_routes
        )
        factorized_recurrent_leaf = any(
            route.get("event") == "pre_optimizer_step"
            and route.get("boundary") == "recurrent"
            and route.get("selected") == "optimized"
            and route.get("implementation") == FACTORIZED_RECURRENT_IMPLEMENTATION
            for route in self.backend_routes
        )
        flattened_linear_leaf = any(
            route.get("event") == "pre_optimizer_step"
            and route.get("boundary") == "linear"
            and route.get("selected") == "optimized"
            and route.get("implementation") == FLATTENED_LINEAR_IMPLEMENTATION
            for route in self.backend_routes
        )
        mix6_leaf = any(
            route.get("event") == "pre_optimizer_step"
            and route.get("boundary") == "mix6"
            and route.get("selected") == "optimized"
            and route.get("implementation") == MIX6_IMPLEMENTATION
            for route in self.backend_routes
        )
        adaptive_fast_program = any(
            route.get("event") == "pre_optimizer_step"
            and route.get("boundary") == "program"
            and route.get("selected") == "optimized"
            and route.get("implementation") == ADAPTIVE_TRAINING_PROGRAM_IMPLEMENTATION
            for route in self.backend_routes
        )
        adaptive_program_fallback = any(
            route.get("event") == "pre_optimizer_step"
            and route.get("boundary") == "program"
            and route.get("selected") == "reference"
            and route.get("implementation") == ADAPTIVE_TRAINING_PROGRAM_IMPLEMENTATION
            for route in self.backend_routes
        )
        status = {
            "finite_loss": self.saw_finite_loss,
            "nonzero_gradient": self.saw_nonzero_gradient,
            "global_step": int(global_step),
            "readable_model_loop": readable_model_loop,
            "matrix_recurrent_leaf": matrix_recurrent_leaf,
            "factorized_recurrent_leaf": factorized_recurrent_leaf,
            "flattened_linear_leaf": flattened_linear_leaf,
            "mix6_leaf": mix6_leaf,
            "adaptive_fast_program": adaptive_fast_program,
            "adaptive_program_fallback": adaptive_program_fallback,
            "clean_leaf_training": bool(
                readable_model_loop
                and factorized_recurrent_leaf
                and flattened_linear_leaf
                and mix6_leaf
            ),
            "historical_whole_model_diagnostic": (historical_whole_model_diagnostic),
        }
        (self.output / "backend_routes.json").write_text(
            json.dumps(self.backend_routes, indent=2) + "\n", encoding="utf-8"
        )
        (self.output / "training_checks.json").write_text(
            json.dumps(status, indent=2) + "\n"
        )
        if historical_whole_model_diagnostic or not all(
            (self.saw_finite_loss, self.saw_nonzero_gradient)
        ):
            raise RuntimeError(f"training checks failed: {status}")


def lora_config():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
    )


def common_arguments(parser):
    parser.add_argument("--model", default="wangyue114514/rwkv7-g1d-0.1b-hf")
    parser.add_argument("--model-revision", default="v0.9.0")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", *TORCH_DTYPES),
        default="auto",
        help=(
            "model and Trainer precision; bfloat16 selects the optional "
            "factorized training leaf on supported NVIDIA GPUs"
        ),
    )
    parser.add_argument(
        "--lora-dtype",
        choices=("float32", "model"),
        default="float32",
        help=(
            "LoRA parameter dtype: float32 is PEFT's stability default; model "
            "retains the selected model dtype for optimized BF16 training"
        ),
    )
    parser.add_argument(
        "--code-sha",
        default=None,
        help="Source revision for rsync deployments without a .git directory",
    )
    parser.add_argument(
        "--hf-wheel",
        default=None,
        help="Exact rwkv7-hf wheel whose SHA256 should be recorded",
    )
    parser.add_argument(
        "--kernel-wheel",
        default=None,
        help="Exact rwkv7-kernels wheel whose SHA256 should be recorded",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Keep the readable reference examples bounded; increase for a larger effective batch",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none")


def report_target(args) -> str:
    return "wandb" if args.report_to == "wandb" else "none"


def record_wandb(output: Path) -> None:
    try:
        import wandb

        run = wandb.run
    except Exception:
        run = None
    row = {
        "enabled": run is not None,
        "id": getattr(run, "id", None),
        "url": getattr(run, "url", None),
    }
    (output / "wandb.json").write_text(json.dumps(row, indent=2) + "\n")
