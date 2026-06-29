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
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.run(
            [sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
            env=child_env,
            stdout=stdout,
            stderr=stderr,
            check=False,
            text=True,
        )
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
            "source_revision": revision(Path(__file__).resolve().parents[2]),
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
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )
    return output


def deterministic_subset(dataset, count: int, seed: int, output: Path, name: str):
    count = min(int(count), len(dataset))
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected = sorted(indices[:count])
    selected_dataset = dataset.select(selected)
    (output / f"{name}_indices.json").write_text(
        json.dumps(selected) + "\n"
    )
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
            raise RuntimeError(f"resume checkpoint has no trainer_state.json: {state_path}")
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
    (output / "checkpoint_inventory.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
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
    trained_model.eval()
    with torch.inference_mode():
        expected = trained_model(input_ids=sample, use_cache=False).logits.float().cpu()
    base = AutoModelForCausalLM.from_pretrained(
        model_id, revision=model_revision, trust_remote_code=True
    ).to(device)
    reloaded = PeftModel.from_pretrained(base, adapter_dir).eval()
    with torch.inference_mode():
        actual = reloaded(input_ids=sample, use_cache=False).logits.float().cpu()
    max_abs = float((expected - actual).abs().max())
    result = {
        "max_abs": max_abs,
        "close": bool(torch.allclose(expected, actual, rtol=1e-5, atol=1e-5)),
    }
    (output / "adapter_reload.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    if not result["close"]:
        raise RuntimeError(f"adapter reload changed logits: {result}")


class ReproCallback(TrainerCallback):
    def __init__(self, output: Path):
        self.output = output
        self.metrics = output / "metrics.jsonl"
        self.saw_finite_loss = False
        self.saw_nonzero_gradient = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = dict(logs or {})
        loss = logs.get("loss")
        if loss is not None and torch.isfinite(torch.tensor(float(loss))):
            self.saw_finite_loss = True
        row = {"step": state.global_step, **logs}
        with self.metrics.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                if torch.isfinite(parameter.grad).all() and float(parameter.grad.abs().sum()) > 0:
                    self.saw_nonzero_gradient = True
                    break

    def write_status(self, global_step: int):
        status = {
            "finite_loss": self.saw_finite_loss,
            "nonzero_gradient": self.saw_nonzero_gradient,
            "global_step": int(global_step),
        }
        (self.output / "training_checks.json").write_text(
            json.dumps(status, indent=2) + "\n"
        )
        if not all((self.saw_finite_loss, self.saw_nonzero_gradient)):
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
    parser.add_argument(
        "--model", default="wangyue114514/rwkv7-g1d-0.1b-hf"
    )
    parser.add_argument("--model-revision", default="v0.9.0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--report-to", choices=("none", "wandb"), default="none"
    )


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
