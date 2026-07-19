from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from scripts.train_native import (
    build_parser,
    convergence_args,
    model_source_sha256,
    preset_config,
    resolve_config,
    run,
)


def _packed_data(tmp_path: Path, *, steps: int = 3, batch: int = 2, tokens: int = 16):
    sequence = tmp_path / "sequence.safetensors"
    validation = tmp_path / "validation.safetensors"
    values = torch.arange(steps * batch * tokens, dtype=torch.long).reshape(
        steps, batch, tokens
    )
    save_file({"input_ids": values, "targets": values.clone()}, sequence)
    save_file(
        {"input_ids": values[0].clone(), "targets": values[0].clone()}, validation
    )
    return sequence, validation


def _model_dir(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"rwkv7_native"}\n')
    (model / "model.safetensors").write_bytes(b"weights")
    return model


def test_official_preset_keeps_shell_hyperparameters_configurable() -> None:
    values = preset_config("official-x070-12x768-b16")
    assert values["batch_size"] == 16
    assert values["seq_length"] == 512
    assert values["learning_rate"] == 6e-4
    assert values["learning_rate_final"] == 6e-5
    assert values["adam_eps"] == 1e-18
    assert values["steps"] is None


def test_official_shell_argument_aliases_map_to_native_fields() -> None:
    args = build_parser().parse_args(
        [
            "--micro-bsz",
            "8",
            "--ctx-len",
            "1024",
            "--max-steps",
            "20",
            "--lr-init",
            "0.0003",
            "--lr-final",
            "0.00003",
        ]
    )
    assert args.batch_size == 8
    assert args.seq_length == 1024
    assert args.steps == 20
    assert args.learning_rate == 3e-4
    assert args.learning_rate_final == 3e-5


def test_config_then_cli_override_and_sequence_shape_inference(tmp_path: Path) -> None:
    sequence, validation = _packed_data(tmp_path)
    model = _model_dir(tmp_path)
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps({"schema_version": 1, "learning_rate": 1e-4, "seed": 7}),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--config",
            str(config),
            "--model",
            str(model),
            "--sequence",
            str(sequence),
            "--validation-batch",
            str(validation),
            "--output-dir",
            str(tmp_path / "out"),
            "--learning-rate",
            "0.0002",
        ]
    )
    values = resolve_config(args)
    assert values["steps"] == 3
    assert values["batch_size"] == 2
    assert values["seq_length"] == 16
    assert values["learning_rate"] == 2e-4
    assert values["seed"] == 7
    assert values["schedule_total_steps"] == 3


def test_train_temp_kernel_shape_is_the_only_ctx_constraint(tmp_path: Path) -> None:
    sequence, validation = _packed_data(tmp_path, tokens=15)
    args = build_parser().parse_args(
        [
            "--model",
            str(_model_dir(tmp_path)),
            "--sequence",
            str(sequence),
            "--validation-batch",
            str(validation),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(ValueError, match="divisible by 16"):
        resolve_config(args)


def test_known_incompatible_model_head_is_rejected_early(tmp_path: Path) -> None:
    sequence, validation = _packed_data(tmp_path)
    model = _model_dir(tmp_path)
    (model / "config.json").write_text(
        '{"model_type":"rwkv7_native","head_dim":32}\n', encoding="utf-8"
    )
    args = build_parser().parse_args(
        [
            "--model",
            str(model),
            "--sequence",
            str(sequence),
            "--validation-batch",
            str(validation),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(ValueError, match="head dimensions of 64"):
        resolve_config(args)


def test_model_source_hash_is_deterministic_and_content_sensitive(
    tmp_path: Path,
) -> None:
    model = _model_dir(tmp_path)
    first = model_source_sha256(model)
    assert first == model_source_sha256(model)
    (model / "model.safetensors").write_bytes(b"new weights")
    assert first != model_source_sha256(model)


def test_convergence_namespace_enables_native_train_temp(tmp_path: Path) -> None:
    sequence, validation = _packed_data(tmp_path)
    model = _model_dir(tmp_path)
    args = build_parser().parse_args(
        [
            "--model",
            str(model),
            "--sequence",
            str(sequence),
            "--validation-batch",
            str(validation),
            "--output-dir",
            str(tmp_path / "out"),
            "--optimizer",
            "torch_adamw",
            "--no-gradient-checkpointing",
        ]
    )
    values = resolve_config(args)
    runner = convergence_args(values, sequence, validation)
    assert runner.native is True
    assert runner.train_temp_cuda is True
    assert runner.optimizer == "torch_adamw"
    assert runner.gradient_checkpointing is False
    assert runner.checkpoint_sha256 == model_source_sha256(model)


def test_source_checkout_dry_run_writes_resolved_config(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    sequence, validation = _packed_data(tmp_path)
    model = _model_dir(tmp_path)
    output = tmp_path / "out"
    env = dict(**__import__("os").environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "train_native.py"),
            "--model",
            str(model),
            "--sequence",
            str(sequence),
            "--validation-batch",
            str(validation),
            "--output-dir",
            str(output),
            "--dry-run",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    result = json.loads(proc.stdout)
    resolved = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    assert result["status"] == "dry_run"
    assert result["backend"] == "hf_native_train_temp_cuda"
    assert resolved["preset"] == "native-default"
    assert resolved["data_prefix"] is None
    assert resolved["batch_size"] == 2
    assert resolved["checkpoint_sha256"] is None

    resumed = build_parser().parse_args(
        [
            "--config",
            str(output / "resolved_config.json"),
            "--resume-from",
            str(output / "checkpoint.pt"),
        ]
    )
    resumed_values = resolve_config(resumed)
    assert resumed_values["preset"] == "native-default"
    assert resumed_values["sequence"] == str(sequence.resolve())
    assert resumed_values["resume_from"] == str((output / "checkpoint.pt").resolve())


def test_data_prefix_dry_run_does_not_materialize_packed_data(tmp_path: Path) -> None:
    model = _model_dir(tmp_path)
    data = tmp_path / "tiny"
    data.with_suffix(".bin").write_bytes(bytes(16 * 10 * 2))
    data.with_suffix(".idx").write_bytes(b"index")
    output = tmp_path / "out"
    args = build_parser().parse_args(
        [
            "--model",
            str(model),
            "--data-prefix",
            str(data),
            "--output-dir",
            str(output),
            "--seq-length",
            "16",
            "--steps",
            "2",
            "--magic-prime",
            "5",
            "--dry-run",
        ]
    )
    result = run(args)
    resolved = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    assert result["status"] == "dry_run"
    assert resolved["data_prefix"] == str(data.resolve())
    assert resolved["sequence"] is None
    assert not (output / "data").exists()
