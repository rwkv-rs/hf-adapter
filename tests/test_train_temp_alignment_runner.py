from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file

from scripts.run_train_temp_official_recipe import (
    build_official_runtime_env,
    build_prepare_command,
    build_run_command,
    load_recipe,
)

from bench.bench_train_temp_alignment import (
    _convergence_exit_code,
    _load_official_config,
    _learning_rate_at_step,
    _memory_stability_summary,
    build_parser,
    compare_artifacts,
    compare_convergence_artifacts,
    compare_convergence_cohorts,
    make_deterministic_batch,
    make_deterministic_sequence,
    normalize_official_tensors,
    write_json_atomic,
)


def test_controlled_partial_convergence_is_a_successful_resumable_exit() -> None:
    assert _convergence_exit_code({"status": "pass", "failure": None}) == 0
    assert _convergence_exit_code({"status": "partial", "failure": None}) == 0
    assert _convergence_exit_code({"status": "fail", "failure": "nan"}) == 1


def test_memory_stability_summary_uses_steady_validation_points() -> None:
    summary = _memory_stability_summary(
        [
            {"step": 0, "memory_allocated_mb": 10.0, "memory_reserved_mb": 20.0},
            {"step": 50, "memory_allocated_mb": 30.0, "memory_reserved_mb": 40.0},
            {"step": 100, "memory_allocated_mb": 31.5, "memory_reserved_mb": 40.0},
            {"step": 150, "memory_allocated_mb": 31.0, "memory_reserved_mb": 42.0},
        ]
    )

    assert summary == {
        "sample_count": 3,
        "first_step": 50,
        "last_step": 150,
        "allocated_first_mb": 30.0,
        "allocated_last_mb": 31.0,
        "allocated_growth_mb": 1.0,
        "allocated_range_mb": 1.5,
        "reserved_first_mb": 40.0,
        "reserved_last_mb": 42.0,
        "reserved_growth_mb": 2.0,
        "reserved_range_mb": 2.0,
    }


def _convergence_artifact(
    path: Path,
    *,
    seed: int,
    validation_losses: list[float],
    backend: str,
) -> None:
    steps = len(validation_losses) - 1
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "axis": "train_temp_alignment_convergence",
            "status": "pass",
            "backend": backend,
            "precision": "bf16",
            "seed": seed,
            "checkpoint_sha256": "checkpoint",
            "sequence_sha256": f"sequence-{seed}",
            "validation_batch_sha256": f"validation-{seed}",
            "steps_requested": steps,
            "steps_completed": steps,
            "batch_size": 1,
            "seq_len": 512,
            "learning_rate": 6e-4,
            "learning_rate_final": 1e-5,
            "schedule_total_steps": 500_000,
            "warmup_steps": -1,
            "grad_clip": 1.0,
            "optimizer": "fused_adam",
            "eval_interval": 1,
            "optimizer_groups": [{"group_name": "lr_1x", "param_names": ["weight"]}],
            "train_curve": [
                {"step": step, "loss": 4.0 / step, "grad_norm": 2.0}
                for step in range(1, steps + 1)
            ],
            "validation_curve": [
                {"step": step, "loss": loss}
                for step, loss in enumerate(validation_losses)
            ],
            "runtime_s": 10.0,
        },
    )


def _artifact(
    path: Path, snapshot: Path, *, loss: float, phase: str = "backward"
) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "axis": "train_temp_alignment_capture",
            "backend": "official" if "official" in path.name else "hf",
            "phase": phase,
            "precision": "bf16",
            "checkpoint_sha256": "checkpoint-sha",
            "batch_sha256": "batch-sha",
            "loss": loss,
            "snapshot_file": str(snapshot),
        },
    )


def test_make_deterministic_batch_is_shifted_and_repeatable(tmp_path: Path) -> None:
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    first_meta = make_deterministic_batch(
        first, vocab_size=32, batch_size=2, seq_len=16, seed=42
    )
    second_meta = make_deterministic_batch(
        second, vocab_size=32, batch_size=2, seq_len=16, seed=42
    )

    a = load_file(first)
    b = load_file(second)
    assert tuple(a["input_ids"].shape) == (2, 16)
    assert tuple(a["targets"].shape) == (2, 16)
    assert torch.equal(a["input_ids"][:, 1:], a["targets"][:, :-1])
    assert torch.equal(a["input_ids"], b["input_ids"])
    assert (
        a["input_ids"].untyped_storage().data_ptr()
        != a["targets"].untyped_storage().data_ptr()
    )
    assert first_meta["content_sha256"] == second_meta["content_sha256"]


def test_make_deterministic_sequence_is_shifted_and_repeatable(tmp_path: Path) -> None:
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    first_meta = make_deterministic_sequence(
        first, vocab_size=32, batch_size=2, seq_len=4, steps=3, seed=101
    )
    second_meta = make_deterministic_sequence(
        second, vocab_size=32, batch_size=2, seq_len=4, steps=3, seed=101
    )

    a = load_file(first)
    b = load_file(second)
    assert tuple(a["input_ids"].shape) == (3, 2, 4)
    assert torch.equal(a["input_ids"][..., 1:], a["targets"][..., :-1])
    assert torch.equal(a["input_ids"], b["input_ids"])
    assert first_meta["content_sha256"] == second_meta["content_sha256"]


def test_increment_sequence_is_learnable_and_vocab_bounded(tmp_path: Path) -> None:
    output = tmp_path / "increment.safetensors"
    metadata = make_deterministic_sequence(
        output,
        vocab_size=65536,
        active_vocab_size=256,
        pattern="increment",
        batch_size=2,
        seq_len=16,
        steps=3,
        seed=101,
    )
    tensors = load_file(output)
    expected = (tensors["input_ids"] + 1) % 256
    assert torch.equal(expected, tensors["targets"])
    assert int(tensors["targets"].max()) < 256
    assert metadata["pattern"] == "increment"
    assert metadata["active_vocab_size"] == 256


def test_learning_rate_matches_train_temp_cosine_and_warmup_shape() -> None:
    initial = _learning_rate_at_step(
        0,
        learning_rate=6e-4,
        learning_rate_final=1e-5,
        schedule_total_steps=100,
        warmup_steps=-1,
    )
    final = _learning_rate_at_step(
        100,
        learning_rate=6e-4,
        learning_rate_final=1e-5,
        schedule_total_steps=100,
        warmup_steps=-1,
    )
    warmup = _learning_rate_at_step(
        0,
        learning_rate=6e-4,
        learning_rate_final=1e-5,
        schedule_total_steps=100,
        warmup_steps=10,
    )
    assert math.isclose(initial, 6e-4)
    assert math.isclose(final, 1e-5)
    assert math.isclose(warmup, 6e-6)


def test_compare_artifacts_accepts_matching_snapshots(tmp_path: Path) -> None:
    official_snapshot = tmp_path / "official.safetensors"
    hf_snapshot = tmp_path / "hf.safetensors"
    tensors = {
        "logits": torch.tensor([1.0, 2.0, 3.0]),
        "grad::model.layers.0.attn.k_proj.weight": torch.tensor([0.5, -0.5]),
    }
    save_file(tensors, official_snapshot)
    save_file({key: value.clone() for key, value in tensors.items()}, hf_snapshot)
    official_json = tmp_path / "official.json"
    hf_json = tmp_path / "hf.json"
    _artifact(official_json, official_snapshot, loss=4.25)
    _artifact(hf_json, hf_snapshot, loss=4.25)

    report = compare_artifacts(official_json, hf_json)

    assert report["status"] == "pass"
    assert report["tensor_count"] == 2
    assert report["missing_in_reference"] == []
    assert report["missing_in_candidate"] == []
    assert report["worst_cosine"] == 1.0
    assert report["max_relative_l2"] == 0.0
    assert report["max_relative_l2_target"] == 0.025


def test_step_compare_rejects_optimizer_group_mismatch(tmp_path: Path) -> None:
    official_snapshot = tmp_path / "official.safetensors"
    hf_snapshot = tmp_path / "hf.safetensors"
    save_file({"delta::weight": torch.ones(2)}, official_snapshot)
    save_file({"delta::weight": torch.ones(2)}, hf_snapshot)
    official_json = tmp_path / "official.json"
    hf_json = tmp_path / "hf.json"
    _artifact(official_json, official_snapshot, loss=1.0, phase="step")
    _artifact(hf_json, hf_snapshot, loss=1.0, phase="step")
    official = json.loads(official_json.read_text(encoding="utf-8"))
    candidate = json.loads(hf_json.read_text(encoding="utf-8"))
    official["optimizer_groups"] = [{"group_name": "decay", "param_names": ["weight"]}]
    candidate["optimizer_groups"] = [{"group_name": "lr_1x", "param_names": ["weight"]}]
    write_json_atomic(official_json, official)
    write_json_atomic(hf_json, candidate)

    report = compare_artifacts(official_json, hf_json)

    assert report["status"] == "fail"
    assert report["optimizer_groups_match"] is False
    assert "optimizer groups mismatch" in report["failures"]


def test_compare_rejects_gradient_checkpointing_mismatch(tmp_path: Path) -> None:
    official_snapshot = tmp_path / "official.safetensors"
    hf_snapshot = tmp_path / "hf.safetensors"
    save_file({"logits": torch.ones(2)}, official_snapshot)
    save_file({"logits": torch.ones(2)}, hf_snapshot)
    official_json = tmp_path / "official.json"
    hf_json = tmp_path / "hf.json"
    _artifact(official_json, official_snapshot, loss=1.0)
    _artifact(hf_json, hf_snapshot, loss=1.0)
    official = json.loads(official_json.read_text(encoding="utf-8"))
    candidate = json.loads(hf_json.read_text(encoding="utf-8"))
    official["gradient_checkpointing"] = True
    candidate["gradient_checkpointing"] = False
    write_json_atomic(official_json, official)
    write_json_atomic(hf_json, candidate)

    report = compare_artifacts(official_json, hf_json)

    assert report["status"] == "fail"
    assert "gradient_checkpointing" in report["provenance_mismatches"]


def test_capture_parser_exposes_memory_bounded_b16_contract() -> None:
    args = build_parser().parse_args(
        [
            "capture-hf",
            "--batch",
            "batch.safetensors",
            "--output-json",
            "capture.json",
            "--snapshot",
            "capture.safetensors",
            "--phase",
            "step",
            "--model",
            "model",
            "--checkpoint-sha256",
            "checkpoint",
            "--native",
            "--train-temp-cuda",
            "--gradient-checkpointing",
            "--omit-logits",
        ]
    )

    assert args.native is True
    assert args.train_temp_cuda is True
    assert args.gradient_checkpointing is True
    assert args.omit_logits is True


def test_convergence_parser_exposes_fail_closed_resume_contract() -> None:
    args = build_parser().parse_args(
        [
            "converge-hf",
            "--sequence",
            "sequence.safetensors",
            "--validation-batch",
            "validation.safetensors",
            "--output-json",
            "curve.json",
            "--seed",
            "131",
            "--model",
            "model",
            "--checkpoint-sha256",
            "checkpoint",
            "--resume-from",
            "resume.pt",
            "--checkpoint-out",
            "latest.pt",
            "--checkpoint-every",
            "50",
            "--stop-after-step",
            "500",
            "--gradient-checkpointing",
        ]
    )

    assert args.resume_from == "resume.pt"
    assert args.checkpoint_out == "latest.pt"
    assert args.checkpoint_every == 50
    assert args.stop_after_step == 500
    assert args.gradient_checkpointing is True


def test_step_compare_keeps_bf16_delta_as_telemetry(tmp_path: Path) -> None:
    official_snapshot = tmp_path / "official.safetensors"
    hf_snapshot = tmp_path / "hf.safetensors"
    save_file(
        {
            "logits": torch.ones(2),
            "post_step_logits": torch.ones(2),
            "delta::weight": torch.tensor([1.0, 0.0]),
        },
        official_snapshot,
    )
    save_file(
        {
            "logits": torch.ones(2),
            "post_step_logits": torch.ones(2),
            "delta::weight": torch.tensor([-1.0, 0.0]),
        },
        hf_snapshot,
    )
    official_json = tmp_path / "official.json"
    hf_json = tmp_path / "hf.json"
    _artifact(official_json, official_snapshot, loss=1.0, phase="step")
    _artifact(hf_json, hf_snapshot, loss=1.0, phase="step")
    group = {"group_name": "lr_1x", "param_names": ["weight"], "param_count": 1}
    for path in (official_json, hf_json):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        artifact["optimizer"] = "fused_adam"
        artifact["optimizer_groups"] = [group]
        artifact["post_step_loss"] = 0.5
        write_json_atomic(path, artifact)

    report = compare_artifacts(official_json, hf_json)

    assert report["status"] == "pass"
    assert report["gated_tensor_count"] == 2
    assert report["delta_tensor_count"] == 1
    assert report["delta_worst_cosine"] == -1.0
    assert report["tensor_failures"] == []


def test_compare_convergence_artifacts_gates_curves_and_provenance(
    tmp_path: Path,
) -> None:
    common = {
        "schema_version": 1,
        "axis": "train_temp_alignment_convergence",
        "status": "pass",
        "precision": "bf16",
        "checkpoint_sha256": "checkpoint",
        "sequence_sha256": "sequence",
        "validation_batch_sha256": "validation",
        "steps_requested": 2,
        "batch_size": 1,
        "seq_len": 4,
        "learning_rate": 6e-4,
        "learning_rate_final": 1e-5,
        "schedule_total_steps": 500_000,
        "warmup_steps": -1,
        "grad_clip": 1.0,
        "optimizer": "fused_adam",
        "optimizer_groups": [
            {
                "group_name": "lr_1x",
                "param_names": ["weight"],
                "param_count": 1,
                "source_param_count": 1,
                "weight_decay": 0.0,
                "my_lr_scale": 1.0,
                "lr": 6e-4,
            }
        ],
        "train_curve": [
            {"step": 1, "loss": 4.0, "grad_norm": 2.0},
            {"step": 2, "loss": 3.0, "grad_norm": 1.5},
        ],
        "validation_curve": [
            {"step": 0, "loss": 4.5},
            {"step": 2, "loss": 3.5},
        ],
    }
    official = {**common, "backend": "official_train_temp"}
    candidate = {**common, "backend": "hf_native"}
    official_path = tmp_path / "official.json"
    candidate_path = tmp_path / "candidate.json"
    write_json_atomic(official_path, official)
    write_json_atomic(candidate_path, candidate)

    passing = compare_convergence_artifacts(official_path, candidate_path)
    assert passing["status"] == "pass"
    assert passing["optimizer_groups_match"] is True

    candidate["sequence_sha256"] = "wrong"
    candidate["validation_curve"] = [
        {"step": 0, "loss": 4.5},
        {"step": 2, "loss": 4.5},
    ]
    write_json_atomic(candidate_path, candidate)
    failing = compare_convergence_artifacts(official_path, candidate_path)
    assert failing["status"] == "fail"
    assert "sequence_sha256" in failing["provenance_mismatches"]
    assert failing["final_validation_relative_diff"] > 0.02


def test_compare_convergence_cohorts_accepts_matching_success_distribution(
    tmp_path: Path,
) -> None:
    references: list[Path] = []
    candidates: list[Path] = []
    for seed in (11, 22, 33):
        reference = tmp_path / f"reference-{seed}.json"
        candidate = tmp_path / f"candidate-{seed}.json"
        _convergence_artifact(
            reference,
            seed=seed,
            validation_losses=[11.0, 4.0, 0.08],
            backend="official_train_temp",
        )
        _convergence_artifact(
            candidate,
            seed=seed,
            validation_losses=[11.0, 3.8, 0.07],
            backend="hf_train_temp_cuda",
        )
        reference_payload = json.loads(reference.read_text(encoding="utf-8"))
        reference_payload.pop("eval_interval")
        write_json_atomic(reference, reference_payload)
        references.append(reference)
        candidates.append(candidate)

    report = compare_convergence_cohorts(references, candidates)

    assert report["status"] == "pass"
    assert report["seeds_match"] is True
    assert report["runs_complete"] is True
    assert report["reference_deep_success_count"] == 3
    assert report["candidate_deep_success_count"] == 3


def test_compare_convergence_cohorts_rejects_lower_success_count(
    tmp_path: Path,
) -> None:
    references: list[Path] = []
    candidates: list[Path] = []
    for seed in (11, 22, 33):
        reference = tmp_path / f"reference-{seed}.json"
        candidate = tmp_path / f"candidate-{seed}.json"
        _convergence_artifact(
            reference,
            seed=seed,
            validation_losses=[11.0, 4.0, 0.08],
            backend="official_train_temp",
        )
        _convergence_artifact(
            candidate,
            seed=seed,
            validation_losses=[11.0, 8.0, 2.0],
            backend="hf_train_temp_cuda",
        )
        references.append(reference)
        candidates.append(candidate)

    report = compare_convergence_cohorts(references, candidates)

    assert report["status"] == "fail"
    assert report["candidate_success_count"] == 0
    assert (
        "candidate convergence success count is below reference" in report["failures"]
    )


def test_compare_artifacts_rejects_provenance_and_tensor_failures(
    tmp_path: Path,
) -> None:
    official_snapshot = tmp_path / "official.safetensors"
    hf_snapshot = tmp_path / "hf.safetensors"
    save_file({"grad::x": torch.tensor([1.0, 0.0])}, official_snapshot)
    save_file(
        {"grad::x": torch.tensor([-1.0, 0.0]), "grad::extra": torch.ones(1)},
        hf_snapshot,
    )
    official_json = tmp_path / "official.json"
    hf_json = tmp_path / "hf.json"
    _artifact(official_json, official_snapshot, loss=1.0)
    _artifact(hf_json, hf_snapshot, loss=2.0)
    candidate = json.loads(hf_json.read_text(encoding="utf-8"))
    candidate["batch_sha256"] = "different-batch"
    write_json_atomic(hf_json, candidate)

    report = compare_artifacts(official_json, hf_json)

    assert report["status"] == "fail"
    assert "batch_sha256" in report["provenance_mismatches"]
    assert report["missing_in_reference"] == ["grad::extra"]
    assert report["loss_abs_diff"] == 1.0
    assert report["worst_cosine"] == -1.0
    assert report["failures"]


def test_write_json_atomic_leaves_valid_json(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "result.json"
    write_json_atomic(output, {"status": "pass", "value": 3})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "pass",
        "value": 3,
    }
    assert not output.with_suffix(output.suffix + ".tmp").exists()


def test_normalize_official_tensors_applies_converter_transpose() -> None:
    tensors = {
        "blocks.0.att.w1": torch.arange(6, dtype=torch.float32).view(2, 3),
        "blocks.0.att.w0": torch.ones(1, 1, 3),
        "blocks.0.att.x_r": torch.full((1, 1, 3), 2.0),
        "blocks.0.att.v1": torch.full((2, 2), 7.0),
    }

    def translate(name: str, num_layers: int) -> tuple[str, bool]:
        assert num_layers == 1
        mapping = {
            "blocks.0.att.w1": ("model.layers.0.attn.w_lora.lora.0.weight", True),
            "blocks.0.att.w0": ("model.layers.0.attn.w_lora.lora.2.bias", False),
            "blocks.0.att.x_r": ("model.layers.0.attn.x_r", False),
            "blocks.0.att.v1": ("", False),
        }
        return mapping[name]

    normalized = normalize_official_tensors(
        tensors,
        num_layers=1,
        prefix="grad::",
        translate=translate,
    )
    assert set(normalized) == {
        "grad::model.layers.0.attn.w_lora.lora.0.weight",
        "grad::model.layers.0.attn.w_lora.lora.2.bias",
        "grad::model.layers.0.attn.x_r",
    }
    torch.testing.assert_close(
        normalized["grad::model.layers.0.attn.w_lora.lora.0.weight"],
        tensors["blocks.0.att.w1"].t(),
    )
    assert tuple(normalized["grad::model.layers.0.attn.w_lora.lora.2.bias"].shape) == (
        3,
    )
    assert tuple(normalized["grad::model.layers.0.attn.x_r"].shape) == (1, 1, 3)


def test_checked_in_official_config_is_production_shaped() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "train_temp_x070_12x768.json").read_text(encoding="utf-8")
    )
    assert config["my_testing"] == "x070"
    assert config["head_size"] == 64
    assert config["n_layer"] == 12
    assert config["n_embd"] == 768
    assert config["dim_att"] == config["n_embd"]
    # Current train_temp fast CMix fixes FFN width at 4x hidden.
    assert config["dim_ffn"] == 4 * config["n_embd"]
    assert config["ctx_len"] >= 512
    assert config["vocab_size"] == 65536


def test_official_shell_recipe_contract_matches_pinned_prepare_and_run() -> None:
    root = Path(__file__).resolve().parents[1]
    recipe = load_recipe(root / "configs" / "train_temp_official_x070_12x768_b16.json")
    model = recipe["model"]
    assert model["n_layer"] == 12
    assert model["n_embd"] == model["dim_att"] == 768
    assert model["reported_dim_ffn"] == 2688
    assert model["effective_dim_ffn"] == 3072
    assert model["ctx_len"] == 512
    assert recipe["prepare"]["micro_bsz"] == 1
    assert recipe["prepare"]["adam_eps"] == 1e-8
    assert recipe["run"]["micro_bsz"] == 16
    assert recipe["run"]["adam_eps"] == 1e-18
    assert recipe["run"]["grad_clip"] == 1.0

    checkout = Path("/d/references/RWKV-LM")
    data_prefix = Path("/d/datasets/minipile/minipile")
    output = Path("/d/bench/train-temp-official/out")
    prepare = build_prepare_command(checkout, data_prefix, output, recipe)
    run = build_run_command(checkout, data_prefix, output, recipe, max_steps=3)
    assert prepare[prepare.index("--micro_bsz") + 1] == "1"
    assert prepare[prepare.index("--accelerator") + 1] == "cpu"
    assert run[run.index("--micro_bsz") + 1] == "16"
    assert run[run.index("--kernel") + 1] == "@rwkv3"
    assert run[run.index("--max_steps") + 1] == "3"
    assert run[run.index("--my_exit_tokens") + 1] == "1498226207"
    alignment = _load_official_config(
        root / "configs" / "train_temp_official_x070_12x768_b16.json"
    )
    assert alignment.dim_ffn == 3072
    assert alignment.betas == (0.9, 0.99)
    assert alignment.grad_cp == 1


def test_official_recipe_runtime_uses_active_python_and_local_extension_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)
    monkeypatch.delenv("MAX_JOBS", raising=False)
    extension_dir = tmp_path / "torch-extensions"
    env, metadata = build_official_runtime_env(extension_dir)

    assert env["PATH"].split(os.pathsep)[0] == str(
        Path(sys.executable).resolve().parent
    )
    assert env["TORCH_EXTENSIONS_DIR"] == str(extension_dir.resolve())
    assert metadata["torch_extensions_dir"] == str(extension_dir.resolve())
    assert metadata["max_jobs"] == env["MAX_JOBS"]
    assert metadata["include_paths"] == [
        path for path in metadata["include_paths"] if Path(path).is_dir()
    ]
