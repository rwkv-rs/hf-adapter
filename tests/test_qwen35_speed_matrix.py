#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.bench_cross_model_speed import (  # noqa: E402
    build_exact_prompt,
    enforce_qwen_backend,
    failure_row,
    model_metadata,
    prepare_rwkv_model_dir,
    qwen_fla_operator_contract,
    validate_args,
)
from bench.compare_qwen35_backend_probe import compare as compare_backend_probe  # noqa: E402
from bench.compare_rwkv_prefill_probe import compare as compare_rwkv_prefill_probe  # noqa: E402
from bench.run_qwen35_speed_matrix import (  # noqa: E402
    MatrixConfig,
    build_run_environment,
    build_run_specs,
    build_worker_environment,
    existing_keys,
    parse_pair_spec,
)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def row(
    role: str,
    *,
    prompt: int,
    prefill: float,
    decode: float,
    status: str = "pass",
    qwen_backend: str = "fla",
) -> dict:
    result = {
        "axis": "qwen35_cross_model_speed",
        "model_pair": "rwkv-1.5b__qwen3.5-2b",
        "benchmark_matrix": "qwen35_test_hf",
        "model_role": role,
        "model_kind": "rwkv" if role == "candidate" else "qwen35",
        "status": status,
        "dtype": "fp16",
        "quantization": "none",
        "prompt_tokens": prompt,
        "decode_tokens": 128,
        "batch_size": 1,
        "prefill_tokps_total": prefill,
        "decode_tokps_total": decode,
    }
    if role == "reference":
        result.update(
            {
                "qwen_backend_requested": qwen_backend,
                "qwen_operator_contract_pass": qwen_backend == "fla",
                "qwen_force_torch": qwen_backend == "torch",
                "effective_backend": (
                    "qwen_fla_gated_delta_rule"
                    if qwen_backend == "fla"
                    else "transformers_torch_fallback"
                ),
            }
        )
    else:
        result.update({"qwen_backend_requested": qwen_backend, "effective_backend": "native_graph"})
    return result


def run_compare(tmp: Path, rows: list[dict], *extra: str) -> subprocess.CompletedProcess[str]:
    results = tmp / "results.jsonl"
    write_rows(results, rows)
    return subprocess.run(
        [
            sys.executable,
            "bench/compare_qwen35_speed_matrix.py",
            "--results",
            str(results),
            "--json-output",
            str(tmp / "summary.json"),
            "--markdown-output",
            str(tmp / "summary.md"),
            *extra,
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_comparator_passes_complete_matrix(tmp_path: Path) -> None:
    tmp = tmp_path
    rows = [
        row("candidate", prompt=128, prefill=120.0, decode=220.0),
        row("reference", prompt=128, prefill=100.0, decode=200.0),
        row("candidate", prompt=512, prefill=210.0, decode=330.0),
        row("reference", prompt=512, prefill=200.0, decode=300.0),
    ]
    proc = run_compare(
        tmp,
        rows,
        "--expected-cells",
        "2",
        "--min-prefill-speedup",
        "1.05",
        "--min-decode-speedup",
        "1.05",
        "--fail-on-gate",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((tmp / "summary.json").read_text(encoding="utf-8"))
    assert summary["coverage"] == {"expected_cells": 2, "joined_cells": 2, "complete": True}
    assert summary["speed"]["min_prefill_speedup"] == 1.05
    assert summary["speed"]["min_decode_speedup"] == 1.1
    assert summary["gates"]["overall_pass"] is True
    assert "Overall: PASS" in (tmp / "summary.md").read_text(encoding="utf-8")


def test_comparator_reports_missing_and_slow_cells(tmp_path: Path) -> None:
    tmp = tmp_path
    rows = [
        row("candidate", prompt=128, prefill=90.0, decode=180.0),
        row("reference", prompt=128, prefill=100.0, decode=200.0),
        row("candidate", prompt=512, prefill=210.0, decode=330.0),
    ]
    proc = run_compare(
        tmp,
        rows,
        "--expected-cells",
        "2",
        "--min-prefill-speedup",
        "1.0",
        "--min-decode-speedup",
        "1.0",
        "--fail-on-gate",
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    summary = json.loads((tmp / "summary.json").read_text(encoding="utf-8"))
    assert summary["coverage"]["joined_cells"] == 1
    assert summary["coverage"]["complete"] is False
    assert len(summary["missing"]["reference"]) == 1
    assert len(summary["red_cells"]) == 1
    assert summary["gates"]["overall_pass"] is False


def test_comparator_can_gate_memory_per_cell(tmp_path: Path) -> None:
    candidate = row("candidate", prompt=128, prefill=120.0, decode=220.0)
    reference = row("reference", prompt=128, prefill=100.0, decode=200.0)
    candidate.update({"model_footprint_mb": 90.0, "peak_vram_mb": 110.0})
    reference.update({"model_footprint_mb": 100.0, "peak_vram_mb": 100.0})
    proc = run_compare(
        tmp_path,
        [candidate, reference],
        "--expected-cells",
        "1",
        "--require-memory-not-larger",
        "--fail-on-gate",
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["gates"]["memory_pass"] is False
    assert summary["red_cells"][0]["memory_pass"] is False


def test_rwkv_prefill_probe_requires_greedy_and_logits_alignment() -> None:
    reference = {
        "input_ids": torch.tensor([[1, 2]]),
        "greedy_tokens": torch.tensor([3, 4]),
        "prompt_logits": torch.tensor([[1.0, 2.0]]),
        "final_logits": torch.tensor([[2.0, 3.0]]),
    }
    native = {key: value.clone() for key, value in reference.items()}
    assert compare_rwkv_prefill_probe(reference, native, 0.9999)["status"] == "pass"
    native["greedy_tokens"][1] = 5
    assert compare_rwkv_prefill_probe(reference, native, 0.9999)["status"] == "fail"


def test_comparator_rejects_torch_qwen_reference(tmp_path: Path) -> None:
    tmp = tmp_path
    rows = [
        row("candidate", prompt=128, prefill=120.0, decode=220.0),
        row("reference", prompt=128, prefill=100.0, decode=200.0, qwen_backend="torch"),
    ]
    proc = run_compare(tmp, rows, "--expected-cells", "1", "--fail-on-gate")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    summary = json.loads((tmp / "summary.json").read_text(encoding="utf-8"))
    assert summary["reference_backend"]["required"] == "fla"
    assert summary["reference_backend"]["matching_cells"] == 0
    assert summary["gates"]["reference_backend_pass"] is False


def test_comparator_accepts_fla_core_with_torch_conv(tmp_path: Path) -> None:
    reference = row("reference", prompt=128, prefill=100.0, decode=200.0)
    reference.update(
        {
            "effective_backend": "qwen_fla_gated_delta_rule_torch_conv",
            "qwen_fla_core_contract_pass": True,
            "qwen_causal_conv1d_contract_pass": False,
            "qwen_full_fused_contract_pass": False,
        }
    )
    proc = run_compare(
        tmp_path,
        [row("candidate", prompt=128, prefill=120.0, decode=220.0), reference],
        "--expected-cells",
        "1",
        "--fail-on-gate",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["reference_backend"]["matching_cells"] == 1
    assert summary["gates"]["reference_backend_pass"] is True


def fake_operator(origin: str):
    module, name = origin.rsplit(".", 1)

    def operator(*_args, **_kwargs):
        return None

    operator.__module__ = module
    operator.__name__ = name
    operator.__qualname__ = name
    return operator


def fake_qwen_model(*, accelerated: bool, fused_conv: bool = True):
    if accelerated:
        prefill = fake_operator("fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule")
        decode = fake_operator("fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule")
        conv = (
            fake_operator("causal_conv1d.causal_conv1d_interface.causal_conv1d_fn")
            if fused_conv
            else None
        )
        conv_update = (
            fake_operator("causal_conv1d.causal_conv1d_interface.causal_conv1d_update")
            if fused_conv
            else fake_operator("transformers.models.qwen3_5.modeling_qwen3_5.torch_causal_conv1d_update")
        )
        norm_type = type("FusedRMSNormGated", (), {})
        norm_type.__module__ = "fla.modules"
    else:
        prefill = fake_operator("transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule")
        decode = fake_operator("transformers.models.qwen3_5.modeling_qwen3_5.torch_recurrent_gated_delta_rule")
        conv = None
        conv_update = fake_operator("transformers.models.qwen3_5.modeling_qwen3_5.torch_causal_conv1d_update")
        norm_type = type("Qwen3_5RMSNormGated", (), {})
        norm_type.__module__ = "transformers.models.qwen3_5.modeling_qwen3_5"
    layer = SimpleNamespace(
        chunk_gated_delta_rule=prefill,
        recurrent_gated_delta_rule=decode,
        causal_conv1d_fn=conv,
        causal_conv1d_update=conv_update,
        norm=norm_type(),
    )
    return SimpleNamespace(named_modules=lambda: [("model.layers.0.linear_attn", layer)])


def test_qwen_fla_operator_contract_checks_bound_operators() -> None:
    model = fake_qwen_model(accelerated=True)
    contract = qwen_fla_operator_contract(model)
    assert contract["qwen_operator_contract_pass"] is True
    assert contract["qwen_linear_attention_layers"] == 1
    assert contract["qwen_fla_prefill_layers"] == 1
    assert contract["qwen_fla_decode_layers"] == 1
    assert contract["qwen_causal_conv1d_prefill_layers"] == 1
    assert contract["qwen_causal_conv1d_update_layers"] == 1
    assert contract["qwen_fla_norm_layers"] == 1
    assert contract["qwen_fla_core_contract_pass"] is True
    assert contract["qwen_causal_conv1d_contract_pass"] is True
    assert contract["qwen_full_fused_contract_pass"] is True
    assert enforce_qwen_backend(model, worker_args(model_kind="qwen35", qwen_backend="fla")) == contract

    windows_model = fake_qwen_model(accelerated=True, fused_conv=False)
    windows_contract = qwen_fla_operator_contract(windows_model)
    assert windows_contract["qwen_operator_contract_pass"] is True
    assert windows_contract["qwen_fla_core_contract_pass"] is True
    assert windows_contract["qwen_causal_conv1d_contract_pass"] is False
    assert windows_contract["qwen_full_fused_contract_pass"] is False
    assert enforce_qwen_backend(
        windows_model, worker_args(model_kind="qwen35", qwen_backend="fla")
    ) == windows_contract

    fallback = fake_qwen_model(accelerated=False)
    fallback_contract = qwen_fla_operator_contract(fallback)
    assert fallback_contract["qwen_operator_contract_pass"] is False
    try:
        enforce_qwen_backend(fallback, worker_args(model_kind="qwen35", qwen_backend="fla"))
    except RuntimeError as exc:
        assert "FLA backend was required" in str(exc)
        assert "chunk_gated_delta_rule" in str(exc)
    else:
        raise AssertionError("required Qwen FLA backend must reject bound torch fallback operators")

    partial_layer = fake_qwen_model(accelerated=True).named_modules()[0][1]
    del partial_layer.recurrent_gated_delta_rule
    partial = SimpleNamespace(named_modules=lambda: [("model.layers.0.linear_attn", partial_layer)])
    partial_contract = qwen_fla_operator_contract(partial)
    assert partial_contract["qwen_linear_attention_layers"] == 1
    assert partial_contract["qwen_fla_decode_layers"] == 0
    assert partial_contract["qwen_operator_contract_pass"] is False


class FakeTokenizer:
    def __call__(self, _text: str, **_kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[5, 6, 7]], dtype=torch.long))


def worker_args(**updates) -> Namespace:
    values = {
        "model": "/models/rwkv7-g1g-1.5b-hf",
        "model_kind": "rwkv",
        "model_role": "candidate",
        "model_pair": "rwkv-1.5b__qwen3.5-2b",
        "model_size_label": "1.5b",
        "benchmark_matrix": "qwen35_test_hf",
        "dtype": "fp16",
        "quantization": "none",
        "device": "cpu",
        "batch_size": 2,
        "prompt_tokens": 8,
        "decode_tokens": 4,
        "warmup": 1,
        "runs": 1,
        "rwkv_code_source": "repo",
        "qwen_backend": "fla",
        "probe_output": "",
        "probe_tokens": 8,
    }
    values.update(updates)
    return Namespace(**values)


def test_worker_helpers_build_exact_shape_and_metadata() -> None:
    args = worker_args()
    validate_args(args)
    ids = build_exact_prompt(FakeTokenizer(), args.prompt_tokens, args.batch_size, "cpu")
    assert ids.tolist() == [[5, 6, 7, 5, 6, 7, 5, 6]] * 2

    config = SimpleNamespace(
        model_type="rwkv7",
        hidden_size=2048,
        num_hidden_layers=24,
        vocab_size=65536,
    )
    metadata = model_metadata(args, SimpleNamespace(config=config))
    assert metadata["model_name"] == "rwkv7-g1g-1.5b-hf"
    assert metadata["model_type"] == "rwkv7"
    assert metadata["hidden_size"] == 2048


def test_worker_helpers_validate_and_emit_failure() -> None:
    args = worker_args(prompt_tokens=0)
    try:
        validate_args(args)
    except ValueError as exc:
        assert "prompt-tokens" in str(exc)
    else:
        raise AssertionError("validate_args should reject prompt_tokens=0")

    args = worker_args()
    result = failure_row(args, RuntimeError("synthetic failure"))
    assert result["axis"] == "qwen35_cross_model_speed"
    assert result["status"] == "fail"
    assert result["model_role"] == "candidate"
    assert "synthetic failure" in result["error"]


def test_repo_code_staging_works_without_symlink_privilege(tmp_path: Path) -> None:
    source = tmp_path / "rwkv-model"
    source.mkdir()
    weight = source / "model.safetensors"
    weight.write_bytes(b"weights")
    (source / "stale_modeling.py").write_text("STALE = True\n", encoding="utf-8")

    staged_path, temporary = prepare_rwkv_model_dir(str(source), "repo")
    assert temporary is not None
    staged = Path(staged_path)
    assert (staged / "model.safetensors").read_bytes() == b"weights"
    assert not (staged / "stale_modeling.py").exists()
    assert (staged / "modeling_rwkv7.py").exists()
    temporary.cleanup()
    assert not staged.exists()


def test_backend_probe_comparator_checks_logits_and_greedy() -> None:
    common = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "prompt_logits": torch.tensor([[0.1, 0.2, 0.3]]),
        "final_logits": torch.tensor([[0.4, 0.5, 0.6]]),
        "greedy_tokens": torch.tensor([3, 4, 5]),
    }
    result = compare_backend_probe(
        {**common, "qwen_backend_requested": "fla"},
        {**common, "qwen_backend_requested": "torch"},
        0.999,
    )
    assert result["status"] == "pass"
    assert result["greedy_tokens_match"] is True

    mismatch = compare_backend_probe(
        {**common, "greedy_tokens": torch.tensor([3, 4, 6])},
        common,
        0.999,
    )
    assert mismatch["status"] == "fail"


def test_orchestrator_expands_432_raw_rows() -> None:
    pairs = [
        parse_pair_spec("rwkv-1.5b__qwen3.5-2b=/rwkv/1.5b::Qwen/Qwen3.5-2B"),
        parse_pair_spec("rwkv-2.9b__qwen3.5-4b=/rwkv/2.9b::Qwen/Qwen3.5-4B"),
        parse_pair_spec("rwkv-7.2b__qwen3.5-9b=/rwkv/7.2b::Qwen/Qwen3.5-9B"),
    ]
    config = MatrixConfig(
        pairs=pairs,
        prompts=[128, 512, 2048],
        decodes=[128, 512],
        batch_sizes=[1, 2, 4, 8],
        quantizations=["none", "bnb8", "bnb4"],
        dtype="fp16",
    )
    specs = build_run_specs(config)
    assert len(specs) == 432
    assert len({spec.cell_key for spec in specs}) == 216
    assert {spec.model_role for spec in specs} == {"candidate", "reference"}
    assert specs[0].model_kind == "rwkv"
    assert specs[1].model_kind == "qwen35"


def test_orchestrator_existing_keys_are_resumable(tmp_path: Path) -> None:
    tmp = tmp_path
    result_path = tmp / "results.jsonl"
    rows = [
        row("candidate", prompt=128, prefill=120.0, decode=220.0),
        row("reference", prompt=128, prefill=100.0, decode=200.0, status="fail"),
    ]
    write_rows(result_path, rows)
    keys = existing_keys(result_path)
    assert len(keys) == 2
    assert any(key[-2] == "candidate" for key in keys)
    assert any(key[-2] == "reference" for key in keys)
    assert {key[-1] for key in keys} == {"fla"}


def test_orchestrator_forces_production_rwkv_wrapper() -> None:
    args = Namespace(rwkv_fast_token_backend="native_graph")
    env = build_run_environment(args, {"RWKV7_NATIVE_MODEL": "1", "PYTHONPATH": "/existing"})
    assert env["RWKV7_NATIVE_MODEL"] == "0"
    assert env["RWKV7_FAST_TOKEN_BACKEND"] == "native_graph"
    assert env["PYTHONPATH"].endswith(f"{os.pathsep}/existing")


def test_orchestrator_isolates_qwen_import_backend() -> None:
    pair = parse_pair_spec("rwkv-1.5b__qwen3.5-2b=/rwkv/1.5b::Qwen/Qwen3.5-2B")
    specs = build_run_specs(
        MatrixConfig(
            pairs=[pair],
            prompts=[128],
            decodes=[8],
            batch_sizes=[1],
            quantizations=["none"],
            dtype="fp16",
        )
    )
    qwen_spec = next(spec for spec in specs if spec.model_kind == "qwen35")
    base = {"RWKV7_QWEN35_FORCE_TORCH": "1"}
    assert "RWKV7_QWEN35_FORCE_TORCH" not in build_worker_environment(base, qwen_spec, "fla")
    assert build_worker_environment({}, qwen_spec, "torch")["RWKV7_QWEN35_FORCE_TORCH"] == "1"
    rwkv_bnb8 = next(spec for spec in specs if spec.model_kind == "rwkv")
    rwkv_bnb8 = type(rwkv_bnb8)(**{**rwkv_bnb8.__dict__, "quantization": "bnb8"})
    env = build_worker_environment({}, rwkv_bnb8, "fla", "decode_rk")
    assert env["RWKV7_BNB_SKIP_POLICY"] == "decode_rk"


def test_5070_qwen_fla_evidence_is_complete() -> None:
    evidence = ROOT / "bench" / "5070_qwen35_fla_matrix_20260713"
    rows = [json.loads(line) for line in (evidence / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 144
    assert all(row["status"] == "pass" for row in rows)

    qwen_rows = [row for row in rows if row["model_role"] == "reference"]
    assert len(qwen_rows) == 72
    assert all(row["qwen_fla_core_contract_pass"] is True for row in qwen_rows)
    assert all(
        row["effective_backend"] == "qwen_fla_gated_delta_rule_torch_conv"
        for row in qwen_rows
    )

    summary = json.loads((evidence / "summary.json").read_text(encoding="utf-8-sig"))
    assert summary["coverage"]["joined_cells"] == 72
    assert summary["reference_backend"]["matching_cells"] == 72
    assert summary["speed"]["strict_gate_cells"] == 35
    assert summary["speed"]["decode_at_least_equal_cells"] == 72
    assert summary["memory"]["model_footprint_not_larger_cells"] == 72
    assert summary["memory"]["peak_vram_not_larger_cells"] == 72

    probe = json.loads((evidence / "fla-vs-torch-probe.json").read_text(encoding="utf-8-sig"))
    assert probe["status"] == "pass"
    assert probe["greedy_tokens_match"] is True
    assert min(probe["prompt_logits_cosine"], probe["final_logits_cosine"]) >= 0.999

    exit_codes = json.loads((evidence / "exit-codes.json").read_text(encoding="utf-8-sig"))
    assert exit_codes
    assert all(code == 0 for code in exit_codes.values())


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        test_comparator_passes_complete_matrix(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_comparator_reports_missing_and_slow_cells(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_comparator_rejects_torch_qwen_reference(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_comparator_accepts_fla_core_with_torch_conv(Path(td))
    test_qwen_fla_operator_contract_checks_bound_operators()
    test_worker_helpers_build_exact_shape_and_metadata()
    test_worker_helpers_validate_and_emit_failure()
    with tempfile.TemporaryDirectory() as td:
        test_repo_code_staging_works_without_symlink_privilege(Path(td))
    test_backend_probe_comparator_checks_logits_and_greedy()
    test_orchestrator_expands_432_raw_rows()
    with tempfile.TemporaryDirectory() as td:
        test_orchestrator_existing_keys_are_resumable(Path(td))
    test_orchestrator_forces_production_rwkv_wrapper()
    test_orchestrator_isolates_qwen_import_backend()
    test_5070_qwen_fla_evidence_is_complete()
    print("QWEN35 SPEED MATRIX TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
