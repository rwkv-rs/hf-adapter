#!/usr/bin/env python3
# coding=utf-8
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_rwkv7_coreml.py"
RUNTIME = ROOT / "bench" / "run_coreml_apple_baseline.py"


def load_module():
    spec = importlib.util.spec_from_file_location("export_rwkv7_coreml", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runtime_module():
    spec = importlib.util.spec_from_file_location("run_coreml_apple_baseline", RUNTIME)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_minimal_model(root: Path) -> Path:
    model = root / "tiny-rwkv7-hf"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "rwkv7",
                "architectures": ["RWKV7ForCausalLM"],
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_heads": 4,
                "head_dim": 16,
                "vocab_size": 1000,
                "max_position_embeddings": 4096,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return model


def test_coreml_export_dry_run_cli_writes_manifest_and_results() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        model = write_minimal_model(tmp)
        out = tmp / "coreml-out"
        results = tmp / "results.jsonl"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(model),
                str(out),
                "--dry-run",
                "--chunks",
                "2",
                "--prefill-seq-length",
                "32",
                "--sample-seq-length",
                "8",
                "--quantization",
                "lut4",
                "--state-mode",
                "wkv-coreml",
                "--results",
                str(results),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        row = json.loads(result.stdout)
        assert row["axis"] == "rwkv7_coreml_export"
        assert row["status"] == "plan"
        assert row["quantization"] == "lut4"
        assert row["state_mode"] == "wkv-coreml"
        manifest_path = out / "coreml_export_manifest.json"
        assert Path(row["manifest"]) == manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["format"] == "rwkv7_coreml_export_manifest_v1"
        assert manifest["export_kind"] == "full-logits"
        assert manifest["chunks"] == 2
        assert manifest["prefill_seq_length"] == 32
        assert manifest["sample_seq_length"] == 8
        assert manifest["quantization"] == "lut4"
        assert manifest["shape"]["hidden_size"] == 64
        assert manifest["shape"]["vocab_size"] == 1000
        functions = {item["name"]: item for item in manifest["functions"]}
        assert functions["full_logits"]["implemented"] is True
        assert functions["full_logits"]["output"]["logits"] == [1, 8, 1000]
        assert functions["decode"]["state_mode"] == "wkv-coreml"
        assert functions["decode"]["implemented"] is False
        assert functions["prefill"]["planned_input"]["input_ids"] == [1, 32]
        result_rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
        assert result_rows == [row]


def test_coreml_export_skip_row_when_stack_unavailable() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        model = write_minimal_model(tmp)
        args = mod.argparse.Namespace(
            model=str(model),
            output=str(tmp / "out"),
            basename="",
            export_kind="full-logits",
            state_mode="wkv-coreml",
            chunks=1,
            prefill_seq_length=16,
            sample_seq_length=4,
            quantization="none",
            compute_units="cpu-and-ne",
            deployment_target="iOS18",
            dry_run=False,
            require_coremltools=False,
            results="",
        )
        config = mod.read_config(args.model)
        manifest = mod.make_manifest(args, config)
        original = mod.import_coreml_stack
        try:
            mod.import_coreml_stack = lambda require: None
            row = mod.export_full_logits(args, manifest)
        finally:
            mod.import_coreml_stack = original
        assert row["axis"] == "rwkv7_coreml_export"
        assert row["status"] == "skip"
        assert row["reason"] == "coremltools/torch/transformers stack not installed"
        assert row["export_kind"] == "full-logits"
        assert row["manifest"].endswith("coreml_export_manifest.json")


def test_coreml_export_require_coremltools_returns_structured_failure() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        model = write_minimal_model(tmp)
        original_import = mod.import_coreml_stack
        original_argv = sys.argv[:]
        stdout = io.StringIO()
        try:
            mod.import_coreml_stack = lambda require: None
            sys.argv = [
                str(SCRIPT),
                str(model),
                str(tmp / "out"),
                "--sample-seq-length",
                "4",
                "--require-coremltools",
            ]
            with contextlib.redirect_stdout(stdout):
                code = mod.main()
        finally:
            mod.import_coreml_stack = original_import
            sys.argv = original_argv
        assert code == 2
        row = json.loads(stdout.getvalue())
        assert row["status"] == "skip"
        assert row["reason"] == "coremltools/torch/transformers stack not installed"


def test_coreml_runtime_dry_run_cli_uses_manifest(tmp_path: Path) -> None:
    model = write_minimal_model(tmp_path)
    export_out = tmp_path / "coreml-export"
    manifest_path = export_out / "coreml_export_manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(model),
            str(export_out),
            "--dry-run",
            "--sample-seq-length",
            "8",
            "--quantization",
            "lut4",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    results = tmp_path / "runtime.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNTIME),
            "--manifest",
            str(manifest_path),
            "--dry-run",
            "--prompt-target-chars",
            "32,64",
            "--decode-lengths",
            "4,8",
            "--repeat",
            "2",
            "--results",
            str(results),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    row = json.loads(result.stdout)
    assert row["axis"] == "rwkv7_coreml_runtime_plan"
    assert row["status"] == "plan"
    assert row["full_logits_seq_len"] == 8
    assert row["quantization"] == "lut4"
    assert row["prompt_target_chars"] == [32, 64]
    assert row["decode_lengths"] == [4, 8]
    assert row["warmup"] == 1
    assert row["pass_status_requires_stateful_decode"] is True
    assert [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()] == [row]


def test_coreml_runtime_missing_package_emits_skip_rows(tmp_path: Path) -> None:
    model = write_minimal_model(tmp_path)
    export_out = tmp_path / "coreml-export"
    subprocess.run(
        [sys.executable, str(SCRIPT), str(model), str(export_out), "--dry-run", "--sample-seq-length", "8"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(RUNTIME),
            "--manifest",
            str(export_out / "coreml_export_manifest.json"),
            "--prompt-target-chars",
            "16",
            "--decode-lengths",
            "4,8",
            "--results",
            str(tmp_path / "missing_package_runtime.jsonl"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert len(rows) == 2
    assert {row["axis"] for row in rows} == {"qwen35_apple_baseline"}
    assert {row["status"] for row in rows} == {"skip"}
    assert {row["runtime"] for row in rows} == {"coreml"}
    assert rows[0]["reason"].startswith("CoreML package does not exist")


def test_coreml_stateful_dry_run_contract(tmp_path: Path) -> None:
    model = write_minimal_model(tmp_path)
    export_out = tmp_path / "coreml-stateful"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(model),
            str(export_out),
            "--dry-run",
            "--export-kind",
            "stateful-multifunction",
            "--prefill-seq-length",
            "64",
            "--quantization",
            "int4",
            "--quant-skip-modules",
            "lm_head",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest_path = export_out / "coreml_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["export_kind"] == "stateful-multifunction"
    assert manifest["quant_skip_modules"] == ["lm_head"]
    functions = {item["name"]: item for item in manifest["functions"]}
    assert functions["prefill"]["implemented"] is True
    assert functions["prefill"]["input"]["input_ids"] == [1, 64]
    assert functions["prefill"]["input"]["token_mask"] == [1, 64]
    assert functions["decode"]["implemented"] is True
    assert functions["decode"]["input"]["input_ids"] == [1, 1]
    contract = manifest["state_contract"]
    assert contract["boundary_dtype"] == "float16"
    assert contract["internal_accumulation_dtype"] == "float32"
    states = {item["name"]: item for item in contract["states"]}
    assert states["rwkv_recurrent_state"]["shape"] == [2, 4, 16, 16]
    assert states["rwkv_recurrent_state_residual"]["encoding"] == "fp16_residual"
    assert states["rwkv_attn_x_prev"]["shape"] == [2, 64]
    assert contract["recurrent_state_encoding"] == "fp16_high_plus_fp16_residual"

    runtime_results = tmp_path / "stateful-runtime.jsonl"
    runtime = subprocess.run(
        [
            sys.executable,
            str(RUNTIME),
            "--manifest",
            str(manifest_path),
            "--prompt-target-chars",
            "16",
            "--decode-lengths",
            "4",
            "--results",
            str(runtime_results),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert runtime.returncode == 0, runtime.stdout + runtime.stderr
    rows = [json.loads(line) for line in runtime.stdout.splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["status"] == "skip"
    assert rows[0]["reason"].startswith("stateful CoreML package does not exist")
    assert rows[0]["coreml_package"].endswith("tiny-rwkv7-hf-stateful-int4.mlpackage")


def test_coreml_stateful_rejects_oversized_static_prefill(tmp_path: Path) -> None:
    model = write_minimal_model(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(model),
            str(tmp_path / "coreml-stateful-too-large"),
            "--dry-run",
            "--export-kind",
            "stateful-multifunction",
            "--prefill-seq-length",
            "129",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--prefill-seq-length must be <= 128" in result.stderr
    assert "repeated masked chunks" in result.stderr


def test_coreml_runtime_require_flags_need_matching_verification(tmp_path: Path) -> None:
    cases = (
        ("--require-chunked-prefill-match", "requires --verify-chunked-prefill"),
        ("--require-hf-greedy-match", "requires --verify-hf-parity"),
    )
    for flag, expected in cases:
        result = subprocess.run(
            [
                sys.executable,
                str(RUNTIME),
                "--manifest",
                str(tmp_path / "not-read.json"),
                "--dry-run",
                flag,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert expected in result.stderr


def test_coreml_state_read_write_helpers() -> None:
    import numpy as np

    runtime = load_runtime_module()

    class FakeState:
        def __init__(self):
            self.values = {}

        def read_state(self, name):
            return self.values[name].copy()

        def write_state(self, name, value):
            self.values[name] = value.copy()

    source = FakeState()
    source.values = {
        "rwkv_recurrent_state": np.arange(8, dtype=np.float16).reshape(1, 2, 2, 2),
        "rwkv_v_first": np.asarray([[1.0, 2.0]], dtype=np.float16),
    }
    copied = runtime._state_values(source, list(source.values))
    target = FakeState()
    runtime._write_state_values(target, copied)
    reread = runtime._state_values(target, list(source.values))
    assert max(runtime._max_abs(np, copied[name], reread[name]) for name in copied) == 0.0


def test_coreml_masked_prefill_chunks_exact_tokens() -> None:
    import numpy as np

    runtime = load_runtime_module()

    class FakeState:
        total = 0

    class FakeModel:
        def predict(self, inputs, state):
            state.total += int(np.sum(inputs["input_ids"] * inputs["token_mask"]))
            return {"logits": np.asarray([[state.total, -state.total]], dtype=np.float32)}

    full_state = FakeState()
    full_state.total = 0
    full_logits, full_calls = runtime._run_stateful_prefill(
        np=np,
        model=FakeModel(),
        state=full_state,
        prompt_ids=[1, 2, 3],
        model_chunk_size=2,
        active_chunk_size=2,
    )
    token_state = FakeState()
    token_state.total = 0
    token_logits, token_calls = runtime._run_stateful_prefill(
        np=np,
        model=FakeModel(),
        state=token_state,
        prompt_ids=[1, 2, 3],
        model_chunk_size=2,
        active_chunk_size=1,
    )
    assert full_calls == 2
    assert token_calls == 3
    assert full_state.total == token_state.total == 6
    assert runtime._max_abs(np, full_logits, token_logits) == 0.0


def test_coreml_runtime_static_contract() -> None:
    text = RUNTIME.read_text(encoding="utf-8")
    assert RUNTIME.exists()
    assert RUNTIME.stat().st_mode & stat.S_IXUSR
    assert "qwen35_apple_baseline" in text
    assert "rwkv7_coreml_runtime_plan" in text
    assert "partial_reason" in text
    assert "full_logits CoreML package lacks stateful recurrent decode/prefill" in text
    assert "ct.ComputeUnit.CPU_AND_NE" in text
    assert "coreml_stateful" in text
    assert "read_state" in text
    assert "write_state" in text
    assert "chunked_prefill_state_max_abs" in text


def test_coreml_export_static_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert SCRIPT.exists()
    assert SCRIPT.stat().st_mode & stat.S_IXUSR
    assert "SUPPORTED_QUANTIZATION" in text
    assert '"int8"' in text
    assert '"int4"' in text
    assert '"lut4"' in text
    assert '"full-logits"' in text
    assert '"stateful-multifunction"' in text
    assert '"wkv-coreml"' in text
    assert "ct.ComputeUnit.CPU_AND_NE" in text
    assert "ct.target.iOS18" in text
    assert "PostTrainingQuantizer" in text
    assert "PostTrainingPalettizer" in text
    assert "ANE runtime benchmark rows" in text
    assert "ct.utils.MultiFunctionDescriptor" in text
    assert "rwkv_recurrent_state" in text
    assert "rwkv_recurrent_state_residual" in text


def main() -> int:
    test_coreml_export_dry_run_cli_writes_manifest_and_results()
    test_coreml_export_skip_row_when_stack_unavailable()
    test_coreml_export_require_coremltools_returns_structured_failure()
    test_coreml_runtime_dry_run_cli_uses_manifest(Path(tempfile.mkdtemp()))
    test_coreml_runtime_missing_package_emits_skip_rows(Path(tempfile.mkdtemp()))
    test_coreml_stateful_dry_run_contract(Path(tempfile.mkdtemp()))
    test_coreml_state_read_write_helpers()
    test_coreml_masked_prefill_chunks_exact_tokens()
    test_coreml_runtime_static_contract()
    test_coreml_export_static_contract()
    print("COREML EXPORT TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
