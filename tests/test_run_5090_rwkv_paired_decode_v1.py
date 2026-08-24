from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bench" / "run_5090_rwkv_paired_decode_v1.sh"


def script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def embedded_python_programs() -> list[str]:
    return re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", script_text(), flags=re.DOTALL)


def test_every_embedded_python_program_compiles() -> None:
    programs = embedded_python_programs()

    assert len(programs) >= 4
    for index, program in enumerate(programs):
        compile(program, f"{SCRIPT.name}:heredoc-{index}", "exec")


def execute_assignment_prefix(program: str, argv: list[str]) -> dict[str, object]:
    tree = ast.parse(program)
    prefix = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            break
        prefix.append(node)
    namespace: dict[str, object] = {}
    previous = sys.argv
    try:
        sys.argv = argv
        exec(
            compile(ast.Module(prefix, type_ignores=[]), SCRIPT.name, "exec"), namespace
        )
    finally:
        sys.argv = previous
    return namespace


def test_manifest_heredoc_argv_contracts_execute_without_slice_drift() -> None:
    programs = embedded_python_programs()
    sm120 = next(
        program
        for program in programs
        if '"protocol": "sm120_b8_decode_ab_v1"' in program
    )
    correctness = next(
        program
        for program in programs
        if '"protocol": "rwkv_native_graph_fla_correctness_v1"' in program
    )

    sm120_ns = execute_assignment_prefix(
        sm120,
        ["-", "ab.json", "hashes", "c" * 40, "out", "model04", "model15"],
    )
    assert sm120_ns["model_paths"] == [Path("model04"), Path("model15")]

    lane_args = [f"lane-{index}.jsonl" for index in range(8)]
    correctness_ns = execute_assignment_prefix(
        correctness,
        [
            "-",
            "correctness.json",
            "hashes",
            "c" * 40,
            "out",
            "runtime-lock.json",
            "model04",
            "model15",
            "model29",
            "model72",
            *lane_args,
        ],
    )
    assert correctness_ns["runtime_lock"] == Path("runtime-lock.json")
    assert correctness_ns["model_paths"] == [
        Path("model04"),
        Path("model15"),
        Path("model29"),
        Path("model72"),
    ]
    assert correctness_ns["lane_paths"] == [Path(value) for value in lane_args]
    assert correctness_ns["lane_path_by_key"][("7p2", 8)] == Path("lane-7.jsonl")


def test_candidate_runner_uses_the_exact_paired_protocol() -> None:
    text = script_text()

    assert 'BENCHMARK_MATRIX="qwen35_paired_decode_v1"' in text
    assert 'OPTIMIZATION_LANE="best_optimized_hf"' in text
    assert "--model-kind rwkv" in text
    assert "--model-role candidate" in text
    assert "--model-kind qwen35" not in text
    assert "--qwen-backend" not in text
    assert "--qwen-conv-backend" not in text
    assert '--batch-sizes "${batch}"' in text
    assert "--prompt-tokens 128 512 2048" in text
    assert "--decode-tokens 128 512" in text
    assert "--prefill-chunk-size 512" in text
    assert "--warmup 3" in text
    assert "--runs 7" in text
    assert "--rwkv-attn-mode fused_recurrent" in text
    assert "--rwkv-code-source repo" in text
    assert "--rwkv-implementation auto" in text
    assert "--fail-fast" in text


def test_candidate_runner_is_eight_fresh_model_batch_processes() -> None:
    text = script_text()
    calls = [line for line in text.splitlines() if line.startswith('run_lane "')]

    assert len(calls) == 8
    assert [line.split()[4:6] for line in calls] == [
        ["1", "0"],
        ["8", "1"],
        ["1", "0"],
        ["8", "1"],
        ["1", "0"],
        ["8", "0"],
        ["1", "0"],
        ["8", "0"],
    ]
    assert "export RWKV7_FAST_TOKEN_BACKEND=native_graph" in text
    assert "export RWKV7_NATIVE_MODEL_BACKEND=native_graph" in text
    assert "export RWKV7_NATIVE_PREFILL_GRAPH=1" not in text
    assert '"RWKV7_NATIVE_PREFILL_GRAPH": "exact_card_policy"' in text
    assert 'RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM="${enabled}"' in text
    assert 'RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G="${enabled}"' in text
    assert 'RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN="${enabled}"' in text
    assert "export RWKV7_NATIVE_GRAPH_RKV_POLICY=vkwr_auto" in text
    assert "unset RWKV7_NATIVE_GRAPH_RKV_POLICY" in text
    assert '"vkwr_auto" if promoted else None' in text
    assert "export RWKV7_BLACKWELL_TORCH_COMPILE=1" in text
    assert "mktemp -d" in text
    assert 'export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor"' in text
    assert 'run_sm120_ab "${RWKV_04_MODEL}"' in text
    assert 'run_sm120_ab "${RWKV_15_MODEL}"' in text
    assert "--cells 8x2048x512" in text
    assert "--probe-tokens 512" in text
    assert "--probe-batch-size 8" in text
    assert "--required-batch-size 8" in text
    assert "--required-probe-tokens 512" in text
    assert "--require-distinct-batch-prompts" in text
    assert "promoted SM120 bundle is not faster" in text
    assert '"raw_decode_speedup"' in text
    assert '"fresh_process_per_variant": True' in text
    assert '"compile_cache": "fresh_unique_directory"' in text


def test_candidate_runner_is_append_never_and_commit_explicit() -> None:
    text = script_text()

    assert "REPOSITORY_COMMIT must be the explicit 40-hex commit under test" in text
    assert "export CUDA_VISIBLE_DEVICES REPOSITORY_COMMIT" in text
    assert 'actual_commit="$(git -C "${ROOT}" rev-parse HEAD)"' in text
    assert 'git -C "${ROOT}" status --porcelain --untracked-files=all' in text
    assert text.count("validate_repository_provenance") >= 3
    assert "requires a completely clean repository worktree" in text
    assert 'OUT_DIR="$(realpath -m -- "${OUT_DIR}")"' in text
    assert "OUT_DIR must be outside the repository" in text
    assert "ROOT must be the repository top level" in text
    assert "evidence would mutate a model input" in text
    assert "refusing to overwrite existing artifact" in text
    assert "refusing to overwrite existing log" in text
    assert "rm -f" not in text
    assert 'candidate_result="${OUT_DIR}/rwkv_candidate.jsonl"' in text
    assert 'candidate_sha256="${OUT_DIR}/rwkv_candidate.sha256"' in text
    assert 'route_manifest="${OUT_DIR}/rwkv_candidate_routes.json"' in text
    assert '"candidate_result": artifact(output)' in text
    assert '"candidate_sha256_sidecar": artifact(checksum_output)' in text
    assert '"schema_version": 1' in text
    assert text.index('with checksum_output.open("x"') < text.index(
        'with manifest_output.open("x"'
    )
    assert '"qwen_rerun": False' in text


def test_candidate_runner_runtime_matches_frozen_reference() -> None:
    text = script_text()

    for value in (
        '"3.10.12"',
        '"2.8.0+cu128"',
        '"12.8"',
        '"3.4.0"',
        '"5.12.1"',
        '"0.5.1"',
        '"1.6.2.post1"',
    ):
        assert value in text
    assert "--model 5090" in text
    assert 'driver_version}" != "595.58.03"' in text
    assert "f5bf8ef181f2c1b29b79d6fae5c8019fa85008df120569b9e18646bd09eee5cf" in text
    assert '"${OUT_DIR}/model_hashes.sha256"' in text
    assert 'sm120_ab_manifest="${OUT_DIR}/rwkv_sm120_b8_ab.json"' in text
    assert '"model_hashes_sha256"' in text
    assert 'model.rglob("*")' in text
    assert 'scope": "every recursive regular file"' in text
    assert "relative_to(model).as_posix()" in text
    assert 'model_hashes_after="${OUT_DIR}/model_hashes.after.sha256"' in text
    assert 'hash_models "${model_hashes}"' in text
    assert 'hash_models "${model_hashes_after}"' in text
    assert 'cmp --silent "${model_hashes}" "${model_hashes_after}"' in text
    assert '"byte_identical": True' in text
    assert '"sm120_b8_ab_manifest": artifact(sm120_ab_manifest)' in text
    assert '"repository_clean_pre_and_post": True' in text
    assert '"model_id_or_path": str(model_path)' in text
    # Canonical realpath accepts a full model directory on any mounted
    # filesystem, including the formal 2.9B copy under /dev/shm.
    assert 'realpath -e -- "${!name}"' in text
    assert "/mnt/qwen-data" not in text
    assert 'runtime_lock="${OUT_DIR}/runtime-lock.json"' in text
    assert 'pip_freeze="${OUT_DIR}/pip-freeze.txt"' in text
    assert 'system_csv="${OUT_DIR}/system.csv"' in text
    assert 'with Path(sys.argv[2]).open("x"' in text
    assert "expected 6 rows" in text
    assert "candidate matrix expected 48 rows" in text


def test_candidate_runner_preserves_virtualenv_python_launcher_symlink() -> None:
    text = script_text()

    assert 'realpath -e -- "${PYTHON_BIN}"' not in text
    assert 'python_dir="$(realpath -e -- "$(dirname -- "${PYTHON_BIN}")")"' in text
    assert 'PYTHON_BIN="${python_dir}/$(basename -- "${PYTHON_BIN}")"' in text
    assert 'PYTHON_BIN="$(command -v -- "${PYTHON_BIN}")"' in text
    assert '[[ ! -x "${PYTHON_BIN}" ]]' in text


def test_candidate_runner_forces_reproducible_process_environment() -> None:
    text = script_text()

    for assignment in (
        'CUDA_VISIBLE_DEVICES="0"',
        'export CUDA_DEVICE_ORDER="PCI_BUS_ID"',
        'export PYTHONPATH="${ROOT}"',
        'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"',
        'export TORCH_CUDA_ARCH_LIST="12.0"',
        "export TORCHDYNAMO_DISABLE=0",
        "export TORCH_COMPILE_DISABLE=0",
        "export HF_HUB_OFFLINE=1",
        "export TRANSFORMERS_OFFLINE=1",
        "export TOKENIZERS_PARALLELISM=false",
    ):
        assert assignment in text
    assert '"forced_environment"' in text


def test_decode_correctness_manifest_covers_every_model_and_batch() -> None:
    text = script_text()

    assert (
        'decode_correctness_manifest="${OUT_DIR}/rwkv_native_graph_fla_correctness.json"'
        in text
    )
    assert text.count("run_fla_decode_correctness_model \\") == 4
    assert "for batch in 1 8" in text
    assert '"baseline_fresh_gpu_processes": 8' in text
    assert '"candidate_additional_gpu_processes": 0' in text
    assert '"candidate_formal_lane_processes": 8' in text
    assert "--benchmark-matrix rwkv_native_graph_fla_correctness_v1" in text
    assert '--cells "${batch}x2048x512"' in text
    assert "--warmup 1" in text
    assert "--runs 1" in text
    assert text.count('--probe-cell "${batch}x2048x512"') == 2
    assert "--probe-tokens 512" in text
    assert '--probe-batch-size "${batch}"' in text
    assert "assemble_decode_correctness_manifest" in text
    assert "expected exactly six production JSON objects" in text
    assert "expected one B{batch}/P2048/D512 production cell" in text
    assert 'with row_path.open("x", encoding="utf-8")' in text
    assert "extracted native row differs from its production source" in text
    assert "from bench.analyzers.compare_rwkv_prefill_probe import compare" in text
    assert '"greedy_tokens": "exact_all_512"' in text
    assert '"prompt_logits_min_row_cosine": 0.9999' in text
    assert '"final_logits_min_row_cosine": 0.9999' in text
    assert '"b8_distinct_prompts": True' in text
    assert '"native_graph_fla_correctness_manifest": artifact(' in text


def test_decode_correctness_reference_and_candidate_routes_are_fail_closed() -> None:
    text = script_text()

    assert "export RWKV7_FAST_TOKEN_BACKEND=fla" in text
    assert "export RWKV7_NATIVE_MODEL=0" in text
    assert "export RWKV7_NATIVE_MODEL_BACKEND=eager" in text
    assert "export RWKV7_FAST_PREFILL=0" in text
    assert "export RWKV7_NATIVE_PREFILL_GRAPH=0" in text
    assert "export RWKV7_FAST_TOKEN_BACKEND=native_graph" in text
    assert "unset RWKV7_NATIVE_MODEL" in text
    assert "export RWKV7_NATIVE_MODEL_BACKEND=native_graph" in text
    assert "unset RWKV7_FAST_PREFILL" in text
    assert "unset RWKV7_NATIVE_PREFILL_GRAPH" in text
    assert '"effective_backend": "fla"' in text
    assert "--rwkv-implementation wrapper_repo" in text
    assert '"rwkv_implementation_requested": "wrapper_repo"' in text
    assert '"rwkv_implementation_effective": "wrapper_repo"' in text
    assert '"rwkv_implementation": "wrapper_repo"' in text
    assert '"rwkv_implementation": "auto"' in text
    assert '"step_backend": "rwkv_fast_token"' in text
    assert '"prefill_backend_effective": None' in text
    assert '"effective_backend": "native_graph"' in text
    assert "native_graph vs FLA correctness failed" in text
    assert '"rwkv_native_graph_sm120_compiled_ffn_full_model_effective"' not in text
    assert 'f"rwkv_native_graph_{route_name}_full_model_effective"' in text
    assert '"model_hashes_sha256"' in text
    assert '"runtime": evidence(runtime_lock)' in text
    assert '"row": evidence(row_path)' in text
    assert '"probe": evidence(probe_path)' in text
    assert '"source_lane": evidence(source_lane_path)' in text
    assert '"resident_probe_cell_selected": True' in text
    assert "is not bound to" in text
    assert '"comparison": evidence(comparison_path)' in text


def test_every_formal_lane_receives_an_append_never_native_probe_path() -> None:
    text = script_text()

    for tag in ("0p4", "1p5", "2p9", "7p2"):
        for batch in (1, 8):
            probe = (
                f'"${{OUT_DIR}}/decode_correctness_{tag}_b{batch}_native_candidate.pt"'
            )
            assert text.count(probe) == 1
    assert (
        '"${OUT_DIR}/decode_correctness_${tag}_b${batch}_native_candidate.pt"' in text
    )
    assert 'local output="$6" log="$7" probe="$8"' in text
    assert '--probe-output "${probe}"' in text
