from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "bench"
    / "run_4080_qwen35_paired_pd_v1.sh"
).read_text(encoding="utf-8")


def test_top_level_runner_is_append_never_and_owns_one_path_tree() -> None:
    assert "formal OUT_DIR and CACHE_ROOT must both be absent" in SCRIPT
    assert 'CACHE_ROOT="${CACHE_ROOT}/rwkv"' in SCRIPT
    for tag in ("qwen_0p8", "qwen_2b", "qwen_4b"):
        assert tag in SCRIPT
    assert "rm -" not in SCRIPT


def test_top_level_runner_locks_the_three_whole_model_qwen_routes() -> None:
    assert SCRIPT.count("static_cache_inductor_cudagraph") == 2
    assert SCRIPT.count("static_cache_raw_cudagraph") == 0
    assert SCRIPT.count("module_call_dynamic") == 1
    assert "qwen_reference.jsonl" in SCRIPT


def test_top_level_runner_only_writes_pass_after_bundle_authentication() -> None:
    validator = SCRIPT.index("validate_qwen35_paired_pd_bundle_v1.py")
    table = SCRIPT.index("paired_pd_table.jsonl")
    exit_code = SCRIPT.index("exit_code.txt")
    assert validator < table < exit_code
    assert "--qwen-route-manifest" in SCRIPT
    assert "--expected-candidate-commit" in SCRIPT
