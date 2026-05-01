from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bench" / "run_4080_qwen35_best_optimized_hf_v1.sh"


def test_4080_qwen_reference_runner_is_append_never_and_exact_card() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--model 4080" in text
    assert "TORCH_CUDA_ARCH_LIST=8.9" in text
    assert "qwen35_best_optimized_hf_v1" in text
    assert "--warmup 3" in text
    assert "--runs 7" in text
    assert "--qwen-backend fla" in text
    assert "--qwen-conv-backend causal_conv1d" in text
    assert "--require-qwen-fast-path" in text
    assert "--qwen-graph-probe-tokens 16" in text
    assert "refusing to overwrite existing artifact" in text
    assert "CACHE_ROOT must be absent or empty" in text
    assert "TORCHINDUCTOR_CACHE_DIR" in text
    assert "model_hashes.after.sha256" in text
    assert "root.rglob" in text
    assert "Qwen model inputs changed during formal capture" in text
    assert "CUDA_TOOLKIT_VIEW" in text
    assert "env -i" in text
    assert 'route_manifest="${OUT_DIR}/${RESULT_NAME%.jsonl}_route.json"' in text
    assert '"repository_clean_pre_and_post": True' in text
    assert '"CACHE_ROOT": str(Path(cache_root).resolve())' in text
    assert "rm -f" not in text


def test_4080_qwen_reference_runner_locks_runtime_and_provenance() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for value in (
        "3.12.2",
        "2.11.0+cu130",
        "13.0",
        "3.6.0",
        "5.12.1",
        "0.5.1",
        "1.6.2.post1",
    ):
        assert value in text
    assert "REPOSITORY_COMMIT must be the explicit 40-hex commit" in text
    assert text.count("validate_repository_provenance") >= 3
    assert '"CUDA_VISIBLE_DEVICES=0"' in text
    assert '"PYTHONPATH=${ROOT}"' in text
    assert "HF_HUB_OFFLINE=1" in text


def test_4080_qwen_reference_runner_supports_graph_and_dynamic_routes() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "module_call_dynamic" in text
    assert "static_cache_inductor_cudagraph" in text
    assert "static_cache_raw_cudagraph" in text
    assert '--qwen-decode-optimization "${QWEN_DECODE_OPTIMIZATION}"' in text
    assert '--qwen-compile-mode "${QWEN_COMPILE_MODE}"' in text
