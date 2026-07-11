from __future__ import annotations


def test_mlx_model_preserves_public_reexports() -> None:
    from rwkv7_hf import mlx_model, mlx_session, mlx_state

    assert mlx_model.MLXRWKV7State is mlx_state.MLXRWKV7State
    assert mlx_model.MLXGenerateOutput is mlx_session.MLXGenerateOutput
    assert mlx_model.MLXSessionStepOutput is mlx_session.MLXSessionStepOutput
    assert mlx_model.MLXGenerationSession is mlx_session.MLXGenerationSession
    assert mlx_model.MLXGenerationSessionBatch is mlx_session.MLXGenerationSessionBatch


def test_mlx_policy_parsing_is_dependency_free(monkeypatch) -> None:
    from rwkv7_hf.mlx_policy import (
        env_choice,
        env_flag,
        env_float,
        env_int,
        env_scan_prefill_mode,
    )

    monkeypatch.setenv("RWKV7_TEST_FLAG", "yes")
    monkeypatch.setenv("RWKV7_TEST_FLOAT", "1.25")
    monkeypatch.setenv("RWKV7_TEST_INT", "999")
    monkeypatch.setenv("RWKV7_TEST_SCAN", "force")
    monkeypatch.setenv("RWKV7_TEST_CHOICE", "compiled")

    assert env_flag("RWKV7_TEST_FLAG") is True
    assert env_float("RWKV7_TEST_FLOAT", 0.0) == 1.25
    assert env_int("RWKV7_TEST_INT", 1, upper=64) == 64
    assert env_scan_prefill_mode("RWKV7_TEST_SCAN") == "on"
    assert env_choice("RWKV7_TEST_CHOICE", "eager", {"eager", "compiled"}) == "compiled"


def test_adapter_manifest_is_unique_and_contains_extracted_runtime_modules() -> None:
    from scripts.adapter_manifest import ADAPTER_FILES

    assert len(ADAPTER_FILES) == len(set(ADAPTER_FILES))
    assert {"mlx_policy.py", "mlx_session.py", "mlx_state.py"}.issubset(ADAPTER_FILES)
