from __future__ import annotations

import os
from types import SimpleNamespace

from bench.bench_speed import configure_fast_token_env


def test_speed_benchmark_sets_wrapper_and_native_backend_selectors(monkeypatch) -> None:
    monkeypatch.delenv("RWKV7_FAST_TOKEN_BACKEND", raising=False)
    monkeypatch.delenv("RWKV7_NATIVE_MODEL_BACKEND", raising=False)
    args = SimpleNamespace(fast_token_layout="3d", fast_token_backend="native_jit")

    configure_fast_token_env(args)

    assert os.environ["RWKV7_FAST_TOKEN_LAYOUT"] == "3d"
    assert os.environ["RWKV7_FAST_TOKEN_BACKEND"] == "native_jit"
    assert os.environ["RWKV7_NATIVE_MODEL_BACKEND"] == "native_jit"
