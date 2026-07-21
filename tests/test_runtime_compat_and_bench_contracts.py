from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

from rwkv7_hf import triton_compat


ROOT = Path(__file__).resolve().parents[1]


def test_configuration_applies_runtime_compat_before_fla_import() -> None:
    source = (ROOT / "rwkv7_hf" / "configuration_rwkv7.py").read_text(encoding="utf-8")
    assert source.index("_rwkv7_apply_runtime_compat()") < source.index(
        "from fla.models.rwkv7.configuration_rwkv7"
    )


def test_batch_sweep_times_prefill_inside_inference_mode() -> None:
    tree = ast.parse((ROOT / "bench" / "bench_batch_sweep.py").read_text(encoding="utf-8"))
    bench_one = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "bench_one")
    guarded = False
    for node in ast.walk(bench_one):
        if not isinstance(node, ast.With):
            continue
        is_inference = any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "inference_mode"
            for item in node.items
        )
        assigns_prefill = any(
            isinstance(child, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == "prefill_dt"
                for target in (child.targets if isinstance(child, ast.Assign) else [child.target])
            )
            for child in ast.walk(node)
        )
        guarded = guarded or (is_inference and assigns_prefill)
    assert guarded


def test_native_benchmarks_set_both_backend_selectors() -> None:
    benches = (
        "bench_speed.py",
        "bench_batch_sweep.py",
        "bench_dynamic_batch.py",
        "bench_decode_micro.py",
        "bench_forward_fast_path.py",
        "bench_generate_fast_path.py",
        "bench_fast_token_warmup.py",
        "bench_larger_model_smoke.py",
        "bench_native_graph_overhead.py",
        "bench_native_quant_e2e_decode.py",
        "run_qwen35_speed_matrix.py",
    )
    for name in benches:
        source = (ROOT / "bench" / name).read_text(encoding="utf-8")
        assert "RWKV7_FAST_TOKEN_BACKEND" in source, name
        assert "RWKV7_NATIVE_MODEL_BACKEND" in source, name


def test_native_reference_benchmarks_force_eager_backend() -> None:
    benches = (
        "bench_speed.py",
        "bench_batch_sweep.py",
        "bench_dynamic_batch.py",
        "bench_decode_micro.py",
        "bench_decode_breakdown.py",
        "bench_forward_fast_path.py",
        "bench_generate_fast_path.py",
        "bench_quantization.py",
        "profile_decode.py",
    )
    for name in benches:
        source = (ROOT / "bench" / name).read_text(encoding="utf-8")
        assert 'RWKV7_NATIVE_MODEL_BACKEND"] = "eager"' in source, name


def test_pytorch26_triton33_disables_worker_compile(monkeypatch) -> None:
    original_compile = object()
    fake_torch = types.SimpleNamespace(__version__="2.6.0", compile=original_compile)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("RWKV7_LEGACY_TORCH_COMPILE", raising=False)
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)
    monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)

    assert triton_compat.maybe_disable_incompatible_torch_compile(True) is True
    assert fake_torch.compile is not original_compile
    assert fake_torch.compile(lambda: 1)() == 1
    assert fake_torch.compile()(lambda: 2)() == 2
    assert __import__("os").environ["TORCHDYNAMO_DISABLE"] == "1"
    assert __import__("os").environ["TORCH_COMPILE_DISABLE"] == "1"


def test_pytorch27_keeps_compile_enabled(monkeypatch) -> None:
    original_compile = object()
    fake_torch = types.SimpleNamespace(__version__="2.7.0", compile=original_compile)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("RWKV7_LEGACY_TORCH_COMPILE", raising=False)

    assert triton_compat.maybe_disable_incompatible_torch_compile(True) is False
    assert fake_torch.compile is original_compile
