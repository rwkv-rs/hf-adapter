#!/usr/bin/env python3
from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from rwkv7_hf.sm70_linear import (
    sm70_linear,
    sm70_linear_available,
    sm70_linear_should_use,
    sm70_linear_threads,
    sm70_ffn_down_add,
    sm70_ffn_down_add_should_use,
    sm70_ffn_up_relu2,
    sm70_ffn_up_relu2_should_use,
    sm70_rkv,
    sm70_rkv_should_use,
    sm70_rkv_threads,
)


def main() -> int:
    if torch is None or F is None:
        print("SKIP sm70 linear: torch unavailable")
        return 0
    torch.manual_seed(701)
    x = torch.randn(2, 32)
    weight = torch.randn(48, 32)
    assert torch.equal(sm70_linear(x, weight, force_fallback=True), F.linear(x, weight))
    assert torch.equal(sm70_linear(x[0], weight, force_fallback=True), F.linear(x[0], weight))
    assert sm70_linear_should_use(1, 65536, 1024, role="head")
    assert not sm70_linear_should_use(2, 65536, 1024, role="head")
    assert sm70_linear_should_use(1, 2048, 2048, role="hidden")
    assert sm70_linear_should_use(1, 1024, 1024, role="hidden")
    assert sm70_linear_should_use(1, 4096, 1024, role="ffn_up")
    assert not sm70_linear_should_use(2, 4096, 1024, role="ffn_up")
    assert sm70_linear_should_use(2, 1024, 4096, role="ffn_down")
    assert sm70_linear_threads(1, 16384, 4096, role="ffn_up") == 64
    assert sm70_rkv_should_use(1, 4096)
    assert sm70_rkv_should_use(2, 2048)
    assert not sm70_rkv_should_use(2, 4096)
    assert sm70_rkv_threads(1, 768) == 256
    assert sm70_rkv_threads(1, 2048) == 64
    assert sm70_rkv_threads(2, 1024) == 128
    square_weight = weight[:32, :32]
    refs = (
        F.linear(x, square_weight),
        F.linear(x + 1, square_weight),
        F.linear(x - 1, square_weight),
    )
    got = sm70_rkv(
        x,
        x + 1,
        x - 1,
        square_weight,
        square_weight,
        square_weight,
        force_fallback=True,
    )
    for actual, expected in zip(got, refs):
        assert torch.equal(actual, expected)
    up_weight = torch.randn(128, 32)
    down_weight = torch.randn(32, 128)
    up_ref = torch.relu(F.linear(x[0], up_weight)) ** 2
    up_got = sm70_ffn_up_relu2(x[0], up_weight, force_fallback=True)
    assert torch.equal(up_got, up_ref)
    residual = torch.randn(32)
    down_ref = residual + F.linear(up_ref, down_weight)
    down_got = sm70_ffn_down_add(up_ref, down_weight, residual, force_fallback=True)
    assert torch.equal(down_got, down_ref)
    assert sm70_ffn_up_relu2_should_use(1, 4096, 1024)
    assert not sm70_ffn_up_relu2_should_use(2, 4096, 1024)
    assert sm70_ffn_down_add_should_use(1, 1024, 4096)
    assert sm70_ffn_down_add_should_use(2, 1024, 4096)
    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) != (7, 0):
        print("SKIP sm70 linear CUDA test")
        return 0
    assert sm70_linear_available("cuda", build=True)
    x = torch.randn(2, 1024, device="cuda", dtype=torch.float16)
    weight = torch.randn(4096, 1024, device="cuda", dtype=torch.float16)
    linear_reference = F.linear(x, weight)
    actual = sm70_linear(x, weight)
    torch.cuda.synchronize()
    assert torch.allclose(actual.float(), linear_reference.float(), atol=4e-3, rtol=3e-3)
    assert float(F.cosine_similarity(actual.float(), linear_reference.float(), dim=-1).min()) >= 0.9999
    hidden = 1024
    xr = torch.randn(1, hidden, device="cuda", dtype=torch.float16)
    xk = torch.randn_like(xr)
    xv = torch.randn_like(xr)
    wr = torch.randn(hidden, hidden, device="cuda", dtype=torch.float16)
    wk = torch.randn_like(wr)
    wv = torch.randn_like(wr)
    references = (F.linear(xr, wr), F.linear(xk, wk), F.linear(xv, wv))
    actuals = sm70_rkv(xr, xk, xv, wr, wk, wv)
    torch.cuda.synchronize()
    for actual, rkv_reference in zip(actuals, references):
        assert float(F.cosine_similarity(actual.float(), rkv_reference.float(), dim=-1).min()) >= 0.9999
    ffn_input = torch.randn(1, hidden, device="cuda", dtype=torch.float16) * 0.1
    up_weight = torch.randn(4 * hidden, hidden, device="cuda", dtype=torch.float16) * 0.03
    up_reference = torch.relu(F.linear(ffn_input, up_weight)) ** 2
    up_actual = sm70_ffn_up_relu2(ffn_input, up_weight)
    residual = torch.randn(1, hidden, device="cuda", dtype=torch.float16) * 0.1
    down_weight = torch.randn(hidden, 4 * hidden, device="cuda", dtype=torch.float16) * 0.01
    down_reference = residual + F.linear(up_reference, down_weight)
    down_actual = sm70_ffn_down_add(up_actual, down_weight, residual)
    torch.cuda.synchronize()
    assert float(F.cosine_similarity(up_actual.float(), up_reference.float(), dim=-1).min()) >= 0.9999
    assert float(F.cosine_similarity(down_actual.float(), down_reference.float(), dim=-1).min()) >= 0.9999
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = sm70_linear(x, weight)
    graph.replay()
    torch.cuda.synchronize()
    assert float(F.cosine_similarity(captured.float(), linear_reference.float(), dim=-1).min()) >= 0.9999
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
