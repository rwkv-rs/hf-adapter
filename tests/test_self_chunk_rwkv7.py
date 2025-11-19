import pytest

torch = pytest.importorskip("torch")

from rwkv7_hf.self_chunk_rwkv7 import self_chunk_rwkv7, self_chunk_rwkv7_available


def test_self_chunk_import_is_dependency_free():
    assert isinstance(self_chunk_rwkv7_available(), bool)


def test_rtx3090_chunk_h_tiles_are_shape_routed(monkeypatch):
    pytest.importorskip("triton")
    import rwkv7_hf.self_chunk_h_fwd as chunk_h

    monkeypatch.setattr(
        chunk_h,
        "check_shared_mem",
        lambda architecture, _device=None: architecture == "ampere",
    )
    monkeypatch.setattr(chunk_h.torch.cuda, "get_device_name", lambda _device=None: "NVIDIA GeForce RTX 3090")
    monkeypatch.delenv("RWKV7_NATIVE_PREFILL_SELF_CHUNK_H_BV", raising=False)
    monkeypatch.delenv("RWKV7_NATIVE_PREFILL_SELF_CHUNK_H_BC", raising=False)

    assert chunk_h.resolve_chunk_h_tiles(0, 32, batch_size=4, tokens=2048) == (16, 16)
    assert chunk_h.resolve_chunk_h_tiles(0, 32, batch_size=1, tokens=2048) == (32, 32)
    assert chunk_h.resolve_chunk_h_tiles(0, 16, batch_size=2, tokens=2048) == (32, 16)


def test_chunk_h_tile_environment_override_wins(monkeypatch):
    pytest.importorskip("triton")
    import rwkv7_hf.self_chunk_h_fwd as chunk_h

    monkeypatch.setattr(
        chunk_h,
        "check_shared_mem",
        lambda architecture, _device=None: architecture == "ampere",
    )
    monkeypatch.setattr(chunk_h.torch.cuda, "get_device_name", lambda _device=None: "NVIDIA GeForce RTX 3090")
    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_SELF_CHUNK_H_BV", "64")
    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_SELF_CHUNK_H_BC", "32")

    assert chunk_h.resolve_chunk_h_tiles(0, 16, batch_size=4, tokens=2048) == (64, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/Triton kernel test")
def test_self_chunk_matches_recurrent_reference():
    from rwkv7_hf.fused_recurrent_update import fused_recurrent_scan

    torch.manual_seed(9)
    shape = (1, 32, 2, 64)
    r = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn_like(r) * 0.1
    v = torch.randn_like(r) * 0.1
    kk = torch.nn.functional.normalize(torch.randn_like(r).float(), dim=-1).half()
    a = torch.sigmoid(torch.randn_like(r))
    w = torch.exp(-0.606531 * torch.sigmoid(torch.randn_like(r).float()))
    state = torch.randn(1, 2, 64, 64, device="cuda") * 0.01

    expected, expected_state = fused_recurrent_scan(
        r, w, k, v, kk, a, state, block_n=64, block_m=8, num_warps=4
    )
    for chunk_size in (16, 32):
        actual, actual_state = self_chunk_rwkv7(r, w, k, v, kk, a, state, chunk_size=chunk_size)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-3, atol=3e-4)
        torch.testing.assert_close(actual_state, expected_state, rtol=2e-3, atol=3e-4)
