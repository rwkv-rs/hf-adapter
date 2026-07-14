from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from rwkv7_hf.native_jit import _native_prefill_stacked_rkv_weights


def _pack(r, k, v):
    values = [None] * 42
    values[20], values[21], values[22] = r, k, v
    return tuple(values)


def test_stacked_rkv_cache_matches_three_linear_weights_and_invalidates():
    torch.manual_seed(3)
    r = torch.randn(4, 4)
    k = torch.randn(4, 4)
    v = torch.randn(4, 4)
    owner = SimpleNamespace()

    first = _native_prefill_stacked_rkv_weights(owner, [_pack(r, k, v)])
    assert tuple(first[0].shape) == (3, 4, 4)
    torch.testing.assert_close(first[0], torch.stack((r.t(), k.t(), v.t())))

    second = _native_prefill_stacked_rkv_weights(owner, [_pack(r, k, v)])
    assert second[0] is first[0]

    r.add_(1.0)
    rebuilt = _native_prefill_stacked_rkv_weights(owner, [_pack(r, k, v)])
    assert rebuilt[0] is not first[0]
    torch.testing.assert_close(rebuilt[0][0], r.t())
