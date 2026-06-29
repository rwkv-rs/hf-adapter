from __future__ import annotations

import torch

from rwkv7_hf.cache_rwkv7 import RWKV7Cache


def make_cache():
    return RWKV7Cache(
        [torch.arange(3 * 2 * 4 * 4).view(3, 2, 4, 4).float()],
        [torch.arange(3 * 8).view(3, 8).float()],
        [torch.arange(3 * 8).view(3, 8).float() + 100],
        seen_tokens=9,
    )


def test_cache_select_repeat_clone_detach_and_reset():
    cache = make_cache()
    clone = cache.clone().select_batch(torch.tensor([2, 0]), inplace=False)
    assert clone.get_batch_size() == 2
    assert cache.get_batch_size() == 3
    torch.testing.assert_close(clone.attention_shift[0][0], cache.attention_shift[0][2])
    clone.batch_repeat_interleave(2)
    assert clone.get_batch_size() == 4
    clone.detach().to(dtype=torch.float64)
    assert clone.recurrent_state[0].dtype == torch.float64
    clone.reorder_cache(torch.tensor([3, 1]))
    assert clone.get_batch_size() == 2
    clone.reset()
    assert clone.get_seq_length() == 0
    assert not clone.is_initialized()


def test_legacy_roundtrip_has_no_v_first():
    cache = make_cache()
    legacy = cache.to_legacy_cache()
    assert len(legacy) == 3
    restored = RWKV7Cache.from_legacy_cache(legacy, seen_tokens=9)
    assert not hasattr(restored, "v_first")
    torch.testing.assert_close(restored.recurrent_state[0], cache.recurrent_state[0])
