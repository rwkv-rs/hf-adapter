from __future__ import annotations

import pytest
import torch

from scripts.compare_official_prefill_self_repeats import compare_repeats


def make_capture(*, delta: float = 0.0, prompt_tokens: int = 8) -> dict:
    base = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    changed = base.clone()
    changed[0, 1] += delta
    state = torch.ones(1, 1, 1, 2, 2, dtype=torch.float16)
    return {
        "engine": "official_v3a_sequence",
        "source_revision": "pinned",
        "precision": "fp16_state_fp16_io",
        "batch_size": 1,
        "prompt_tokens": prompt_tokens,
        "prompt_ids": torch.arange(prompt_tokens).view(1, -1),
        "logits": changed,
        "first_decode_logits": changed,
        "layer_outputs": changed,
        "first_token": torch.tensor([[1]]),
        "first_decode_token": torch.tensor([1]),
        "prefill": {
            "state": state.clone(),
            "xpa": base.clone(),
            "xpf": base.clone(),
        },
    }


def test_prefill_self_repeat_builds_shape_bound_envelope() -> None:
    report = compare_repeats(
        make_capture(),
        [make_capture(delta=0.125), make_capture(delta=0.0625)],
    )

    assert report["status"] == "pass"
    assert report["axis"] == "official_prefill_self_repeat"
    assert report["repetitions"] == 3
    assert report["envelope"]["first_decode_logits"]["max_abs"] == 0.125
    assert report["envelope"]["first_decode_logits"]["total"] == 2


def test_prefill_self_repeat_rejects_a_different_shape() -> None:
    with pytest.raises(ValueError, match="prompt_tokens"):
        compare_repeats(make_capture(), [make_capture(prompt_tokens=16)])
