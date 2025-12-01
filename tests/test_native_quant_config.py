#!/usr/bin/env python3
# coding=utf-8
"""Config-driven native MM8/MM4 quantization smoke.

The Apple / no-bitsandbytes lane uses the native PyTorch backend, so persisted
``use_native_mm8`` / ``use_native_mm4`` flags must work for
``NativeRWKV7ForCausalLM.from_pretrained`` and must keep decode/generate on the
module-call path when layer linears are packed.
"""
from __future__ import annotations

import shutil
import tempfile

import torch
import torch.nn.functional as F

from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM
from rwkv7_hf.native_quant_mm4 import native_mm4_group_size_for_module


def build_tiny_config(*, quantization: str | None = None, policy: str = "memory") -> NativeRWKV7Config:
    return NativeRWKV7Config(
        vocab_size=41,
        hidden_size=16,
        num_hidden_layers=2,
        head_dim=4,
        intermediate_size=32,
        decay_low_rank_dim=4,
        gate_low_rank_dim=4,
        a_low_rank_dim=4,
        v_low_rank_dim=4,
        use_cache=True,
        use_native_mm8=quantization == "mm8",
        use_native_mm4=quantization == "mm4",
        native_mm8_min_params=1,
        native_mm4_min_params=1,
        native_mm8_policy=policy,
        native_mm4_policy=policy,
    )


def assert_quantized_roundtrip(quantization: str, class_name: str) -> None:
    torch.manual_seed(20260704)
    dense = NativeRWKV7ForCausalLM(build_tiny_config()).eval()
    source = NativeRWKV7ForCausalLM(build_tiny_config(quantization=quantization)).eval()
    source.load_state_dict(dense.state_dict())

    tmp = tempfile.mkdtemp(prefix=f"native_{quantization}_cfg_")
    try:
        source.save_pretrained(tmp)
        reloaded = NativeRWKV7ForCausalLM.from_pretrained(tmp).eval()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    replaced = int(getattr(reloaded, "_rwkv7_native_mm_replaced_modules", 0))
    count = sum(1 for module in reloaded.modules() if type(module).__name__ == class_name)
    assert getattr(reloaded, "_rwkv7_native_mm_quantization", None) == quantization
    assert replaced == count
    assert count >= 1
    assert reloaded._native_model_quantized()

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        ref = dense(input_ids, use_cache=True)
        got = reloaded(input_ids, use_cache=True)
        next_id = got.logits[:, -1:].argmax(dim=-1)
        dec = reloaded(next_id, past_key_values=got.past_key_values, use_cache=True)
        gen = reloaded.generate(
            input_ids,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
        )
    assert got.logits.shape == ref.logits.shape
    assert dec.logits.shape == (1, 1, reloaded.config.vocab_size)
    assert gen.shape == (1, input_ids.shape[1] + 2)
    assert torch.isfinite(got.logits).all()
    assert torch.isfinite(dec.logits).all()
    assert reloaded.rwkv7_native_model_last_decode_backend() == "eager"

    cos = F.cosine_similarity(ref.logits.flatten().float(), got.logits.flatten().float(), dim=0).item()
    floor = 0.95 if quantization == "mm8" else 0.70
    assert cos >= floor, (quantization, cos, floor)


def test_native_mm8_config_roundtrip() -> None:
    assert_quantized_roundtrip("mm8", "MM8Linear")


def test_native_mm4_config_roundtrip() -> None:
    assert_quantized_roundtrip("mm4", "MM4Linear")


def test_native_mm8_mm4_are_mutually_exclusive() -> None:
    cfg = build_tiny_config()
    cfg.use_native_mm8 = True
    cfg.use_native_mm4 = True
    model = NativeRWKV7ForCausalLM(cfg)
    try:
        model.apply_native_mm_quantization_from_config()
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected mutually-exclusive native quant config to fail")


def test_native_mm4_group_size_persists_in_config() -> None:
    for group_size in (128, 256):
        config = build_tiny_config(quantization="mm4")
        config.native_mm4_group_size = group_size
        config.native_mm4_group_policy = "lm_head"
        restored = NativeRWKV7Config.from_dict(config.to_dict())
        assert restored.native_mm4_group_size == group_size
        assert restored.native_mm4_group_policy == "lm_head"


def test_native_mm4_group_policy_selects_only_requested_modules() -> None:
    assert native_mm4_group_size_for_module("lm_head", 128, "lm_head") == 128
    assert (
        native_mm4_group_size_for_module(
            "model.layers.0.ffn.key", 128, "lm_head"
        )
        == 0
    )
    assert (
        native_mm4_group_size_for_module(
            "model.layers.0.ffn.key", 128, "lm_head_and_key"
        )
        == 128
    )
    assert (
        native_mm4_group_size_for_module(
            "model.layers.0.ffn.value", 128, "lm_head_and_key"
        )
        == 0
    )


def test_native_mm_speed_policy_quantizes_lm_head_only() -> None:
    for quantization, class_name in [("mm8", "MM8Linear"), ("mm4", "MM4Linear")]:
        model = NativeRWKV7ForCausalLM(build_tiny_config(quantization=quantization, policy="speed")).eval()
        replaced = model.apply_native_mm_quantization_from_config()
        count = sum(1 for module in model.modules() if type(module).__name__ == class_name)
        assert replaced == 1
        assert count == 1
        assert type(model.lm_head).__name__ == class_name


def main() -> int:
    test_native_mm8_config_roundtrip()
    test_native_mm4_config_roundtrip()
    test_native_mm8_mm4_are_mutually_exclusive()
    test_native_mm_speed_policy_quantizes_lm_head_only()
    print("NATIVE QUANT CONFIG PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
