import torch
import torch.nn as nn

from rwkv7_hf.ascend_quant import AscendW8A16Linear, ascend_w8a16_decision


def test_w8_linear_cpu_oracle_and_payload():
    torch.manual_seed(1)
    dense = nn.Linear(32, 64, bias=False, dtype=torch.float16)
    quant = AscendW8A16Linear.from_float(dense, chunk_rows=7)
    x = torch.randn(5, 32, dtype=torch.float16)
    ref = dense(x).float()
    out = quant(x).float()
    cosine = torch.nn.functional.cosine_similarity(ref.flatten(), out.flatten(), dim=0)
    assert float(cosine.detach()) > 0.9999
    assert quant.packed_bytes < quant.dense_fp16_bytes
    assert not any(True for _ in quant.parameters())


def test_speed_policy_is_exact_shape_role_card_and_rows(monkeypatch):
    monkeypatch.delenv("RWKV7_ALLOW_UNVALIDATED_ASCEND", raising=False)
    exact_stack = dict(
        device_name="Ascend910B3",
        torch_version="2.9.0+cpu",
        torch_npu_version="2.9.0",
        cann_version="8.5.0",
    )
    rejected = ascend_w8a16_decision(
        "model.layers.0.ffn.value", 16384, 4096,
        **exact_stack, rows=8, dtype=torch.bfloat16, policy="speed",
    )
    assert not rejected.enabled and not rejected.speed_validated
    candidate = ascend_w8a16_decision(
        "model.layers.0.ffn.value", 16384, 4096,
        **exact_stack, rows=8, dtype=torch.bfloat16, policy="candidate",
    )
    assert candidate.enabled and not candidate.speed_validated and candidate.stack_validated
    assert not ascend_w8a16_decision(
        "model.layers.0.att.key", 4096, 16384,
        device_name="Ascend910B3", rows=8, policy="speed",
    ).enabled
    assert not ascend_w8a16_decision(
        "model.layers.0.ffn.key", 4096, 16384,
        device_name="Ascend910B3", rows=8, dtype=torch.bfloat16, policy="speed",
    ).enabled
    assert ascend_w8a16_decision(
        "model.layers.0.ffn.key", 4096, 16384,
        **exact_stack, rows=8, dtype=torch.bfloat16, policy="memory",
    ).enabled
    assert not ascend_w8a16_decision(
        "model.layers.0.ffn.key", 2048, 8192,
        device_name="Ascend910B3", rows=8, policy="speed",
    ).enabled
    assert not ascend_w8a16_decision(
        "model.layers.0.ffn.key", 4096, 16384,
        device_name="Ascend910B2C", rows=8, policy="speed",
    ).enabled
    unmeasured = ascend_w8a16_decision(
        "model.layers.0.ffn.value", 16384, 4096,
        **exact_stack, rows=128, policy="candidate",
    )
    assert unmeasured.enabled and not unmeasured.speed_validated
    spoofed = ascend_w8a16_decision(
        "model.layers.0.ffn.value", 16384, 4096,
        **{**exact_stack, "device_name": "FakeAscend910B3X"}, policy="candidate",
    )
    assert not spoofed.enabled and not spoofed.stack_validated
    wrong_stack = ascend_w8a16_decision(
        "model.layers.0.ffn.value", 16384, 4096,
        **{**exact_stack, "torch_npu_version": "2.9.0.post1"}, policy="candidate",
    )
    assert not wrong_stack.enabled
    monkeypatch.setenv("RWKV7_ALLOW_UNVALIDATED_ASCEND", "1")
    overridden = ascend_w8a16_decision(
        "model.layers.0.ffn.value", 16384, 4096,
        **{**exact_stack, "device_name": "Ascend910B4"}, policy="candidate",
    )
    assert overridden.enabled and not overridden.stack_validated
    assert "unvalidated" in overridden.reason


def test_w4_candidate_is_explicit_and_packed(tmp_path):
    from rwkv7_hf.ascend_quant_w4 import (
        AscendWeightOnlyLinear,
        load_quantized_linear,
        save_quantized_linear,
    )

    torch.manual_seed(3)
    dense = nn.Linear(32, 64, bias=False, dtype=torch.float16)
    quant = AscendWeightOnlyLinear.from_float(
        dense, bit=4, group_size=16, enforce_verified_shape=False
    )
    x = torch.randn(3, 32, dtype=torch.float16)
    cosine = torch.nn.functional.cosine_similarity(
        dense(x).float().flatten(), quant(x).float().flatten(), dim=0
    )
    assert float(cosine.detach()) > 0.98
    assert quant.packed_weight_bytes() < dense.weight.numel() * dense.weight.element_size()
    save_quantized_linear(quant, tmp_path)
    restored = load_quantized_linear(tmp_path, enforce_verified_shape=False)
    assert torch.equal(restored.qweight, quant.qweight)
    assert torch.equal(restored.scales, quant.scales)


def test_w4_raw_candidate_is_not_a_production_promotion():
    from rwkv7_hf.ascend_quant_w4 import raw_candidate_supported, should_quantize

    exact = dict(
        in_features=4096,
        out_features=16384,
        batch=4,
        bit=4,
        dtype=torch.float16,
        device_name="Ascend910B3",
        cann_version="8.5.0",
        torch_version="2.9.0+cpu",
        torch_npu_version="2.9.0",
    )
    assert raw_candidate_supported(**exact)
    assert not should_quantize(**exact)
    assert not raw_candidate_supported(**{**exact, "cann_version": "8.4.0"})
    assert not raw_candidate_supported(**{**exact, "torch_npu_version": "2.9.0.post1"})
    assert not raw_candidate_supported(**{**exact, "torch_version": "2.9.0"})
    assert not raw_candidate_supported(**{**exact, "device_name": "Ascend910B2C"})
    assert not raw_candidate_supported(**{**exact, "device_name": "FakeAscend910B3X"})
    assert not raw_candidate_supported(**{**exact, "dtype": torch.bfloat16})
    assert not raw_candidate_supported(**{**exact, "batch": 64})
    for device in ("Ascend310P3", "Ascend910A", "Ascend910B1", "Ascend910B4"):
        assert not should_quantize(**{**exact, "device_name": device})


def test_w4_bf16_conversion_rejects_before_mutation(monkeypatch):
    import rwkv7_hf.ascend_quant_w4 as quant_w4

    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.key = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
            self.value = nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = FFN()

    monkeypatch.setattr(
        quant_w4,
        "RAW_CANDIDATE_FFN_SHAPES",
        ((8, 16), (16, 8)),
    )
    model = Model()
    key, value = model.ffn.key, model.ffn.value
    try:
        quant_w4.quantize_ascend_w4a16_candidate(
            model,
            group_size=8,
            require_explicit_candidate=False,
        )
    except TypeError as exc:
        assert "FP16-only" in str(exc) and "made no changes" in str(exc)
    else:  # pragma: no cover - the fail-closed behavior is mandatory
        raise AssertionError("BF16 W4 conversion was unexpectedly accepted")
    assert model.ffn.key is key and model.ffn.value is value

    direct = quant_w4.AscendWeightOnlyLinear(8, 16, bit=4, group_size=8)
    try:
        direct.load_fp_weight(key.weight)
    except TypeError as exc:
        assert "BF16" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("BF16 W4 packing was unexpectedly accepted")
