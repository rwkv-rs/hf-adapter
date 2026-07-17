from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from rwkv7_hf.train_temp_alignment import (
    build_train_temp_param_groups,
    compare_tensors,
    train_temp_cross_entropy,
)


def test_l2wrap_adds_official_argmax_gradient() -> None:
    logits = torch.tensor(
        [[[1.0, 2.0, -1.0], [3.0, 0.5, -2.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor([[0, 2]], dtype=torch.long)

    base_logits = logits.detach().clone().requires_grad_(True)
    base_loss = F.cross_entropy(base_logits.view(-1, 3), targets.view(-1))
    base_loss.backward()

    loss = train_temp_cross_entropy(logits, targets)
    loss.backward()

    expected_extra = torch.zeros_like(logits)
    expected_extra[0, 0, 1] = 2.0 * 1.0e-4 / 2
    expected_extra[0, 1, 0] = 3.0 * 1.0e-4 / 2
    torch.testing.assert_close(loss, base_loss)
    torch.testing.assert_close(logits.grad, base_logits.grad + expected_extra)


def test_l2wrap_respects_upstream_loss_scale() -> None:
    logits = torch.tensor([[[0.25, 1.5]]], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[0]], dtype=torch.long)
    (train_temp_cross_entropy(logits, targets) * 3.0).backward()

    base_logits = logits.detach().clone().requires_grad_(True)
    (F.cross_entropy(base_logits.view(-1, 2), targets.view(-1)) * 3.0).backward()
    expected = torch.zeros_like(logits)
    expected[0, 0, 1] = 3.0 * 1.5e-4
    torch.testing.assert_close(logits.grad, base_logits.grad + expected)


def test_low_precision_cross_entropy_accumulates_and_returns_fp32() -> None:
    logits = torch.tensor(
        [[[1.0, 2.0, -1.0], [3.0, 0.5, -2.0]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    targets = torch.tensor([[0, 2]], dtype=torch.long)

    loss = train_temp_cross_entropy(logits, targets)
    loss.backward()

    reference_logits = logits.detach().clone().requires_grad_(True)
    reference_loss = F.cross_entropy(
        reference_logits.float().view(-1, 3),
        targets.view(-1),
    )
    reference_loss.backward()
    expected_extra = torch.zeros_like(logits)
    expected_extra[0, 0, 1] = 2.0 * 1.0e-4 / 2
    expected_extra[0, 1, 0] = 3.0 * 1.0e-4 / 2

    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(logits.grad, reference_logits.grad + expected_extra)


def _parameter(*shape: int) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.zeros(shape))


def _group_names(groups: list[dict]) -> dict[str, set[str]]:
    return {str(group["group_name"]): set(group["param_names"]) for group in groups}


def test_official_parameter_groups_match_train_temp_recipe() -> None:
    named = [
        ("blocks.0.att.w0", _parameter(1, 1, 8)),
        ("blocks.0.att.key.weight", _parameter(8, 8)),
        ("blocks.0.ln1.weight", _parameter(8)),
        ("blocks.0.att.x_r", _parameter(1, 1, 8)),
    ]
    groups = build_train_temp_param_groups(named, weight_decay=0.001, naming="official")
    names = _group_names(groups)

    assert names["lr_2x"] == {"blocks.0.att.w0"}
    assert names["decay"] == {"blocks.0.att.key.weight"}
    assert names["lr_1x"] == {"blocks.0.att.x_r", "blocks.0.ln1.weight"}
    assert {group["group_name"]: group["my_lr_scale"] for group in groups} == {
        "lr_1x": 1.0,
        "lr_2x": 2.0,
        "decay": 1.0,
    }


def test_hf_parameter_groups_preserve_translated_w0_semantics() -> None:
    named = [
        ("model.layers.0.attn.w_lora.lora.2.bias", _parameter(8)),
        ("model.layers.0.attn.w_lora.lora.0.weight", _parameter(8, 4)),
        ("model.layers.0.attn.w_lora.lora.2.weight", _parameter(4, 8)),
        ("model.layers.0.attn.a_lora.lora.0.weight", _parameter(8, 4)),
        ("model.layers.0.attn.a_lora.lora.2.weight", _parameter(4, 8)),
        ("model.layers.0.attn.g_lora.lora.0.weight", _parameter(8, 4)),
        ("model.layers.0.attn.g_lora.lora.2.weight", _parameter(4, 8)),
        ("model.layers.1.attn.v_lora.lora.0.weight", _parameter(8, 4)),
        ("model.layers.1.attn.v_lora.lora.2.weight", _parameter(4, 8)),
        ("model.layers.0.attn.g_norm.weight", _parameter(8)),
        ("model.layers.0.ffn.value.weight", _parameter(8, 16)),
    ]
    groups = build_train_temp_param_groups(named, weight_decay=0.001, naming="hf")
    names = _group_names(groups)

    assert names["lr_2x"] == {"model.layers.0.attn.w_lora.lora.2.bias"}
    assert names["decay"] == {"model.layers.0.ffn.value.weight"}
    assert names["lr_1x"] == {
        "model.layers.0.attn.w_lora.lora.0.weight",
        "model.layers.0.attn.w_lora.lora.2.weight",
        "model.layers.0.attn.a_lora.lora.0.weight",
        "model.layers.0.attn.a_lora.lora.2.weight",
        "model.layers.0.attn.g_lora.lora.0.weight",
        "model.layers.0.attn.g_lora.lora.2.weight",
        "model.layers.1.attn.v_lora.lora.0.weight",
        "model.layers.1.attn.v_lora.lora.2.weight",
        "model.layers.0.attn.g_norm.weight",
    }


def test_frozen_parameters_are_not_grouped() -> None:
    frozen = _parameter(4, 4)
    frozen.requires_grad_(False)
    groups = build_train_temp_param_groups(
        [("blocks.0.att.key.weight", frozen)],
        weight_decay=0.001,
        naming="official",
    )
    assert groups == []


def test_tensor_comparison_reports_shape_finite_and_error_metrics() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0])
    candidate = torch.tensor([1.0, 2.0, 2.5])
    metrics = compare_tensors(reference, candidate)

    assert metrics["shape_match"] is True
    assert metrics["finite"] is True
    assert math.isclose(metrics["max_abs"], 0.5)
    assert 0.0 < metrics["relative_l2"] < 1.0
    assert 0.99 < metrics["cosine"] < 1.0

    mismatch = compare_tensors(reference, candidate.view(1, 3))
    assert mismatch["shape_match"] is False
    assert mismatch["comparable"] is False

    nonfinite = compare_tensors(reference, torch.tensor([1.0, float("nan"), 3.0]))
    assert nonfinite["finite"] is False
    assert nonfinite["comparable"] is False
