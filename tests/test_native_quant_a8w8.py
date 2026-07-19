from __future__ import annotations

import pytest
import torch

from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM
from rwkv7_hf.native_quant_a8w8 import A8W8Linear, quantize_model_a8w8


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 16, bias=False)
        self.wide = torch.nn.Linear(4097, 16, bias=False)
        self.lm_head = torch.nn.Linear(16, 32, bias=False)


def test_cpu_fallback_tracks_dense_linear() -> None:
    torch.manual_seed(3)
    dense = torch.nn.Linear(32, 48, bias=True)
    quantized = A8W8Linear(dense)
    x = torch.randn(3, 32)
    reference = dense(x)
    actual = quantized(x)
    cosine = torch.nn.functional.cosine_similarity(
        reference.flatten().unsqueeze(0), actual.flatten().unsqueeze(0)
    ).item()
    assert cosine >= 0.9999

    output = torch.empty_like(reference)
    returned = quantized.rwkv7_forward_into(x, output)
    assert returned.data_ptr() == output.data_ptr()
    assert torch.equal(returned, output)


def test_cpu_fallback_tracks_dense_linear_for_strided_last_token_slice() -> None:
    torch.manual_seed(4)
    dense = torch.nn.Linear(32, 48, bias=False)
    quantized = A8W8Linear(dense)
    x = torch.randn(3, 11, 32)[:, -1:, :]
    assert not x.is_contiguous()
    reference = dense(x)
    actual = quantized(x)
    cosine = torch.nn.functional.cosine_similarity(
        reference.flatten().unsqueeze(0), actual.flatten().unsqueeze(0)
    ).item()
    assert cosine >= 0.9999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch_size", [2, 4, 8, 17])
def test_cuda_kernel_tracks_strided_last_token_slice(batch_size: int) -> None:
    torch.manual_seed(5)
    dense = torch.nn.Linear(1024, 2048, bias=False, dtype=torch.float16, device="cuda")
    quantized = A8W8Linear(dense)
    x = torch.randn(batch_size, 11, 1024, dtype=torch.float16, device="cuda")[:, -1:, :]
    assert not x.is_contiguous()
    reference = dense(x).float()
    actual = quantized(x).float()
    cosine = torch.nn.functional.cosine_similarity(
        reference.flatten().unsqueeze(0), actual.flatten().unsqueeze(0)
    ).item()
    assert cosine >= 0.999


def test_speed_policy_replaces_head_only() -> None:
    model = TinyLM()
    replaced = quantize_model_a8w8(model, min_params=1, policy="speed")
    assert replaced == 1
    assert isinstance(model.lm_head, A8W8Linear)
    assert isinstance(model.proj, torch.nn.Linear)
    assert model._rwkv7_native_mm_quantization == "a8w8"
    assert model._rwkv7_native_mm_quantized_head is True
    assert model._rwkv7_native_mm_block_replaced_modules == 0


def test_memory_policy_replaces_all_size_gated_linears() -> None:
    model = TinyLM()
    replaced = quantize_model_a8w8(model, min_params=1, policy="memory")
    assert replaced == 2
    assert isinstance(model.proj, A8W8Linear)
    assert isinstance(model.wide, torch.nn.Linear)
    assert isinstance(model.lm_head, A8W8Linear)


def test_native_mm_modules_require_graph_operand_packs() -> None:
    class Dummy:
        _rwkv7_native_mm_quantization = "a8w8"
        _rwkv7_native_mm_replaced_modules = 2
        _rwkv7_native_mm_block_replaced_modules = 2

        def _rwkv7_uses_external_quantization(self) -> bool:
            return False

        _rwkv7_uses_native_quant_operands = RWKV7ForCausalLM._rwkv7_uses_native_quant_operands

    dummy = Dummy()
    assert RWKV7ForCausalLM._rwkv7_uses_native_quant_operands(dummy)
    assert RWKV7ForCausalLM._rwkv7_uses_quantized_linear_operands(dummy)

    dummy._rwkv7_native_mm_block_replaced_modules = 0
    assert not RWKV7ForCausalLM._rwkv7_uses_native_quant_operands(dummy)
    assert not RWKV7ForCausalLM._rwkv7_uses_quantized_linear_operands(dummy)
