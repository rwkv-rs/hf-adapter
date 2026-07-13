import torch

from rwkv7_hf.native_quant_mm4 import MM4Linear
from rwkv7_hf.native_quant_mm8 import MM8Linear
from rwkv7_hf.native_quant_q4km import (
    native_q4km_bits_for_module,
    quantize_model_q4km,
)


def test_native_q4km_sensitive_module_policy():
    assert native_q4km_bits_for_module("model.layers.3.ffn.key") == 4
    assert native_q4km_bits_for_module("model.layers.3.ffn.value") == 8
    assert native_q4km_bits_for_module("model.layers.3.attn.r_proj") == 8
    assert native_q4km_bits_for_module("model.layers.3.attn.v_proj") == 8
    assert native_q4km_bits_for_module("model.layers.3.attn.k_proj") == 4
    assert native_q4km_bits_for_module("lm_head") == 8


class TinyQ4KMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = torch.nn.Module()
        self.ffn.key = torch.nn.Linear(64, 128, bias=False)
        self.ffn.value = torch.nn.Linear(128, 64, bias=False)
        self.lm_head = torch.nn.Linear(64, 256, bias=False)


def test_native_q4km_replaces_w4_and_sensitive_w8_modules():
    model = TinyQ4KMModel()
    replaced = quantize_model_q4km(model, min_params=0)
    assert replaced == 3
    assert isinstance(model.ffn.key, MM4Linear)
    assert isinstance(model.ffn.value, MM8Linear)
    assert isinstance(model.lm_head, MM8Linear)
    assert model._rwkv7_native_mm_quantization == "mm4_q4km"
    assert model._rwkv7_native_mm_bits_histogram == {"4": 1, "8": 2}
