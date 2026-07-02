#!/usr/bin/env python3
from __future__ import annotations

from rwkv7_hf import RWKV7Config, RWKV7ForCausalLM


def main() -> int:
    cfg = RWKV7Config(num_hidden_layers=2)

    memory = RWKV7ForCausalLM.rwkv7_bnb_skip_modules("memory", cfg)
    assert "lm_head" in memory
    assert r".*_lora\.lora\.[02]" in memory
    assert "model.layers.0.attn.w_lora.lora.0" in memory
    assert "model.layers.1.attn.v_lora.lora.2" in memory
    assert "model.layers.0.attn.r_proj" not in memory

    decode_hot = RWKV7ForCausalLM.rwkv7_bnb_skip_modules("decode_hot", cfg)
    assert r".*attn\.(r_proj|k_proj|v_proj|o_proj)" in decode_hot
    assert "model.layers.0.attn.r_proj" in decode_hot
    assert "model.layers.1.attn.o_proj" in decode_hot
    assert "model.layers.0.ffn.key" not in decode_hot

    dense = RWKV7ForCausalLM.rwkv7_bnb_skip_modules("dense", cfg)
    assert r".*ffn\.(key|value)" in dense
    assert "model.layers.0.ffn.key" in dense
    assert "model.layers.1.ffn.value" in dense

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
