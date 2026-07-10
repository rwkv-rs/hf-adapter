from __future__ import annotations

import importlib
from pathlib import Path
import sys

import torch

from rwkv7_hf.cache_rwkv7 import RWKV7Cache
from rwkv7_hf.modeling_rwkv7 import RWKV7Model


ROOT = Path(__file__).resolve().parents[1]


def _load_dense_backend(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "kernels"))
    for name in tuple(sys.modules):
        if name == "rwkv7_kernels" or name.startswith("rwkv7_kernels."):
            sys.modules.pop(name)
    return importlib.import_module("rwkv7_kernels.model.dense")


def test_migrated_dense_model_math_cache_padding_and_hidden_states(
    tiny_config, monkeypatch
):
    dense = _load_dense_backend(monkeypatch)
    torch.manual_seed(107)
    model = RWKV7Model(tiny_config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.03)

    ids = torch.tensor([[3, 5, 7, 0], [0, 11, 13, 17]])
    mask = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=torch.bool)
    with torch.inference_mode():
        expected = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=True,
            output_hidden_states=True,
        )
        hidden = model.embeddings(ids) * mask.unsqueeze(-1)
        actual = dense.run_base_model(
            model,
            {
                "model_kind": "base",
                "hidden_states": hidden,
                "attention_mask": mask,
                "past_key_values": RWKV7Cache(
                    num_layers=tiny_config.num_hidden_layers
                ),
                "training": False,
                "use_cache": True,
                "output_hidden_states": True,
            },
        )

    torch.testing.assert_close(
        actual["last_hidden_state"],
        expected.last_hidden_state,
        rtol=2e-5,
        atol=2e-6,
    )
    assert len(actual["hidden_states"]) == len(expected.hidden_states)
    for migrated, reference in zip(actual["hidden_states"], expected.hidden_states):
        torch.testing.assert_close(migrated, reference, rtol=2e-5, atol=2e-6)
    assert actual["past_key_values"].seen_tokens == ids.shape[1]
    for migrated, reference in zip(
        actual["past_key_values"].recurrent_state,
        expected.past_key_values.recurrent_state,
    ):
        # This assertion also catches an accidental [V,K] ABI leak.
        torch.testing.assert_close(migrated, reference, rtol=2e-5, atol=2e-6)
