from __future__ import annotations

from types import SimpleNamespace

from bench.bench_chunked_prefill import model_metadata as chunk_metadata
from bench.bench_native_graph_fused_output import model_metadata as output_metadata
from bench.bench_native_graph_fused_recurrent_output import (
    model_metadata as recurrent_output_metadata,
)


def test_shared_matrix_rows_include_model_identity() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=24,
            head_dim=64,
            num_heads=32,
        )
    )
    for helper in (chunk_metadata, output_metadata, recurrent_output_metadata):
        row = helper("/models/rwkv7-g1g-1.5b-hf", model)
        assert row == {
            "model_name": "rwkv7-g1g-1.5b-hf",
            "model_size_label": "1.5b",
            "hf_model_dir": "/models/rwkv7-g1g-1.5b-hf",
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 24,
            "head_dim": 64,
            "num_heads": 32,
        }
