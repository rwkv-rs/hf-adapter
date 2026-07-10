from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NVIDIA = ROOT / "kernels" / "rwkv7_kernels" / "nvidia"


def test_nvidia_migration_manifest_is_complete_and_byte_verified():
    manifest = json.loads((NVIDIA / "MIGRATION_MANIFEST.json").read_text())
    assert manifest["schema"] == "rwkv7-nvidia-source-migration-v1"
    assert manifest["source_branch"] == "perf/native-kernels-v0.8"
    assert len(manifest["files"]) == 99

    destinations = set()
    for entry in manifest["files"]:
        destination = ROOT / entry["destination"]
        assert destination.is_file(), entry["destination"]
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == entry[
            "destination_sha256"
        ]
        assert len(entry["git_blob"]) == 40
        destinations.add(destination.name)

    required_families = {
        "fused_attention_projection.py",
        "fused_decode_norm_mix.py",
        "fused_ffn.py",
        "fused_lora.py",
        "fused_output.py",
        "fused_prefill.py",
        "fused_recurrent_update.py",
        "dplr_prefill.py",
        "self_chunk_rwkv7.py",
        "sm70_linear.py",
        "sm70_quant.py",
        "sm70_wagv.py",
        "ada_lora.py",
        "ada_sparse_ffn.py",
        "blackwell_norm_mix.py",
        "native_quant_mm4.py",
        "native_quant_mm8.py",
        "native_quant_a8w8.py",
        "native_quant_bnb8.py",
        "native_quant_marlin.py",
        "native_quant_torchao.py",
        "native_jit.py",
        "native_jit_decode.py",
        "native_jit_packing.py",
        "native_graph_runtime.py",
        "recurrent_state.py",
        "train_temp_cuda.py",
    }
    assert required_families <= destinations


def test_nvidia_sources_do_not_reintroduce_model_config_or_cache_ownership():
    forbidden_names = {
        "modeling_rwkv7.py",
        "native_model.py",
        "model_cache.py",
        "model_config.py",
    }
    assert not forbidden_names.intersection(path.name for path in NVIDIA.rglob("*"))

    for path in NVIDIA.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not name.name.startswith("rwkv7_hf") for name in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("rwkv7_hf")
