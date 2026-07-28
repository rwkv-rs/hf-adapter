from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from scripts.audit_release_wheels import historical_tree_oid


ROOT = Path(__file__).resolve().parents[1]
NVIDIA = ROOT / "kernels" / "rwkv7_kernels" / "nvidia"


def test_nvidia_migration_manifest_is_complete_and_byte_verified():
    manifest = json.loads((NVIDIA / "MIGRATION_MANIFEST.json").read_text())
    assert manifest["schema"] == "rwkv7-nvidia-source-migration-v1"
    assert manifest["source_branch"] == "perf/native-kernels-v0.8"
    assert len(manifest["files"]) == 102

    destinations = set()
    transfers = {"byte_identical": 0, "adapted_clean_boundary": 0}
    for entry in manifest["files"]:
        destination = ROOT / entry["destination"]
        assert destination.is_file(), entry["destination"]
        payload = destination.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["destination_sha256"]
        transfer = entry["transfer"]
        transfers[transfer] += 1
        if transfer == "byte_identical":
            header = b"blob " + str(len(payload)).encode() + b"\0"
            assert (
                hashlib.sha1(  # noqa: S324 - verifies historical Git identity.
                    header + payload,
                    usedforsecurity=False,
                ).hexdigest()
                == entry["git_blob"]
            )
        else:
            assert entry["source"] in {
                "rwkv7_hf/extension_build.py",
                "rwkv7_hf/csrc/train_temp/rwkv7_clampw_v3.cpp",
                "rwkv7_hf/csrc/train_temp/rwkv7_clampw_v3_for_h100.cu",
                "rwkv7_hf/csrc/train_temp/rwkv7_cmix_bf16_v5.cu",
                "rwkv7_hf/csrc/train_temp/rwkv7_tmix_kk_pre_bf16_v5.cu",
                "rwkv7_hf/csrc/train_temp/rwkv7_tmix_mix6_bf16_v5.cpp",
                "rwkv7_hf/csrc/train_temp/rwkv7_tmix_mix6_bf16_v5.cu",
                "rwkv7_hf/fused_prefill.py",
                "rwkv7_hf/kernel_policy.py",
                "rwkv7_hf/native_graph_runtime.py",
                "rwkv7_hf/native_jit_decode.py",
                "rwkv7_hf/native_jit_linear.py",
                "rwkv7_hf/native_jit_packing.py",
                "rwkv7_hf/native_jit_prefill.py",
                "rwkv7_hf/native_quant_a8w8.py",
                "rwkv7_hf/train_temp_cuda.py",
            }
            assert entry["adaptation"]
        destinations.add(destination.name)
    assert transfers == {"byte_identical": 86, "adapted_clean_boundary": 16}

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
        "bn_tn_tuning.py",
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
        "official_training_cuda.py",
        "triton_compat.py",
        "SELF_CHUNK_LICENSE",
    }
    assert required_families <= destinations


def test_nvidia_capability_inventory_maps_every_migrated_byte_once():
    manifest = json.loads((NVIDIA / "MIGRATION_MANIFEST.json").read_text())
    inventory = json.loads((NVIDIA / "CAPABILITY_INVENTORY.json").read_text())
    assert inventory["schema"] == "rwkv7-nvidia-capability-inventory-v1"
    assert inventory["kernel_api_version"] == 4
    assert inventory["production_auto"].startswith("disabled")

    migrated = {
        Path(entry["destination"]).relative_to("kernels").as_posix()
        for entry in manifest["files"]
    }
    mapped = [
        member
        for capability in inventory["capabilities"]
        for member in capability["migration_files"]
    ]
    assert len(mapped) == len(set(mapped)) == 102
    assert set(mapped) == migrated
    assert all(
        capability["implementation_status"] == "migrated"
        for capability in inventory["capabilities"]
    )


def test_historical_source_scope_classifies_the_entire_frozen_git_tree():
    manifest = json.loads((NVIDIA / "MIGRATION_MANIFEST.json").read_text())
    scope = json.loads((NVIDIA / "SOURCE_SCOPE.json").read_text())
    assert scope["schema"] == "rwkv7-performance-source-scope-v1"
    assert scope["source_commit"] == "1014acf1a52fa4dee1e4d2b46e6059275c1d3bea"
    assert scope["source_subtree_git_tree"] == historical_tree_oid(scope["entries"])
    assert len(scope["entries"]) == 153
    assert scope["counts"] == {
        "adapted_protocol": 26,
        "byte_migrated_nvidia": 86,
        "canonical_reference": 7,
        "non_kernel_feature_retired": 1,
        "separate_hardware_distribution": 27,
        "tooling_relocated_or_retired": 6,
    }
    migrated_sources = {entry["source"] for entry in manifest["files"]}
    scoped_sources = {
        entry["source"]
        for entry in scope["entries"]
        if "destination" in entry
        and entry["disposition"] in {"byte_migrated_nvidia", "adapted_protocol"}
    }
    assert scoped_sources == migrated_sources
    assert not any(
        entry["disposition"] in {"unknown", "unclassified"}
        for entry in scope["entries"]
    )


def test_recurrent_source_scope_preserves_the_complete_v010_kernel_package():
    scope = json.loads((NVIDIA / "RECURRENT_SOURCE_SCOPE.json").read_text())
    assert scope["schema"] == "rwkv7-recurrent-source-scope-v1"
    assert scope["source_commit"] == "0c5ea30ac6868974ba9836c4a065fa8b2847af68"
    assert scope["source_subtree_git_tree"] == historical_tree_oid(
        scope["entries"],
        scope["source_subtree"],
    )
    assert len(scope["entries"]) == 3
    migrated = [
        entry
        for entry in scope["entries"]
        if entry["disposition"] == "byte_migrated_nvidia"
    ]
    assert len(migrated) == 2
    for entry in migrated:
        destination = ROOT / "kernels" / entry["destination"]
        payload = destination.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["destination_sha256"]
        header = b"blob " + str(len(payload)).encode() + b"\0"
        assert (
            hashlib.sha1(  # noqa: S324 - verifies historical Git identity.
                header + payload,
                usedforsecurity=False,
            ).hexdigest()
            == entry["git_blob"]
        )


def test_nvidia_sources_do_not_reintroduce_model_config_or_cache_ownership():
    forbidden_names = {
        "modeling_rwkv7.py",
        "native_model.py",
        "model_cache.py",
        "model_config.py",
    }
    assert not forbidden_names.intersection(path.name for path in NVIDIA.rglob("*"))
    assert (NVIDIA / "prefill_graph_runtime.py").is_file()
    assert (NVIDIA / "prefill_graph_pool.py").is_file()

    for path in NVIDIA.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not name.name.startswith("rwkv7_hf") for name in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("rwkv7_hf")

    training_runtime = (NVIDIA / "training_runtime.py").read_text()
    train_temp_runtime = (NVIDIA / "official_training_cuda.py").read_text()
    for source in (training_runtime, train_temp_runtime):
        assert ".forward =" not in source
        assert "types.MethodType" not in source
