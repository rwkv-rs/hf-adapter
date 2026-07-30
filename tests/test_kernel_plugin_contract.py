from __future__ import annotations

import ast
import json
from pathlib import Path

from rwkv7_hf import ops_rwkv7


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kernels" / "rwkv7_kernels"


def test_frozen_kernel_plugin_contract_matches_both_packages():
    contract = json.loads((KERNELS / "KERNEL_PLUGIN_API.json").read_text())
    assert contract == {
        "schema": "rwkv7-kernel-plugin-api-v1",
        "api_version": 4,
        "entrypoint": "rwkv7_kernels.execute_optional_v4",
        "operations": [
            "training_program",
            "model_forward",
            "linear_training",
            "mix6_training",
            "recurrent",
        ],
        "envelope_fields": [
            "api_version",
            "kind",
            "supported",
            "implementation",
            "reason",
            "result",
            "phase",
        ],
        "unsupported_result": None,
        "public_cache_layout": "B,H,K,V",
        "failure_policy": {
            "auto_negative_probe": "reference",
            "optimized_negative_probe": "error",
            "positive_execution_failure": "fail_closed",
        },
    }
    assert ops_rwkv7._KERNEL_API_VERSION == contract["api_version"]


def test_kernel_top_level_keeps_one_execution_entrypoint():
    source = (KERNELS / "__init__.py").read_text()
    assert (
        '__all__ = [\n    "__version__",\n    "RWKV7_KERNEL_API_VERSION",\n    "execute_optional_v4",\n]'
        in source
    )


def test_hf_core_imports_only_the_plugin_top_level():
    source = (ROOT / "rwkv7_hf" / "ops_rwkv7.py").read_text()
    assert 'importlib.import_module("rwkv7_kernels")' in source
    for path in (ROOT / "rwkv7_hf").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    not item.name.startswith("rwkv7_kernels") for item in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("rwkv7_kernels")
