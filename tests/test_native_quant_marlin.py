from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import torch

from rwkv7_hf import native_quant_marlin as marlin
from rwkv7_hf import native_quant_marlin_sources as bundled
from scripts import build_marlin_source_bundle


def test_marlin_scalar_type_id_matches_u4b8_layout() -> None:
    expected = (4 << 8) + (8 << 17) + (1 << 50)
    assert marlin._MARLIN_U4B8_TYPE_ID == expected


def test_vendored_marlin_sources_are_complete_and_namespaced() -> None:
    sources = [Path(path) for path in marlin._marlin_sources()]
    assert len(sources) == 9
    assert all(path.is_file() for path in sources)
    registration = marlin._marlin_source_root() / "marlin_torch_bf16.cpp"
    text = registration.read_text(encoding="utf-8")
    assert "TORCH_LIBRARY(rwkv7_marlin_bf16, m)" in text
    assert "TORCH_LIBRARY_IMPL(rwkv7_marlin_bf16, CUDA, m)" in text


def test_marlin_group128_scale_permutation_preserves_shape_and_values() -> None:
    scales = torch.arange(2 * 128, dtype=torch.bfloat16).reshape(2, 128)
    permuted = marlin._permute_marlin_scales(scales, out_features=128)
    assert permuted.shape == scales.shape
    assert torch.equal(torch.sort(permuted.float().flatten()).values, torch.sort(scales.float().flatten()).values)


def test_remote_code_source_bundle_is_current_and_extracts(monkeypatch, tmp_path) -> None:
    payload, names = build_marlin_source_bundle.build_payload()
    embedded = base64.b85decode(bundled._BUNDLE_B85.encode("ascii"))
    assert embedded == payload
    assert bundled._BUNDLE_SHA256 == hashlib.sha256(payload).hexdigest()
    assert tuple(names) == bundled._BUNDLE_FILES

    cache = tmp_path / "cache"
    monkeypatch.setenv("RWKV7_MARLIN_SOURCE_CACHE", str(cache))
    local_source = marlin._marlin_source_root()
    extracted = bundled.materialize_marlin_sources()
    for name in bundled._BUNDLE_FILES:
        assert (extracted / name).read_bytes() == (local_source / name).read_bytes()

    remote_module = tmp_path / "transformers_modules" / "native_quant_marlin.py"
    remote_module.parent.mkdir(parents=True)
    remote_module.touch()
    monkeypatch.setattr(marlin, "__file__", str(remote_module))
    assert marlin._marlin_source_root() == extracted
