from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
import types
import zipfile

import torch

from rwkv7_hf import native_quant_marlin as marlin
from rwkv7_hf import native_quant_marlin_sources as bundled
from rwkv7_hf import native_jit
from scripts import build_marlin_source_bundle


def test_marlin_scalar_type_id_matches_u4b8_layout() -> None:
    expected = (4 << 8) + (8 << 17) + (1 << 50)
    assert marlin._MARLIN_U4B8_TYPE_ID == expected


def test_marlin_schedule_normalization_is_fail_closed() -> None:
    assert marlin._normalize_marlin_schedule(None) == (-1, -1, -1, -1, -1)
    assert marlin._normalize_marlin_schedule((64, 256, 256)) == (64, 256, 256, -1, -1)
    assert marlin._normalize_marlin_schedule((64, 256, 256, 128)) == (64, 256, 256, 128, -1)
    assert marlin._normalize_marlin_schedule((-1, -1, -1, 128)) == (-1, -1, -1, 128, -1)
    assert marlin._normalize_marlin_schedule((64, 128, 128, 128, 2)) == (64, 128, 128, 128, 2)
    for bad in ((64, 256), (32, 128, 128), (64, 512, 256), (64, 128, 64)):
        try:
            marlin._normalize_marlin_schedule(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid schedule accepted: {bad}")


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


def test_marlin_relu2_uses_explicit_abi_without_changing_linear_forward() -> None:
    module = marlin.MarlinW4Linear.__new__(marlin.MarlinW4Linear)
    torch.nn.Module.__init__(module)
    module.in_features = 2
    module.out_features = 2
    module.fused_relu2 = True
    module.bias = None
    module.register_buffer("qweight", torch.empty(0, dtype=torch.bfloat16))
    calls: list[bool] = []

    def fake_apply(self, x2, out=None, **kwargs):
        calls.append(bool(kwargs["fuse_relu2"]))
        return x2 + 1

    module._apply_marlin = types.MethodType(fake_apply, module)
    x = torch.zeros(1, 2, dtype=torch.bfloat16)

    assert torch.equal(module(x), torch.ones_like(x))
    assert torch.equal(module.rwkv7_forward_relu2(x), torch.ones_like(x))
    assert calls == [False, True]


def test_native_graph_relu2_dispatch_uses_explicit_fused_method() -> None:
    class FakePacked(torch.nn.Module):
        in_features = 2
        out_features = 2
        fused_relu2 = True

        def forward(self, _x):
            raise AssertionError("native graph must use the explicit fused ABI")

        def rwkv7_forward_relu2(self, x):
            return x + 1

    x = torch.zeros(1, 2)
    assert torch.equal(
        native_jit._native_graph_ffn_up_relu2_dispatch(x, FakePacked()),
        torch.ones_like(x),
    )


def test_remote_code_source_bundle_is_current_and_extracts(monkeypatch, tmp_path) -> None:
    payload, names = build_marlin_source_bundle.build_payload()
    embedded = base64.b85decode(bundled._BUNDLE_B85.encode("ascii"))
    assert set(names) == set(bundled._BUNDLE_FILES)
    assert bundled._BUNDLE_SHA256 == hashlib.sha256(embedded).hexdigest()
    with zipfile.ZipFile(io.BytesIO(payload)) as expected_archive:
        expected = {
            name: expected_archive.read(name) for name in expected_archive.namelist()
        }
    with zipfile.ZipFile(io.BytesIO(embedded)) as embedded_archive:
        actual = {
            name: embedded_archive.read(name) for name in embedded_archive.namelist()
        }
    assert actual == expected

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
