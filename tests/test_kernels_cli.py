from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from rwkv7_hf import kernels_cli


def test_read_index_accepts_local_protocol_file(tmp_path) -> None:
    path = tmp_path / "index.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "rwkv7-kernel-index-v1",
                "wheels": [],
            }
        ),
        encoding="utf-8",
    )
    assert kernels_cli._read_index(str(path))["wheels"] == []  # noqa: SLF001

    path.write_text(json.dumps({"protocol": "wrong"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsupported kernel index protocol"):
        kernels_cli._read_index(str(path))  # noqa: SLF001


def test_install_dry_run_uses_hash_pinned_direct_url(monkeypatch, capsys) -> None:
    fake_torch = SimpleNamespace()
    wheel = {
        "filename": "rwkv7_kernels.whl",
        "url": "https://example.invalid/rwkv7_kernels.whl",
        "sha256": "a" * 64,
        "manifest": {},
    }
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        kernels_cli,
        "inspect_kernel_package",
        lambda **_: {
            "status": "missing",
            "mode": "auto",
            "manifest": None,
            "recommended_distribution": "rwkv7-kernels",
            "recommended_build": "cu124-torch26-sm89",
            "reasons": [],
        },
    )
    monkeypatch.setattr(kernels_cli, "_read_index", lambda _location: {"wheels": []})
    monkeypatch.setattr(
        kernels_cli,
        "_matching_wheels",
        lambda _index, _torch, _device: [wheel],
    )

    assert kernels_cli.main(["install", "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["wheel"] == wheel
    assert "--no-deps" in payload["command"]
    assert payload["command"][-1] == f"{wheel['url']}#sha256={wheel['sha256']}"
