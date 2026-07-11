from __future__ import annotations

from argparse import Namespace
import hashlib
import io
import json
from pathlib import Path

from evaluation.audit_pypi_release import run


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_pypi_audit_matches_exact_artifact(monkeypatch, tmp_path: Path):
    artifact = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    artifact.write_bytes(b"wheel")
    payload = {
        "info": {"name": "rwkv7-hf", "version": "1.0.0", "requires_python": ">=3.10"},
        "urls": [
            {
                "filename": artifact.name,
                "packagetype": "bdist_wheel",
                "python_version": "py3",
                "size": artifact.stat().st_size,
                "digests": {"sha256": hashlib.sha256(b"wheel").hexdigest()},
                "upload_time_iso_8601": "2026-08-28T00:00:00Z",
                "yanked": False,
            }
        ],
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: Response(json.dumps(payload).encode()),
    )
    report = run(
        Namespace(
            distribution=["rwkv7-hf=1.0.0"],
            artifact=[f"rwkv7-hf={artifact}"],
            index_url="https://pypi.example/pypi",
            timeout=5.0,
            harness_sha="a" * 40,
        )
    )
    assert report["status"] == "passed"
    assert (
        report["distributions"][0]["expected_artifact"]["sha256"]
        == hashlib.sha256(b"wheel").hexdigest()
    )


def test_pypi_audit_rejects_different_published_bytes(monkeypatch, tmp_path: Path):
    artifact = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    artifact.write_bytes(b"local")
    payload = {
        "info": {"name": "rwkv7-kernels", "version": "1.0.0"},
        "urls": [
            {
                "filename": artifact.name,
                "packagetype": "bdist_wheel",
                "python_version": "py3",
                "size": len(b"remote"),
                "digests": {"sha256": hashlib.sha256(b"remote").hexdigest()},
                "yanked": False,
            }
        ],
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: Response(json.dumps(payload).encode()),
    )
    report = run(
        Namespace(
            distribution=["rwkv7-kernels=1.0.0"],
            artifact=[f"rwkv7-kernels={artifact}"],
            index_url="https://pypi.example/pypi",
            timeout=5.0,
            harness_sha=None,
        )
    )
    assert report["status"] == "failed"
    assert "published artifact bytes differ" in report["distributions"][0]["reasons"][0]
