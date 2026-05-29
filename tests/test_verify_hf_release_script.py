from types import SimpleNamespace

import pytest

from scripts.verify_hf_release import verify_remote_weights


def _file(name: str, size: int, sha256: str | None = None) -> SimpleNamespace:
    lfs = None if sha256 is None else SimpleNamespace(sha256=sha256)
    return SimpleNamespace(rfilename=name, size=size, lfs=lfs)


def test_verify_remote_single_weight() -> None:
    info = SimpleNamespace(siblings=[_file("model.safetensors", 123, "a" * 64)])
    manifest = {
        "weights": {
            "layout": "single",
            "filename": "model.safetensors",
            "size_bytes": 123,
            "sha256": "a" * 64,
        }
    }

    assert verify_remote_weights(info, manifest) == {
        "layout": "single",
        "files": 1,
        "size_bytes": 123,
    }


def test_verify_legacy_single_weight_without_layout() -> None:
    info = SimpleNamespace(siblings=[_file("model.safetensors", 123, "a" * 64)])
    manifest = {
        "weights": {
            "filename": "model.safetensors",
            "size_bytes": 123,
            "sha256": "a" * 64,
        }
    }

    assert verify_remote_weights(info, manifest)["layout"] == "single"


def test_verify_remote_sharded_weights() -> None:
    info = SimpleNamespace(
        siblings=[
            _file("model.safetensors.index.json", 50),
            _file("model-00001-of-00002.safetensors", 100, "a" * 64),
            _file("model-00002-of-00002.safetensors", 80, "b" * 64),
        ]
    )
    manifest = {
        "weights": {
            "layout": "sharded",
            "index_filename": "model.safetensors.index.json",
            "shards": [
                {
                    "filename": "model-00001-of-00002.safetensors",
                    "size_bytes": 100,
                    "sha256": "a" * 64,
                },
                {
                    "filename": "model-00002-of-00002.safetensors",
                    "size_bytes": 80,
                    "sha256": "b" * 64,
                },
            ],
        }
    }

    assert verify_remote_weights(info, manifest) == {
        "layout": "sharded",
        "files": 2,
        "size_bytes": 180,
    }


def test_verify_remote_weight_hash_mismatch_fails() -> None:
    info = SimpleNamespace(siblings=[_file("model.safetensors", 123, "b" * 64)])
    manifest = {
        "weights": {
            "layout": "single",
            "filename": "model.safetensors",
            "size_bytes": 123,
            "sha256": "a" * 64,
        }
    }

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        verify_remote_weights(info, manifest)
