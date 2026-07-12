from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_hub_release", ROOT / "evaluation" / "audit_hub_release.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeApi:
    def __init__(self, files, *, sha="abc", tags=None):
        self.files = files
        self.sha = sha
        self.tags = tags or {}

    def model_info(self, repo, revision, files_metadata):
        assert revision == "main"
        assert files_metadata is True
        return SimpleNamespace(sha=self.sha, siblings=self.files)

    def list_repo_refs(self, repo_id, repo_type):
        assert repo_type == "model"
        return SimpleNamespace(
            tags=[
                SimpleNamespace(name=name, target_commit=target)
                for name, target in self.tags.items()
            ]
        )


def fake_files(source_dir: Path):
    names = set(MODULE.REQUIRED_FILES)
    names.add("model.safetensors")
    return [
        SimpleNamespace(
            rfilename=name,
            lfs=(
                SimpleNamespace(size=123, sha256="weight-sha")
                if name.endswith(".safetensors")
                else None
            ),
        )
        for name in sorted(names)
    ]


def test_hub_audit_accepts_identical_code_weights_and_tag(tmp_path):
    source = tmp_path / "source"
    remote = tmp_path / "remote"
    source.mkdir()
    remote.mkdir()
    for name in MODULE.CANONICAL_CODE:
        (source / name).write_text(f"canonical {name}\n")
        (remote / name).write_text(f"canonical {name}\n")
    (remote / "config.json").write_text(
        '{"model_type":"rwkv7","auto_map":'
        '{"AutoConfig":"configuration_rwkv7.RWKV7Config",'
        '"AutoModel":"modeling_rwkv7.RWKV7Model",'
        '"AutoModelForCausalLM":"modeling_rwkv7.RWKV7ForCausalLM"}}'
    )
    api = FakeApi(fake_files(source), tags={"v1.0.0": "abc"})

    row = MODULE.audit_repository(
        api=api,
        downloader=lambda **kwargs: str(remote / kwargs["filename"]),
        repo="owner/model",
        revision="main",
        source_dir=source,
        required_tag="v1.0.0",
        baseline_weights={"model.safetensors": {"size": 123, "sha256": "weight-sha"}},
    )

    assert row.status == "passed"
    assert not row.failures


def test_release_manifest_requires_all_six_repositories(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema":"rwkv7-hub-release-stage-v1","repositories":[]}'
    )
    try:
        MODULE.load_release_manifest(manifest)
    except ValueError as exc:
        assert "six repositories" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("incomplete release manifest was accepted")


def test_hub_audit_rejects_code_drift_forbidden_file_and_weight_change(tmp_path):
    source = tmp_path / "source"
    remote = tmp_path / "remote"
    source.mkdir()
    remote.mkdir()
    for name in MODULE.CANONICAL_CODE:
        (source / name).write_text(f"canonical {name}\n")
        (remote / name).write_text(f"canonical {name}\n")
    (remote / "ops_rwkv7.py").write_text("stale\n")
    (remote / "config.json").write_text('{"model_type":"wrong"}')
    files = fake_files(source)
    files.append(SimpleNamespace(rfilename="kernel_bridge.py", lfs=None))
    api = FakeApi(files)

    row = MODULE.audit_repository(
        api=api,
        downloader=lambda **kwargs: str(remote / kwargs["filename"]),
        repo="owner/model",
        revision="main",
        source_dir=source,
        required_tag="v1.0.0",
        baseline_weights={"model.safetensors": {"size": 999, "sha256": "old"}},
    )

    assert row.status == "failed"
    assert any(
        "canonical source differs: ops_rwkv7.py" in item for item in row.failures
    )
    assert any("forbidden files present" in item for item in row.failures)
    assert any("weight LFS metadata differs" in item for item in row.failures)
    assert any("required tag is missing" in item for item in row.failures)
