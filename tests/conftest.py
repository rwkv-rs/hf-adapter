from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

from rwkv7_hf.configuration_rwkv7 import RWKV7Config
from scripts.audit_release_wheels import (
    CAPABILITY_INVENTORY,
    HF_REQUIRED,
    HF_TOOL_REQUIRED,
    KERNEL_REQUIRED,
    MIGRATION_MANIFEST,
    RECURRENT_SOURCE_SCOPE,
    SOURCE_SCOPE,
)


ROOT = Path(__file__).resolve().parents[1]


def write_valid_sdist(
    path: Path,
    *,
    wheel: Path,
    distribution: str,
    version: str = "1.0.0",
    replace: dict[str, bytes] | None = None,
) -> None:
    root = f"{distribution.replace('-', '_')}-{version}"
    prefixes = (
        ("rwkv7_hf/", "rwkv7_hf_tools/")
        if distribution == "rwkv7-hf"
        else ("rwkv7_kernels/",)
    )
    files = {
        "PKG-INFO": (
            f"Metadata-Version: 2.4\nName: {distribution}\nVersion: {version}\n"
        ).encode(),
        "pyproject.toml": (
            f'[project]\nname = "{distribution}"\nversion = "{version}"\n'
        ).encode(),
    }
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if any(name.startswith(prefix) for prefix in prefixes):
                files[name] = archive.read(name)
    files.update(replace or {})
    with tarfile.open(path, "w:gz") as archive:
        for relative, payload in sorted(files.items()):
            info = tarfile.TarInfo(f"{root}/{relative}")
            info.mode = 0o644
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def write_valid_hf_wheel(
    path: Path,
    *,
    extra: dict[str, bytes] | None = None,
    metadata: bytes | None = None,
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for member in sorted(HF_REQUIRED | HF_TOOL_REQUIRED):
            source = ROOT / member
            archive.writestr(member, source.read_bytes())
        for member, payload in sorted((extra or {}).items()):
            archive.writestr(member, payload)
        archive.writestr(
            "rwkv7_hf-1.0.0.dist-info/METADATA",
            metadata
            or (
                "Metadata-Version: 2.4\n"
                "Name: rwkv7-hf\n"
                "Version: 1.0.0\n"
                "Provides-Extra: kernels\n"
                'Requires-Dist: rwkv7-kernels==1.0.0; extra == "kernels"\n'
            ).encode(),
        )


def write_valid_kernel_wheel(
    path: Path,
    *,
    omit: str | None = None,
    tamper: str | None = None,
    extra: dict[str, bytes] | None = None,
    metadata: bytes | None = None,
) -> None:
    manifest_path = ROOT / "kernels/rwkv7_kernels/nvidia/MIGRATION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    members: dict[str, bytes] = {MIGRATION_MANIFEST: manifest_path.read_bytes()}
    for member in (CAPABILITY_INVENTORY, RECURRENT_SOURCE_SCOPE, SOURCE_SCOPE):
        members[member] = (ROOT / "kernels" / member).read_bytes()
    for row in manifest["files"]:
        source = ROOT / row["destination"]
        member = str(Path(*Path(row["destination"]).parts[1:])).replace("\\", "/")
        members[member] = source.read_bytes()
    for member in KERNEL_REQUIRED:
        members[member] = (ROOT / "kernels" / member).read_bytes()
    members["rwkv7_kernels-1.0.0.dist-info/METADATA"] = metadata or (
        b"Metadata-Version: 2.4\n"
        b"Name: rwkv7-kernels\n"
        b"Version: 1.0.0\n"
        b"Requires-Dist: torch\n"
        b"Requires-Dist: numpy\n"
        b"Requires-Dist: packaging\n"
    )
    members.update(extra or {})
    if omit is not None:
        members.pop(omit)
    if tamper is not None:
        members[tamper] += b"\ntampered\n"
    with zipfile.ZipFile(path, "w") as archive:
        for member, payload in sorted(members.items()):
            archive.writestr(member, payload)


@pytest.fixture
def tiny_config():
    return RWKV7Config(
        vocab_size=64,
        hidden_size=16,
        attention_hidden_size=16,
        num_hidden_layers=2,
        num_heads=2,
        head_dim=8,
        intermediate_size=32,
        decay_low_rank_dim=4,
        gate_low_rank_dim=4,
        a_low_rank_dim=4,
        v_low_rank_dim=4,
        pad_token_id=0,
        eos_token_id=0,
        bos_token_id=1,
    )
