from __future__ import annotations

import base64
import csv
import hashlib
import io
from pathlib import Path
import tarfile
import zipfile

import pytest

from rwkv7_hf.configuration_rwkv7 import RWKV7Config
from scripts.audit_release_wheels import (
    HF_REQUIRED,
    HF_TOOL_REQUIRED,
    _project_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _wheel_metadata(
    *,
    distribution: str,
    project_path: Path,
    license_expression: str | None = None,
    license_file: str | None = None,
) -> bytes:
    requirements, extras, _ = _project_contract(project_path)
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {distribution}",
        "Version: 1.0.0",
    ]
    if license_expression is not None:
        lines.append(f"License-Expression: {license_expression}")
    if license_file is not None:
        lines.append(f"License-File: {license_file}")
    lines.extend(f"Provides-Extra: {extra}" for extra in extras)
    lines.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
    return ("\n".join(lines) + "\n").encode()


def _finalize_wheel_members(
    members: dict[str, bytes],
    *,
    dist_info: str,
    top_levels: tuple[str, ...],
    scripts: dict[str, str],
) -> None:
    record_member = f"{dist_info}/RECORD"
    record_override = members.pop(record_member, None)
    members.setdefault(
        f"{dist_info}/WHEEL",
        (
            "Wheel-Version: 1.0\n"
            "Generator: rwkv7-test-fixture\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ).encode(),
    )
    members.setdefault(
        f"{dist_info}/top_level.txt",
        "".join(f"{name}\n" for name in sorted(top_levels)).encode(),
    )
    if scripts:
        members.setdefault(
            f"{dist_info}/entry_points.txt",
            (
                "[console_scripts]\n"
                + "".join(f"{name} = {value}\n" for name, value in scripts.items())
            ).encode(),
        )

    if record_override is not None:
        members[record_member] = record_override
        return
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for member, payload in sorted(members.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
        writer.writerow(
            (
                member,
                f"sha256={digest.rstrip(b'=').decode('ascii')}",
                str(len(payload)),
            )
        )
    writer.writerow((record_member, "", ""))
    members[record_member] = stream.getvalue().encode()


def _write_wheel(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for member, payload in sorted(members.items()):
            archive.writestr(member, payload)


def write_valid_sdist(
    path: Path,
    *,
    wheel: Path,
    distribution: str,
    version: str = "1.0.0",
    replace: dict[str, bytes] | None = None,
    omit: str | None = None,
) -> None:
    root = f"{distribution.replace('-', '_')}-{version}"
    prefixes = (
        ("rwkv7_hf/", "rwkv7_hf_tools/")
        if distribution == "rwkv7-hf"
        else ("rwkv7_kernels/",)
    )
    source_root = ROOT if distribution == "rwkv7-hf" else ROOT / "kernels"
    with zipfile.ZipFile(wheel) as archive:
        metadata_member = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        files = {
            "PKG-INFO": archive.read(metadata_member),
            "pyproject.toml": (source_root / "pyproject.toml").read_bytes(),
            "README.md": (source_root / "README.md").read_bytes(),
        }
        license_path = source_root / "LICENSE"
        if license_path.is_file():
            files["LICENSE"] = license_path.read_bytes()
        for name in archive.namelist():
            if any(name.startswith(prefix) for prefix in prefixes):
                files[name] = archive.read(name)
    files.update(replace or {})
    if omit is not None:
        files.pop(omit)
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
    additions = extra or {}
    members = {
        member: additions.get(member, (ROOT / member).read_bytes())
        for member in HF_REQUIRED | HF_TOOL_REQUIRED
    }
    dist_info = "rwkv7_hf-1.0.0.dist-info"
    members[f"{dist_info}/METADATA"] = metadata or _wheel_metadata(
        distribution="rwkv7-hf",
        project_path=ROOT / "pyproject.toml",
    )
    members.update(
        {
            member: payload
            for member, payload in additions.items()
            if member not in HF_REQUIRED | HF_TOOL_REQUIRED
        }
    )
    _, _, scripts = _project_contract(ROOT / "pyproject.toml")
    _finalize_wheel_members(
        members,
        dist_info=dist_info,
        top_levels=("rwkv7_hf", "rwkv7_hf_tools"),
        scripts=scripts,
    )
    _write_wheel(path, members)


def write_valid_kernel_wheel(
    path: Path,
    *,
    omit: str | None = None,
    tamper: str | None = None,
    extra: dict[str, bytes] | None = None,
    metadata: bytes | None = None,
) -> None:
    package_root = ROOT / "kernels" / "rwkv7_kernels"
    members: dict[str, bytes] = {}
    for source in sorted(package_root.rglob("*")):
        relative = source.relative_to(package_root)
        if (
            not source.is_file()
            or source.is_symlink()
            or "__pycache__" in relative.parts
            or source.suffix in {".pyc", ".pyo"}
        ):
            continue
        member = f"rwkv7_kernels/{relative.as_posix()}"
        members[member] = source.read_bytes()
    dist_info = "rwkv7_kernels-1.0.0.dist-info"
    members[f"{dist_info}/METADATA"] = metadata or _wheel_metadata(
        distribution="rwkv7-kernels",
        project_path=ROOT / "kernels" / "pyproject.toml",
        license_expression="MIT",
        license_file="LICENSE",
    )
    members[f"{dist_info}/licenses/LICENSE"] = (ROOT / "kernels" / "LICENSE").read_bytes()
    members.update(extra or {})
    if omit is not None:
        members.pop(omit)
    if tamper is not None:
        members[tamper] += b"\ntampered\n"
    _finalize_wheel_members(
        members,
        dist_info=dist_info,
        top_levels=("rwkv7_kernels",),
        scripts={},
    )
    _write_wheel(path, members)


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
