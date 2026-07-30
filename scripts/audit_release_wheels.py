#!/usr/bin/env python3
"""Audit clean HF ownership and the complete NVIDIA migration inside wheels."""

from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
import configparser
import csv
from email.parser import BytesParser
from email.policy import default as email_policy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
from typing import Any
import zipfile

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


HF_REQUIRED = {
    "rwkv7_hf/__init__.py",
    "rwkv7_hf/cache_rwkv7.py",
    "rwkv7_hf/chat_template.jinja",
    "rwkv7_hf/configuration_rwkv7.py",
    "rwkv7_hf/modeling_rwkv7.py",
    "rwkv7_hf/ops_rwkv7.py",
    "rwkv7_hf/tokenization_rwkv7.py",
}
HF_TOOL_REQUIRED = {
    "rwkv7_hf_tools/__init__.py",
    "rwkv7_hf_tools/cli.py",
    "rwkv7_hf_tools/converter.py",
    "rwkv7_hf_tools/manifest.py",
    "rwkv7_hf_tools/smoke.py",
}
HF_FORBIDDEN = {
    "adapter_manifest.py",
    "cli.py",
    "converter.py",
    "model_cache.py",
    "model_config.py",
    "native_model.py",
    "smoke.py",
}
KERNEL_REQUIRED = {
    "rwkv7_kernels/KERNEL_PLUGIN_API.json",
    "rwkv7_kernels/__init__.py",
    "rwkv7_kernels/_runtime_preflight.py",
    "rwkv7_kernels/backend.py",
    "rwkv7_kernels/dispatcher.py",
    "rwkv7_kernels/linear/__init__.py",
    "rwkv7_kernels/linear/training_flattened.py",
    "rwkv7_kernels/model/dense.py",
    "rwkv7_kernels/model/dense_step.py",
    "rwkv7_kernels/model/packing.py",
    "rwkv7_kernels/model_dispatcher.py",
    "rwkv7_kernels/nvidia/graph_pool.py",
    "rwkv7_kernels/nvidia/prefill_graph_pool.py",
    "rwkv7_kernels/nvidia/prefill_graph_runtime.py",
    "rwkv7_kernels/nvidia/official_training_alignment.py",
    "rwkv7_kernels/nvidia/official_training_checkpoint.py",
    "rwkv7_kernels/nvidia/official_training_cuda.py",
    "rwkv7_kernels/nvidia/training_math.py",
    "rwkv7_kernels/nvidia/training_runtime.py",
    "rwkv7_kernels/nvidia/csrc/training/rwkv7_tmix_mix6_shifted_bf16_v1.cpp",
    "rwkv7_kernels/nvidia/csrc/training/rwkv7_tmix_mix6_shifted_bf16_v1.cu",
    "rwkv7_kernels/protocol.py",
    "rwkv7_kernels/quantization.py",
    "rwkv7_kernels/recurrent/graph.py",
    "rwkv7_kernels/recurrent/training_factorized.py",
    "rwkv7_kernels/recurrent/training_matrix.py",
    "rwkv7_kernels/recurrent/triton.py",
    "rwkv7_kernels/trace.py",
    "rwkv7_kernels/training_dispatcher.py",
    "rwkv7_kernels/time_mix/__init__.py",
    "rwkv7_kernels/time_mix/training_mix6.py",
}
KERNEL_FORBIDDEN = {
    "cache_rwkv7.py",
    "configuration_rwkv7.py",
    "model_cache.py",
    "model_config.py",
    "modeling_rwkv7.py",
    "native_model.py",
    "tokenization_rwkv7.py",
}
KERNEL_INIT = "rwkv7_kernels/__init__.py"
KERNEL_CONTRACT = "rwkv7_kernels/KERNEL_PLUGIN_API.json"
KERNEL_PROTOCOL = "rwkv7_kernels/protocol.py"
KERNEL_BACKEND = "rwkv7_kernels/backend.py"
HF_OPS = "rwkv7_hf/ops_rwkv7.py"
KERNEL_API_VERSION = 4
KERNEL_OPERATIONS = (
    "training_program",
    "model_forward",
    "linear_training",
    "mix6_training",
    "recurrent",
)
KERNEL_ENVELOPE_FIELDS = (
    "api_version",
    "kind",
    "supported",
    "implementation",
    "reason",
    "result",
    "phase",
)
MIGRATION_MANIFEST = "rwkv7_kernels/nvidia/MIGRATION_MANIFEST.json"
CAPABILITY_INVENTORY = "rwkv7_kernels/nvidia/CAPABILITY_INVENTORY.json"
SOURCE_SCOPE = "rwkv7_kernels/nvidia/SOURCE_SCOPE.json"
RECURRENT_SOURCE_SCOPE = "rwkv7_kernels/nvidia/RECURRENT_SOURCE_SCOPE.json"
SOURCE_COMMIT = "1014acf1a52fa4dee1e4d2b46e6059275c1d3bea"
SOURCE_TREE = "1bb1fe1cd64662bbd6d29f72c9002a8513af3691"
RECURRENT_SOURCE_COMMIT = "0c5ea30ac6868974ba9836c4a065fa8b2847af68"
RECURRENT_SOURCE_TREE = "7d2fe3ffff72ec2cd44993e14757ef4443ddfcbb"
REQUIRED_CAPABILITIES = {
    "recurrent",
    "dense_decode",
    "fused_prefill",
    "graph_state_pool",
    "sm70_policy",
    "ada_policy",
    "blackwell_policy",
    "quant_w8",
    "quant_w4",
    "quant_a8w8",
    "quant_bntn",
    "quant_bitsandbytes",
    "quant_marlin",
    "quant_torchao",
    "quant_runtime",
    "training_autograd",
}
ROOT = Path(__file__).resolve().parents[1]
HF_PROJECT_PATH = ROOT / "pyproject.toml"
KERNEL_PROJECT_PATH = ROOT / "kernels" / "pyproject.toml"
KERNEL_LICENSE_PATH = ROOT / "kernels" / "LICENSE"
ALLOWED_PHASES = {"prefill", "decode", "training", "quantize"}
ALLOWED_ACTIVATION = {
    "auto_or_explicit",
    "exact_device_policy",
    "explicit_user_opt_in",
    "diagnostic_until_release_gate",
}
ALLOWED_DISPOSITIONS = {
    "adapted_protocol",
    "byte_migrated_nvidia",
    "canonical_reference",
    "non_kernel_feature_retired",
    "separate_hardware_distribution",
    "tooling_relocated_or_retired",
}
ALLOWED_TRANSFERS = {"byte_identical", "adapted_clean_boundary"}
EXPECTED_MIGRATION_TRANSFERS = {
    "adapted_clean_boundary": 16,
    "byte_identical": 86,
}
EXPECTED_SOURCE_SCOPE_DISPOSITIONS = {
    "adapted_protocol": 26,
    "byte_migrated_nvidia": 86,
    "canonical_reference": 7,
    "non_kernel_feature_retired": 1,
    "separate_hardware_distribution": 27,
    "tooling_relocated_or_retired": 6,
}
EXPECTED_RECURRENT_SCOPE_DISPOSITIONS = {
    "adapted_protocol": 1,
    "byte_migrated_nvidia": 2,
}
ADAPTED_MIGRATION_SOURCES = {
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


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-wheel", type=Path, required=True)
    parser.add_argument("--kernel-wheel", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def safe_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise ValueError(f"unsafe wheel member: {info.filename}")
        if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
            raise ValueError(f"symbolic link is not allowed in wheel: {info.filename}")
        if info.filename in members:
            raise ValueError(f"duplicate wheel member: {info.filename}")
        if not info.is_dir():
            members[info.filename] = info
    return members


def open_wheel(path: Path) -> tuple[zipfile.ZipFile, dict[str, zipfile.ZipInfo]]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink() or path.suffix != ".whl":
        raise ValueError(f"missing or unsafe wheel: {path}")
    archive = zipfile.ZipFile(path)
    try:
        return archive, safe_members(archive)
    except Exception:
        archive.close()
        raise


def _project_contract(
    pyproject_path: Path,
) -> tuple[list[Requirement], list[str], dict[str, str]]:
    """Read the exact dependency, extra, and console-script release contract."""

    document = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = document.get("project")
    if not isinstance(project, dict):
        raise ValueError(f"project metadata is missing: {pyproject_path}")

    requirements: list[Requirement] = []
    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list) or not all(
        isinstance(value, str) for value in dependencies
    ):
        raise ValueError(f"project dependencies are malformed: {pyproject_path}")
    requirements.extend(Requirement(value) for value in dependencies)

    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, dict):
        raise ValueError(
            f"project optional dependencies are malformed: {pyproject_path}"
        )
    extras: list[str] = []
    for extra, values in optional.items():
        if (
            not isinstance(extra, str)
            or not isinstance(values, list)
            or not all(isinstance(value, str) for value in values)
        ):
            raise ValueError(
                f"project optional dependencies are malformed: {pyproject_path}"
            )
        extras.append(extra)
        for value in values:
            requirement = Requirement(value)
            extra_marker = f'extra == "{extra}"'
            if requirement.marker is not None:
                requirement.marker = Marker(
                    f"({requirement.marker}) and {extra_marker}"
                )
            else:
                requirement.marker = Marker(extra_marker)
            requirements.append(requirement)

    raw_scripts = project.get("scripts", {})
    if not isinstance(raw_scripts, dict) or not all(
        isinstance(name, str) and isinstance(value, str)
        for name, value in raw_scripts.items()
    ):
        raise ValueError(f"project console scripts are malformed: {pyproject_path}")
    return requirements, sorted(extras), dict(raw_scripts)


def _requirement_key(requirement: Requirement) -> tuple[Any, ...]:
    """Canonical, hashable identity for one complete Requires-Dist field."""

    return (
        canonicalize_name(requirement.name),
        tuple(sorted(canonicalize_name(extra) for extra in requirement.extras)),
        str(requirement.specifier),
        requirement.url or "",
        str(requirement.marker) if requirement.marker is not None else "",
    )


def _audit_record(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    record_member: str,
) -> dict[str, Any]:
    """Require RECORD to cover and authenticate every non-directory member."""

    try:
        text = archive.read(record_member).decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (KeyError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("wheel RECORD is missing or malformed") from exc

    records: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3 or not row[0]:
            raise ValueError("wheel RECORD contains a malformed row")
        name = row[0]
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"wheel RECORD contains an unsafe path: {name}")
        if name in records:
            raise ValueError(f"wheel RECORD contains a duplicate row: {name}")
        records[name] = (row[1], row[2])

    actual_members = set(members)
    recorded_members = set(records)
    if recorded_members != actual_members:
        missing = sorted(actual_members - recorded_members)
        extra = sorted(recorded_members - actual_members)
        raise ValueError(
            f"wheel RECORD coverage differs: missing={missing}, extra={extra}"
        )

    for name in sorted(actual_members):
        digest, size = records[name]
        if name == record_member:
            if digest or size:
                raise ValueError("wheel RECORD self-row must omit hash and size")
            continue
        payload = archive.read(name)
        expected_digest = "sha256=" + base64.urlsafe_b64encode(
            hashlib.sha256(payload).digest()
        ).rstrip(b"=").decode("ascii")
        if digest != expected_digest:
            raise ValueError(f"wheel RECORD hash differs: {name}")
        if size != str(len(payload)):
            raise ValueError(f"wheel RECORD size differs: {name}")
    return {"status": "passed", "members": len(records)}


def _audit_dist_info(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    *,
    distribution: str,
    version: str,
    pyproject_path: Path,
    top_levels: tuple[str, ...],
) -> tuple[Any, dict[str, Any]]:
    """Bind all install-controlling wheel metadata to the tagged project."""

    expected_root = f"{distribution.replace('-', '_')}-{version}.dist-info"
    dist_info_paths = {
        name
        for name in members
        if any(part.endswith(".dist-info") for part in PurePosixPath(name).parts)
    }
    if not dist_info_paths or any(
        PurePosixPath(name).parts[0] != expected_root for name in dist_info_paths
    ):
        raise ValueError(
            f"wheel must contain exactly the expected dist-info root: {expected_root}"
        )

    metadata_member = f"{expected_root}/METADATA"
    wheel_member = f"{expected_root}/WHEEL"
    top_level_member = f"{expected_root}/top_level.txt"
    entry_points_member = f"{expected_root}/entry_points.txt"
    record_member = f"{expected_root}/RECORD"
    required = {metadata_member, wheel_member, top_level_member, record_member}
    missing = sorted(required - set(members))
    if missing:
        raise ValueError(f"wheel dist-info is missing required files: {missing}")

    metadata = BytesParser(policy=email_policy).parsebytes(
        archive.read(metadata_member)
    )
    if metadata.defects:
        raise ValueError("wheel METADATA is malformed")
    if metadata.get_all("Name", []) != [distribution] or metadata.get_all(
        "Version", []
    ) != [version]:
        raise ValueError("wheel metadata name/version differs from release")

    wheel_metadata = BytesParser(policy=email_policy).parsebytes(
        archive.read(wheel_member)
    )
    if wheel_metadata.defects:
        raise ValueError("wheel WHEEL metadata is malformed")
    if wheel_metadata.get_all("Wheel-Version", []) != ["1.0"]:
        raise ValueError("wheel must declare Wheel-Version: 1.0")
    if wheel_metadata.get_all("Root-Is-Purelib", []) != ["true"]:
        raise ValueError("wheel must declare Root-Is-Purelib: true")
    if wheel_metadata.get_all("Tag", []) != ["py3-none-any"]:
        raise ValueError("wheel tag must be exactly py3-none-any")

    expected_top_level = "".join(f"{name}\n" for name in sorted(top_levels)).encode()
    if archive.read(top_level_member) != expected_top_level:
        raise ValueError("wheel top_level.txt differs from the package contract")

    expected_requirements, expected_extras, expected_scripts = _project_contract(
        pyproject_path
    )
    actual_extras = metadata.get_all("Provides-Extra", [])
    if Counter(actual_extras) != Counter(expected_extras):
        raise ValueError(
            "wheel Provides-Extra contract differs: "
            f"expected={expected_extras} actual={actual_extras}"
        )
    try:
        actual_requirements = [
            Requirement(value) for value in metadata.get_all("Requires-Dist", [])
        ]
    except ValueError as exc:
        raise ValueError("wheel Requires-Dist contains an invalid requirement") from exc
    expected_counter = Counter(map(_requirement_key, expected_requirements))
    actual_counter = Counter(map(_requirement_key, actual_requirements))
    if actual_counter != expected_counter:
        raise ValueError(
            "wheel Requires-Dist contract differs: "
            f"expected={sorted(map(str, expected_requirements))} "
            f"actual={sorted(map(str, actual_requirements))}"
        )

    if expected_scripts:
        if entry_points_member not in members:
            raise ValueError("wheel is missing the declared console entry points")
        parser = configparser.ConfigParser(
            interpolation=None,
            strict=True,
            delimiters=("=",),
        )
        parser.optionxform = str
        try:
            parser.read_string(archive.read(entry_points_member).decode("utf-8"))
        except (UnicodeDecodeError, configparser.Error) as exc:
            raise ValueError("wheel entry_points.txt is malformed") from exc
        if parser.defaults() or parser.sections() != ["console_scripts"]:
            raise ValueError("wheel entry_points.txt contains undeclared groups")
        actual_scripts = dict(parser.items("console_scripts", raw=True))
        if actual_scripts != expected_scripts:
            raise ValueError(
                "wheel console entry points differ from pyproject.toml: "
                f"expected={expected_scripts} actual={actual_scripts}"
            )
    elif entry_points_member in members:
        raise ValueError("wheel contains undeclared entry_points.txt")

    record_report = _audit_record(archive, members, record_member)
    return metadata, {
        "status": "passed",
        "dist_info": expected_root,
        "tag": "py3-none-any",
        "top_levels": sorted(top_levels),
        "requirements": len(actual_requirements),
        "extras": expected_extras,
        "console_scripts": expected_scripts,
        "record": record_report,
    }


def audit_hf_wheel(path: Path) -> dict[str, Any]:
    archive, members = open_wheel(path)
    try:
        names = set(members)
        missing = sorted((HF_REQUIRED | HF_TOOL_REQUIRED) - names)
        if missing:
            raise ValueError(f"HF wheel is missing canonical files: {missing}")
        if any(name.startswith("rwkv7_kernels/") for name in names):
            raise ValueError("HF wheel contains the optional kernel package")
        direct_model_files = {
            PurePosixPath(name).name
            for name in names
            if PurePosixPath(name).parent == PurePosixPath("rwkv7_hf")
        }
        forbidden = sorted(HF_FORBIDDEN & direct_model_files)
        if forbidden:
            raise ValueError(
                f"HF model package contains tooling/compatibility files: {forbidden}"
            )
        metadata, dist_info_report = _audit_dist_info(
            archive,
            members,
            distribution="rwkv7-hf",
            version="1.0.0",
            pyproject_path=HF_PROJECT_PATH,
            top_levels=("rwkv7_hf", "rwkv7_hf_tools"),
        )
        kernel_requirements = [
            Requirement(value)
            for value in metadata.get_all("Requires-Dist", [])
            if Requirement(value).name == "rwkv7-kernels"
        ]
        if len(kernel_requirements) != 1:
            raise ValueError("HF wheel must declare one rwkv7-kernels extra")
        kernel_requirement = kernel_requirements[0]
        if (
            str(kernel_requirement.specifier) != "==1.0.0"
            or kernel_requirement.marker is None
            or 'extra == "kernels"' not in str(kernel_requirement.marker)
        ):
            raise ValueError("HF wheel rwkv7-kernels extra is not pinned to 1.0.0")
        ops_tree = module_tree(archive, HF_OPS)
        expected_kernel_api = literal_assignment(
            ops_tree,
            "_KERNEL_API_VERSION",
            member=HF_OPS,
        )
        if type(expected_kernel_api) is not int or (
            expected_kernel_api != KERNEL_API_VERSION
        ):
            raise ValueError(
                "HF optional boundary kernel API version must be "
                f"{KERNEL_API_VERSION}; got {expected_kernel_api!r}"
            )
        return {
            "status": "passed",
            "canonical_files": len(HF_REQUIRED),
            "tool_files": len(HF_TOOL_REQUIRED),
            "kernel_extra": str(kernel_requirement),
            "expected_kernel_api": expected_kernel_api,
            "dist_info": dist_info_report,
            "members": len(members),
        }
    finally:
        archive.close()


def manifest_member(entry: dict[str, Any]) -> str:
    destination = PurePosixPath(str(entry.get("destination", "")))
    if not destination.parts or destination.parts[0] != "kernels":
        raise ValueError(f"unsafe NVIDIA migration destination: {destination}")
    relative = PurePosixPath(*destination.parts[1:])
    if not str(relative).startswith("rwkv7_kernels/nvidia/"):
        raise ValueError(f"migration destination escaped NVIDIA package: {destination}")
    return str(relative)


def inventory_member(value: Any) -> str:
    member = PurePosixPath(str(value))
    if (
        member.is_absolute()
        or ".." in member.parts
        or not member.parts
        or member.parts[0] != "rwkv7_kernels"
    ):
        raise ValueError(f"unsafe capability inventory member: {member}")
    return str(member)


def kernel_policy_fields(archive: zipfile.ZipFile) -> set[str]:
    member = "rwkv7_kernels/nvidia/kernel_policy.py"
    tree = ast.parse(archive.read(member), filename=member)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "KernelPolicy":
            return {
                child.target.id
                for child in node.body
                if isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
            }
    raise ValueError("kernel wheel has no KernelPolicy declaration")


def module_tree(archive: zipfile.ZipFile, member: str) -> ast.Module:
    """Parse one Python wheel member without importing untrusted wheel code."""

    try:
        source = archive.read(member).decode("utf-8")
        return ast.parse(source, filename=member)
    except (KeyError, UnicodeDecodeError, SyntaxError) as exc:
        raise ValueError(
            f"kernel wheel has an unreadable Python module: {member}"
        ) from exc


def literal_assignment(tree: ast.Module, name: str, *, member: str) -> Any:
    """Return one literal module assignment, rejecting ambiguity or code."""

    values: list[ast.expr] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            values.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            values.append(node.value)
    if len(values) != 1:
        raise ValueError(f"{member} must define exactly one literal {name}")
    try:
        return ast.literal_eval(values[0])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{member} {name} must be a literal") from exc


def audit_kernel_protocol(
    archive: zipfile.ZipFile,
    members: set[str],
) -> dict[str, Any]:
    """Bind the executable public protocol to the advertised API inventory."""

    missing = sorted(
        {KERNEL_INIT, KERNEL_CONTRACT, KERNEL_PROTOCOL, KERNEL_BACKEND} - members
    )
    if missing:
        raise ValueError(f"kernel wheel is missing public protocol files: {missing}")

    protocol_tree = module_tree(archive, KERNEL_PROTOCOL)
    api_version = literal_assignment(
        protocol_tree,
        "RWKV7_KERNEL_API_VERSION",
        member=KERNEL_PROTOCOL,
    )
    if api_version != KERNEL_API_VERSION:
        raise ValueError(
            "kernel protocol API version must be "
            f"{KERNEL_API_VERSION}; got {api_version!r}"
        )
    operations = literal_assignment(
        protocol_tree,
        "RWKV7_OPTIONAL_OPERATIONS",
        member=KERNEL_PROTOCOL,
    )
    if operations != KERNEL_OPERATIONS:
        raise ValueError(
            "kernel protocol operations differ from the frozen API-v4 contract"
        )

    contract = json.loads(archive.read(KERNEL_CONTRACT))
    expected_contract = {
        "schema": "rwkv7-kernel-plugin-api-v1",
        "api_version": KERNEL_API_VERSION,
        "entrypoint": "rwkv7_kernels.execute_optional_v4",
        "operations": list(KERNEL_OPERATIONS),
        "envelope_fields": list(KERNEL_ENVELOPE_FIELDS),
        "unsupported_result": None,
        "public_cache_layout": "B,H,K,V",
        "failure_policy": {
            "auto_negative_probe": "reference",
            "optimized_negative_probe": "error",
            "positive_execution_failure": "fail_closed",
        },
    }
    if contract != expected_contract:
        raise ValueError("kernel plugin contract JSON differs from frozen API v4")

    init_tree = module_tree(archive, KERNEL_INIT)
    public_name = "execute_optional_v4"
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == "backend"
        and any(
            alias.name == public_name and alias.asname in (None, public_name)
            for alias in node.names
        )
        for node in init_tree.body
    )
    exports = literal_assignment(init_tree, "__all__", member=KERNEL_INIT)
    if not isinstance(exports, (list, tuple)) or not all(
        isinstance(value, str) for value in exports
    ):
        raise ValueError(f"{KERNEL_INIT} __all__ must be a literal string sequence")
    expected_exports = {
        "__version__",
        "RWKV7_KERNEL_API_VERSION",
        public_name,
    }
    if set(exports) != expected_exports:
        raise ValueError(
            "kernel root __all__ must expose exactly the API-v4 public surface"
        )
    forbidden_v1_imports = {
        alias.name
        for node in init_tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name.endswith("_v1") or alias.name.startswith("probe_")
    }
    if forbidden_v1_imports:
        raise ValueError("kernel root imports legacy v1 symbols")
    backend_tree = module_tree(archive, KERNEL_BACKEND)
    backend_definitions = {
        node.name
        for node in backend_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if public_name not in backend_definitions:
        raise ValueError(f"kernel backend does not define {public_name}")
    if not imported or public_name not in exports:
        raise ValueError(
            f"kernel wheel does not publicly export {public_name} from backend"
        )
    return {
        "status": "passed",
        "api_version": api_version,
        "contract_schema": contract["schema"],
        "operations": list(operations),
        "optional_backend_entrypoint": public_name,
    }


def historical_tree_oid(
    rows: list[dict[str, Any]],
    source_subtree: str = "rwkv7_hf",
) -> str:
    """Rebuild a frozen Git subtree from mode/blob evidence."""

    root: dict[str, Any] = {}
    prefix = PurePosixPath(source_subtree)
    if prefix.is_absolute() or ".." in prefix.parts or not prefix.parts:
        raise ValueError(f"unsafe historical source subtree: {prefix}")
    for row in rows:
        source = PurePosixPath(str(row.get("source", "")))
        if (
            len(source.parts) <= len(prefix.parts)
            or source.parts[: len(prefix.parts)] != prefix.parts
        ):
            raise ValueError(f"unsafe historical source path: {source}")
        mode = str(row.get("git_mode", ""))
        blob = str(row.get("git_blob", ""))
        if mode not in {"100644", "100755"}:
            raise ValueError(f"historical source has invalid Git mode: {source}")
        try:
            blob_bytes = bytes.fromhex(blob)
        except ValueError as exc:
            raise ValueError(
                f"historical source has invalid Git blob: {source}"
            ) from exc
        if len(blob_bytes) != 20:
            raise ValueError(f"historical source has invalid Git blob: {source}")
        node = root
        for part in source.parts[len(prefix.parts) : -1]:
            current = node.setdefault(part, {})
            if not isinstance(current, dict):
                raise ValueError(f"historical source path collides: {source}")
            node = current
        leaf = source.parts[-1]
        if leaf in node:
            raise ValueError(f"historical source path is duplicated: {source}")
        node[leaf] = (mode, blob)

    def tree_oid(node: dict[str, Any]) -> str:
        payload: list[bytes] = []
        ordered = sorted(
            node.items(),
            key=lambda item: (
                item[0] + "/" if isinstance(item[1], dict) else item[0]
            ).encode(),
        )
        for name, value in ordered:
            if isinstance(value, dict):
                mode, digest = "40000", tree_oid(value)
            else:
                mode, digest = value
            payload.append(
                mode.encode() + b" " + name.encode() + b"\0" + bytes.fromhex(digest)
            )
        body = b"".join(payload)
        header = b"tree " + str(len(body)).encode() + b"\0"
        return hashlib.sha1(  # noqa: S324 - Git object identity is SHA-1 by design.
            header + body,
            usedforsecurity=False,
        ).hexdigest()

    return tree_oid(root)


def git_blob_oid(payload: bytes) -> str:
    header = b"blob " + str(len(payload)).encode() + b"\0"
    return hashlib.sha1(  # noqa: S324 - Git object identity is SHA-1 by design.
        header + payload,
        usedforsecurity=False,
    ).hexdigest()


def audit_capability_inventory(
    archive: zipfile.ZipFile,
    members: set[str],
    migrated: set[str],
) -> dict[str, Any]:
    if CAPABILITY_INVENTORY not in members:
        raise ValueError("kernel wheel is missing NVIDIA capability inventory")
    inventory = json.loads(archive.read(CAPABILITY_INVENTORY))
    if inventory.get("schema") != "rwkv7-nvidia-capability-inventory-v1":
        raise ValueError("unexpected NVIDIA capability inventory schema")
    if inventory.get("kernel_api_version") != KERNEL_API_VERSION:
        raise ValueError(
            f"capability inventory must bind kernel API version {KERNEL_API_VERSION}"
        )
    rows = inventory.get("capabilities")
    if not isinstance(rows, list):
        raise ValueError("capability inventory must contain a capabilities list")
    ids = [str(row.get("id", "")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("capability inventory contains duplicate ids")
    if set(ids) != REQUIRED_CAPABILITIES:
        missing = sorted(REQUIRED_CAPABILITIES - set(ids))
        extra = sorted(set(ids) - REQUIRED_CAPABILITIES)
        raise ValueError(
            f"capability inventory ids differ: missing={missing}, extra={extra}"
        )

    policy_fields = kernel_policy_fields(archive)
    mapped_migration: list[str] = []
    runtime_members: set[str] = set()
    for row in rows:
        capability = str(row["id"])
        phases = row.get("phases")
        if (
            not isinstance(phases, list)
            or not phases
            or not set(phases) <= ALLOWED_PHASES
        ):
            raise ValueError(f"capability {capability} has invalid phases")
        if row.get("implementation_status") != "migrated":
            raise ValueError(f"capability {capability} is not marked migrated")
        if row.get("activation") not in ALLOWED_ACTIVATION:
            raise ValueError(f"capability {capability} has invalid activation")
        for key in ("runtime_files", "migration_files"):
            values = row.get(key)
            if not isinstance(values, list) or not values:
                raise ValueError(f"capability {capability} has no {key}")
            normalized = [inventory_member(value) for value in values]
            absent = sorted(set(normalized) - members)
            if absent:
                raise ValueError(
                    f"capability {capability} references missing {key}: {absent}"
                )
            if key == "runtime_files":
                runtime_members.update(normalized)
            else:
                mapped_migration.extend(normalized)
        flags = row.get("policy_flags")
        if not isinstance(flags, list):
            raise ValueError(f"capability {capability} policy_flags must be a list")
        unknown_flags = sorted(set(map(str, flags)) - policy_fields)
        if unknown_flags:
            raise ValueError(
                f"capability {capability} references unknown policy flags: {unknown_flags}"
            )

    duplicates = sorted(
        member
        for member in set(mapped_migration)
        if mapped_migration.count(member) != 1
    )
    if duplicates:
        raise ValueError(
            f"migrated sources must map to exactly one capability: {duplicates}"
        )
    if set(mapped_migration) != migrated:
        missing = sorted(migrated - set(mapped_migration))
        extra = sorted(set(mapped_migration) - migrated)
        raise ValueError(
            f"capability migration coverage differs: missing={missing}, extra={extra}"
        )
    return {
        "status": "passed",
        "capabilities": len(rows),
        "mapped_migration_files": len(mapped_migration),
        "runtime_files": len(runtime_members),
        "policy_flags": len(
            {str(flag) for row in rows for flag in row.get("policy_flags", [])}
        ),
    }


def audit_source_scope(
    archive: zipfile.ZipFile,
    members: set[str],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if SOURCE_SCOPE not in members:
        raise ValueError("kernel wheel is missing historical source-scope inventory")
    scope = json.loads(archive.read(SOURCE_SCOPE))
    if scope.get("schema") != "rwkv7-performance-source-scope-v1":
        raise ValueError("unexpected historical source-scope schema")
    if (
        scope.get("source_branch") != "perf/native-kernels-v0.8"
        or scope.get("source_commit") != SOURCE_COMMIT
        or scope.get("source_subtree") != "rwkv7_hf"
        or scope.get("source_subtree_git_tree") != SOURCE_TREE
    ):
        raise ValueError("historical source-scope identity differs")
    rows = scope.get("entries")
    if not isinstance(rows, list) or len(rows) != 153:
        raise ValueError("historical source scope must classify all 153 files")
    sources = [str(row.get("source", "")) for row in rows]
    if len(sources) != len(set(sources)) or any(
        not source.startswith("rwkv7_hf/") for source in sources
    ):
        raise ValueError("historical source scope has duplicate or unsafe paths")
    rebuilt_tree = historical_tree_oid(rows)
    if rebuilt_tree != SOURCE_TREE:
        raise ValueError(
            "historical source-scope entries do not reconstruct the frozen Git tree"
        )
    dispositions = [str(row.get("disposition", "")) for row in rows]
    if not set(dispositions) <= ALLOWED_DISPOSITIONS:
        unknown = sorted(set(dispositions) - ALLOWED_DISPOSITIONS)
        raise ValueError(f"historical source scope has unknown dispositions: {unknown}")

    counts = {name: dispositions.count(name) for name in ALLOWED_DISPOSITIONS}
    counts = {name: count for name, count in sorted(counts.items()) if count}
    if scope.get("counts") != counts:
        raise ValueError("historical source-scope counts differ from its entries")
    if counts != EXPECTED_SOURCE_SCOPE_DISPOSITIONS:
        raise ValueError(
            "historical source-scope canonical disposition counts differ: "
            f"expected={EXPECTED_SOURCE_SCOPE_DISPOSITIONS} actual={counts}"
        )

    migration_by_source = {
        str(row["source"]): (
            manifest_member(row),
            str(row["destination_sha256"]),
            str(row["git_blob"]),
            str(row["transfer"]),
            str(row.get("adaptation", "")),
        )
        for row in manifest["files"]
    }
    scoped_migration: dict[str, tuple[str, str, str, str, str]] = {}
    adapted_kernel_files: set[str] = set()
    hardware_families: set[str] = set()
    for row in rows:
        source = str(row["source"])
        disposition = str(row["disposition"])
        if disposition == "byte_migrated_nvidia":
            destination = inventory_member(row.get("destination"))
            digest = str(row.get("destination_sha256", ""))
            blob = str(row.get("git_blob", ""))
            if destination not in members:
                raise ValueError(f"historical source destination is absent: {source}")
            scoped_migration[source] = (
                destination,
                digest,
                blob,
                "byte_identical",
                "",
            )
        elif disposition == "adapted_protocol":
            replacements = row.get("replacements")
            if not isinstance(replacements, list) or not replacements:
                raise ValueError(f"adapted source has no replacements: {source}")
            for replacement in map(str, replacements):
                if replacement.startswith("rwkv7_kernels/"):
                    normalized = inventory_member(replacement)
                    if normalized not in members:
                        raise ValueError(
                            f"adapted source replacement is absent: {source} -> {normalized}"
                        )
                    adapted_kernel_files.add(normalized)
            if "destination" in row:
                destination = inventory_member(row.get("destination"))
                digest = str(row.get("destination_sha256", ""))
                blob = str(row.get("git_blob", ""))
                if destination not in members:
                    raise ValueError(
                        f"adapted historical destination is absent: {source}"
                    )
                scoped_migration[source] = (
                    destination,
                    digest,
                    blob,
                    "adapted_clean_boundary",
                    str(row.get("adaptation", "")),
                )
        elif disposition == "separate_hardware_distribution":
            family = str(row.get("hardware_family", ""))
            if family not in {"ascend", "apple_mlx", "biren", "metax", "musa"}:
                raise ValueError(
                    f"separate hardware source has invalid family: {source}"
                )
            hardware_families.add(family)
        elif disposition == "non_kernel_feature_retired" and not row.get("reason"):
            raise ValueError(f"retired non-kernel source has no reason: {source}")
    if scoped_migration != migration_by_source:
        missing = sorted(set(migration_by_source) - set(scoped_migration))
        extra = sorted(set(scoped_migration) - set(migration_by_source))
        changed = sorted(
            source
            for source in set(scoped_migration) & set(migration_by_source)
            if scoped_migration[source] != migration_by_source[source]
        )
        raise ValueError(
            "historical NVIDIA migration scope differs from manifest: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return {
        "status": "passed",
        "historical_files": len(rows),
        "reconstructed_git_tree": rebuilt_tree,
        "dispositions": counts,
        "adapted_kernel_files": len(adapted_kernel_files),
        "separate_hardware_families": sorted(hardware_families),
    }


def audit_recurrent_source_scope(
    archive: zipfile.ZipFile,
    members: set[str],
) -> dict[str, Any]:
    """Verify the earlier optional recurrent wheel was not lost in v2.

    The complete NVIDIA scope above proves the large v0.8 implementation
    migration.  The later v0.10 line introduced the independently packaged
    Graph and Triton recurrence files, so it has a second frozen subtree and
    provenance record.  Both implementation files must still be byte-for-byte
    identical to their historical Git blobs.
    """

    if RECURRENT_SOURCE_SCOPE not in members:
        raise ValueError("kernel wheel is missing recurrent source-scope inventory")
    scope = json.loads(archive.read(RECURRENT_SOURCE_SCOPE))
    if scope.get("schema") != "rwkv7-recurrent-source-scope-v1":
        raise ValueError("unexpected recurrent source-scope schema")
    subtree = "kernel_wheel/rwkv7_kernels"
    if (
        scope.get("source_branch") != "perf/optional-native-backend-v0.10"
        or scope.get("source_commit") != RECURRENT_SOURCE_COMMIT
        or scope.get("source_subtree") != subtree
        or scope.get("source_subtree_git_tree") != RECURRENT_SOURCE_TREE
    ):
        raise ValueError("recurrent source-scope identity differs")
    rows = scope.get("entries")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("recurrent source scope must classify all 3 package files")
    sources = [str(row.get("source", "")) for row in rows]
    if len(sources) != len(set(sources)) or any(
        not source.startswith(f"{subtree}/") for source in sources
    ):
        raise ValueError("recurrent source scope has duplicate or unsafe paths")
    rebuilt_tree = historical_tree_oid(rows, subtree)
    if rebuilt_tree != RECURRENT_SOURCE_TREE:
        raise ValueError(
            "recurrent source-scope entries do not reconstruct the frozen Git tree"
        )
    dispositions = [str(row.get("disposition", "")) for row in rows]
    if not set(dispositions) <= ALLOWED_DISPOSITIONS:
        unknown = sorted(set(dispositions) - ALLOWED_DISPOSITIONS)
        raise ValueError(f"recurrent source scope has unknown dispositions: {unknown}")
    counts = {name: dispositions.count(name) for name in ALLOWED_DISPOSITIONS}
    counts = {name: count for name, count in sorted(counts.items()) if count}
    if scope.get("counts") != counts:
        raise ValueError("recurrent source-scope counts differ from its entries")
    if counts != EXPECTED_RECURRENT_SCOPE_DISPOSITIONS:
        raise ValueError(
            "recurrent source-scope canonical disposition counts differ: "
            f"expected={EXPECTED_RECURRENT_SCOPE_DISPOSITIONS} actual={counts}"
        )

    byte_migrations = 0
    adapted_files: set[str] = set()
    for row in rows:
        source = str(row["source"])
        disposition = str(row["disposition"])
        if disposition == "byte_migrated_nvidia":
            destination = inventory_member(row.get("destination"))
            if destination not in members:
                raise ValueError(f"recurrent source destination is absent: {source}")
            payload = archive.read(destination)
            if sha256_bytes(payload) != row.get("destination_sha256"):
                raise ValueError(
                    f"recurrent source hash mismatch in wheel: {destination}"
                )
            if git_blob_oid(payload) != row.get("git_blob"):
                raise ValueError(
                    f"recurrent source is not byte-identical to Git blob: {source}"
                )
            byte_migrations += 1
        elif disposition == "adapted_protocol":
            replacements = row.get("replacements")
            if not isinstance(replacements, list) or not replacements:
                raise ValueError(
                    f"adapted recurrent source has no replacements: {source}"
                )
            for replacement in map(inventory_member, replacements):
                if replacement not in members:
                    raise ValueError(
                        "adapted recurrent source replacement is absent: "
                        f"{source} -> {replacement}"
                    )
                adapted_files.add(replacement)
    if byte_migrations != 2:
        raise ValueError(
            "recurrent source scope must preserve both implementation files"
        )
    return {
        "status": "passed",
        "historical_files": len(rows),
        "reconstructed_git_tree": rebuilt_tree,
        "dispositions": counts,
        "byte_identical_implementations": byte_migrations,
        "adapted_protocol_files": len(adapted_files),
    }


def audit_kernel_wheel(path: Path) -> dict[str, Any]:
    archive, members = open_wheel(path)
    try:
        names = set(members)
        if CAPABILITY_INVENTORY not in names:
            raise ValueError("kernel wheel is missing NVIDIA capability inventory")
        if SOURCE_SCOPE not in names:
            raise ValueError(
                "kernel wheel is missing historical source-scope inventory"
            )
        if RECURRENT_SOURCE_SCOPE not in names:
            raise ValueError("kernel wheel is missing recurrent source-scope inventory")
        missing = sorted(KERNEL_REQUIRED - names)
        if missing:
            raise ValueError(f"kernel wheel is missing runtime files: {missing}")
        if any(name.startswith("rwkv7_hf/") for name in names):
            raise ValueError("kernel wheel contains the Hugging Face model package")
        forbidden = sorted(
            name for name in names if PurePosixPath(name).name in KERNEL_FORBIDDEN
        )
        if forbidden:
            raise ValueError(
                f"kernel wheel reintroduces model/config/cache ownership: {forbidden}"
            )
        metadata, dist_info_report = _audit_dist_info(
            archive,
            members,
            distribution="rwkv7-kernels",
            version="1.0.0",
            pyproject_path=KERNEL_PROJECT_PATH,
            top_levels=("rwkv7_kernels",),
        )
        if metadata.get("License-Expression") != "MIT":
            raise ValueError("kernel wheel must declare the MIT License-Expression")
        license_files = metadata.get_all("License-File", [])
        if license_files != ["LICENSE"]:
            raise ValueError(
                "kernel wheel must declare exactly one License-File: LICENSE"
            )
        dist_info_root = PurePosixPath(dist_info_report["dist_info"])
        license_member = str(dist_info_root / "licenses" / "LICENSE")
        if license_member not in names:
            raise ValueError("kernel wheel is missing its declared LICENSE payload")
        if (
            not KERNEL_LICENSE_PATH.is_file()
            or KERNEL_LICENSE_PATH.is_symlink()
            or archive.read(license_member) != KERNEL_LICENSE_PATH.read_bytes()
        ):
            raise ValueError("kernel wheel LICENSE payload differs from checkout")
        dependencies = {
            canonicalize_name(Requirement(value).name)
            for value in metadata.get_all("Requires-Dist", [])
            if Requirement(value).marker is None
        }
        if MIGRATION_MANIFEST not in members:
            raise ValueError("kernel wheel is missing NVIDIA migration manifest")
        manifest = json.loads(archive.read(MIGRATION_MANIFEST))
        if manifest.get("schema") != "rwkv7-nvidia-source-migration-v1":
            raise ValueError("unexpected NVIDIA migration manifest schema")
        if manifest.get("source_branch") != "perf/native-kernels-v0.8":
            raise ValueError("unexpected NVIDIA migration source branch")
        rows = manifest.get("files")
        if not isinstance(rows, list) or len(rows) != 102:
            raise ValueError("NVIDIA migration manifest must contain all 102 files")
        migrated: set[str] = set()
        transfers: dict[str, int] = {name: 0 for name in ALLOWED_TRANSFERS}
        for entry in rows:
            member = manifest_member(entry)
            if member in migrated:
                raise ValueError(f"duplicate NVIDIA migration member: {member}")
            if member not in members:
                raise ValueError(f"kernel wheel omitted migrated source: {member}")
            payload = archive.read(member)
            digest = sha256_bytes(payload)
            if digest != entry.get("destination_sha256"):
                raise ValueError(f"migrated source hash mismatch in wheel: {member}")
            transfer = str(entry.get("transfer", ""))
            if transfer not in ALLOWED_TRANSFERS:
                raise ValueError(f"migrated source has invalid transfer: {member}")
            source = str(entry.get("source", ""))
            if transfer == "byte_identical" and git_blob_oid(payload) != entry.get(
                "git_blob"
            ):
                raise ValueError(
                    f"migrated source Git blob mismatch in wheel: {member}"
                )
            if transfer == "adapted_clean_boundary":
                if source not in ADAPTED_MIGRATION_SOURCES:
                    raise ValueError(f"unexpected clean-boundary adaptation: {source}")
                if not str(entry.get("adaptation", "")).strip():
                    raise ValueError(
                        f"clean-boundary adaptation has no rationale: {source}"
                    )
            transfers[transfer] += 1
            migrated.add(member)
        if sum(transfers.values()) != len(rows):
            raise ValueError(
                "NVIDIA migration transfer classes do not cover the 102-file manifest"
            )
        if transfers != EXPECTED_MIGRATION_TRANSFERS:
            raise ValueError(
                "NVIDIA migration canonical transfer counts differ: "
                f"expected={EXPECTED_MIGRATION_TRANSFERS} actual={transfers}"
            )
        capability_report = audit_capability_inventory(
            archive,
            names,
            migrated,
        )
        protocol_report = audit_kernel_protocol(archive, names)
        source_scope_report = audit_source_scope(archive, names, manifest)
        recurrent_source_scope_report = audit_recurrent_source_scope(
            archive,
            names,
        )
        return {
            "status": "passed",
            "public_protocol": protocol_report,
            "capability_inventory": capability_report,
            "source_scope": source_scope_report,
            "recurrent_source_scope": recurrent_source_scope_report,
            "migrated_files": len(migrated),
            "transfers": transfers,
            "runtime_files": len(KERNEL_REQUIRED),
            "dependencies": sorted(dependencies),
            "dist_info": dist_info_report,
            "license": {
                "expression": "MIT",
                "file": "LICENSE",
                "member": license_member,
            },
            "members": len(members),
        }
    finally:
        archive.close()


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    hf_report = audit_hf_wheel(args.hf_wheel)
    kernel_report = audit_kernel_wheel(args.kernel_wheel)
    kernel_api = kernel_report["public_protocol"]["api_version"]
    if hf_report["expected_kernel_api"] != kernel_api:
        raise ValueError(
            "HF/kernel wheel API mismatch: "
            f"hf={hf_report['expected_kernel_api']!r} kernel={kernel_api!r}"
        )
    report = {
        "status": "passed",
        "hf": hf_report,
        "kernels": kernel_report,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
