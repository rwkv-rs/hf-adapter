#!/usr/bin/env python3
"""Audit clean HF ownership and the complete NVIDIA migration inside wheels."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
from typing import Any
import zipfile


HF_REQUIRED = {
    "rwkv7_hf/__init__.py",
    "rwkv7_hf/cache_rwkv7.py",
    "rwkv7_hf/chat_template.jinja",
    "rwkv7_hf/configuration_rwkv7.py",
    "rwkv7_hf/modeling_rwkv7.py",
    "rwkv7_hf/ops_rwkv7.py",
    "rwkv7_hf/tokenization_rwkv7.py",
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
    "rwkv7_kernels/__init__.py",
    "rwkv7_kernels/dispatcher.py",
    "rwkv7_kernels/model/dense.py",
    "rwkv7_kernels/model/dense_step.py",
    "rwkv7_kernels/model/packing.py",
    "rwkv7_kernels/model_dispatcher.py",
    "rwkv7_kernels/nvidia/CAPABILITY_INVENTORY.json",
    "rwkv7_kernels/nvidia/graph_pool.py",
    "rwkv7_kernels/nvidia/prefill_graph_pool.py",
    "rwkv7_kernels/nvidia/prefill_graph_runtime.py",
    "rwkv7_kernels/nvidia/training_runtime.py",
    "rwkv7_kernels/protocol.py",
    "rwkv7_kernels/quantization.py",
    "rwkv7_kernels/recurrent/graph.py",
    "rwkv7_kernels/recurrent/triton.py",
    "rwkv7_kernels/trace.py",
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
MIGRATION_MANIFEST = "rwkv7_kernels/nvidia/MIGRATION_MANIFEST.json"
CAPABILITY_INVENTORY = "rwkv7_kernels/nvidia/CAPABILITY_INVENTORY.json"
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
ALLOWED_PHASES = {"prefill", "decode", "training", "quantize"}
ALLOWED_ACTIVATION = {
    "auto_or_explicit",
    "exact_device_policy",
    "explicit_user_opt_in",
    "diagnostic_until_release_gate",
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


def audit_hf_wheel(path: Path) -> dict[str, Any]:
    archive, members = open_wheel(path)
    try:
        names = set(members)
        missing = sorted(HF_REQUIRED - names)
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
        return {
            "status": "passed",
            "canonical_files": len(HF_REQUIRED),
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
    if inventory.get("kernel_api_version") != 2:
        raise ValueError("capability inventory must bind kernel API version 2")
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
            {
                str(flag)
                for row in rows
                for flag in row.get("policy_flags", [])
            }
        ),
    }


def audit_kernel_wheel(path: Path) -> dict[str, Any]:
    archive, members = open_wheel(path)
    try:
        names = set(members)
        if CAPABILITY_INVENTORY not in names:
            raise ValueError("kernel wheel is missing NVIDIA capability inventory")
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
        for entry in rows:
            member = manifest_member(entry)
            if member in migrated:
                raise ValueError(f"duplicate NVIDIA migration member: {member}")
            if member not in members:
                raise ValueError(f"kernel wheel omitted migrated source: {member}")
            digest = sha256_bytes(archive.read(member))
            if digest != entry.get("destination_sha256"):
                raise ValueError(f"migrated source hash mismatch in wheel: {member}")
            migrated.add(member)
        capability_report = audit_capability_inventory(
            archive,
            names,
            migrated,
        )
        return {
            "status": "passed",
            "capability_inventory": capability_report,
            "migrated_files": len(migrated),
            "runtime_files": len(KERNEL_REQUIRED),
            "members": len(members),
        }
    finally:
        archive.close()


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    report = {
        "status": "passed",
        "hf": audit_hf_wheel(args.hf_wheel),
        "kernels": audit_kernel_wheel(args.kernel_wheel),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
