"""Hardware-scoped DeepSpeed profile selection for RWKV-7 HF training.

The generic ZeRO presets retain their historical cross-card defaults.  The
validated exact-card profile lowers only its FP16 initial loss scale.
Callers may always pass an explicit config path to bypass automatic selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .kernel_policy import GPUProfile, detect_gpu_profile, is_v100_name


def deepspeed_config_basename(stage: int, profile: GPUProfile) -> str:
    """Return the exact-card preset name without broad capability promotion."""

    stage = int(stage)
    if stage not in {2, 3}:
        raise ValueError(f"DeepSpeed ZeRO stage must be 2 or 3, got {stage}")
    exact_v100 = bool(
        profile.family == "volta"
        and profile.capability == (7, 0)
        and is_v100_name(profile.name)
    )
    suffix = "_v100" if exact_v100 else ""
    return f"zero{stage}{suffix}.json"


def select_deepspeed_config(
    config_dir: str | Path,
    stage: int,
    *,
    override: str = "",
    torch_module: Any | None = None,
    profile: GPUProfile | None = None,
) -> Path:
    """Select a validated config for the live card, or honor an override."""

    root = Path(config_dir)
    if override:
        requested = Path(override)
        path = requested if requested.is_absolute() else root / requested
    else:
        selected_profile = profile or detect_gpu_profile(torch_module=torch_module)
        path = root / deepspeed_config_basename(stage, selected_profile)
    if not path.exists():
        raise FileNotFoundError(f"DeepSpeed config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = int((payload.get("zero_optimization") or {}).get("stage", -1))
    if actual != int(stage):
        raise ValueError(f"{path} has ZeRO stage {actual}, expected {stage}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the hardware-scoped RWKV-7 DeepSpeed config path"
    )
    parser.add_argument("--stage", type=int, choices=(2, 3), required=True)
    parser.add_argument("--config-dir", default="configs/deepspeed")
    parser.add_argument("--override", default="")
    args = parser.parse_args()
    print(
        select_deepspeed_config(
            args.config_dir,
            args.stage,
            override=args.override,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
