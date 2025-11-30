from __future__ import annotations

from pathlib import Path
import re

import torch
import pytest

from examples.check_environment import (
    build_parser as build_doctor_parser,
    inspect_model_directory,
)
from examples.generate import (
    build_parser,
    resolve_device,
    resolve_dtype,
    select_native_backend,
)


def test_generate_example_defaults_are_beginner_safe() -> None:
    args = build_parser().parse_args(["--model", "model", "--prompt", "hello"])
    assert args.device == "auto"
    assert args.dtype == "auto"
    assert args.backend == "auto"
    assert args.temperature == 0.0
    assert args.max_new_tokens == 64


def test_generate_example_cpu_defaults_to_fp32() -> None:
    device = resolve_device("cpu")
    assert device.type == "cpu"
    assert resolve_dtype("auto", device) == torch.float32
    assert resolve_dtype("bf16", device) == torch.bfloat16


def test_backend_auto_falls_back_without_fla() -> None:
    assert select_native_backend(
        "auto", device_type="cpu", fla_available=False, native_env_enabled=False
    )
    assert select_native_backend(
        "auto", device_type="cuda", fla_available=False, native_env_enabled=False
    )
    assert not select_native_backend(
        "auto", device_type="cuda", fla_available=True, native_env_enabled=False
    )
    assert select_native_backend(
        "auto", device_type="cuda", fla_available=True, native_env_enabled=True
    )
    assert select_native_backend(
        "native", device_type="cuda", fla_available=True, native_env_enabled=False
    )
    assert not select_native_backend(
        "fla", device_type="cuda", fla_available=True, native_env_enabled=True
    )
    with pytest.raises(RuntimeError, match="requires flash-linear-attention"):
        select_native_backend(
            "fla", device_type="cuda", fla_available=False, native_env_enabled=False
        )


def test_environment_doctor_model_directory_contract(tmp_path: Path) -> None:
    assert build_doctor_parser().parse_args([]).model is None
    assert "missing config.json" in inspect_model_directory(tmp_path)
    assert "missing model weights (*.safetensors or *.bin)" in inspect_model_directory(
        tmp_path
    )

    for name in (
        "config.json",
        "tokenizer_config.json",
        "rwkv_vocab_v20230424.txt",
        "model.safetensors",
    ):
        (tmp_path / name).write_bytes(b"test")
    assert inspect_model_directory(tmp_path) == []


def test_human_and_ai_quickstart_entries_stay_discoverable() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    guide_zh = (root / "docs" / "USER_GUIDE_ZH.md").read_text(encoding="utf-8")
    ai_guide = (root / "docs" / "AI_ASSISTED_SETUP.md").read_text(encoding="utf-8")
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")

    for text in (readme, guide_zh, ai_guide, agents):
        assert "docs/AI_ASSISTED_SETUP.md" in text or "AI_ASSISTED_SETUP.md" in text
    for command in (
        "python examples/check_environment.py",
        "python examples/generate.py",
        "RESULT: READY",
    ):
        assert command in ai_guide
    assert "Do not paste passwords" in ai_guide
    assert "不要发送密码" in guide_zh


def test_quickstart_relative_links_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    documents = (
        root / "README.md",
        root / "docs" / "README.md",
        root / "docs" / "USER_GUIDE.md",
        root / "docs" / "USER_GUIDE_ZH.md",
        root / "docs" / "AI_ASSISTED_SETUP.md",
    )
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            relative = target.split("#", 1)[0]
            assert (document.parent / relative).exists(), (
                f"broken link in {document.relative_to(root)}: {target}"
            )
