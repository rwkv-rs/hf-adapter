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


def test_backend_auto_is_native_even_when_fla_is_installed() -> None:
    assert select_native_backend("auto")
    assert select_native_backend("native")
    with pytest.raises(ValueError, match="unsupported user-facing backend"):
        select_native_backend("fla")


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
    readme_zh = (root / "README_ZH.md").read_text(encoding="utf-8")
    guide_zh = (root / "docs" / "USER_GUIDE_ZH.md").read_text(encoding="utf-8")
    ai_guide = (root / "docs" / "AI_ASSISTED_SETUP.md").read_text(encoding="utf-8")
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")

    for text in (readme, readme_zh, guide_zh, ai_guide, agents):
        assert "docs/AI_ASSISTED_SETUP.md" in text or "AI_ASSISTED_SETUP.md" in text
    assert "[中文](README_ZH.md)" in readme
    assert "[English](README.md)" in readme_zh
    for command in (
        "python examples/check_environment.py",
        "python examples/generate.py",
        "RESULT: READY",
    ):
        assert command in ai_guide
    assert "不要在提示词里提供密码" in ai_guide
    assert "不要发送密码" in guide_zh


def test_quickstart_relative_links_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    documents = (
        root / "README.md",
        root / "README_ZH.md",
        root / "docs" / "README.md",
        root / "docs" / "USER_GUIDE.md",
        root / "docs" / "USER_GUIDE_ZH.md",
        root / "docs" / "AI_ASSISTED_SETUP.md",
        root / "docs" / "ADVANCED_USAGE.md",
        root / "docs" / "ADVANCED_USAGE_ZH.md",
        root / "docs" / "COMPLETE_ADAPTER_GUIDE.md",
        root / "docs" / "INFERENCE_WORKFLOWS.md",
        root / "docs" / "TRAINING_WORKFLOWS.md",
        root / "docs" / "QUANTIZATION_USAGE.md",
        root / "docs" / "APPLE_USAGE.md",
        root / "docs" / "WINDOWS_CPU.md",
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


def test_browser_screenshots_and_workflow_commands_stay_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    guide = (root / "docs" / "ADVANCED_USAGE.md").read_text(encoding="utf-8")
    guide_zh = (root / "docs" / "ADVANCED_USAGE_ZH.md").read_text(
        encoding="utf-8"
    )
    for name in (
        "11-huggingface-model-download.jpg",
        "12-github-tokenizer-download.jpg",
    ):
        path = root / "docs" / "assets" / "tutorials" / name
        payload = path.read_bytes()
        assert payload.startswith(b"\xff\xd8\xff"), name
        assert len(payload) > 40_000, name
        assert name in (root / "docs" / "USER_GUIDE_ZH.md").read_text(
            encoding="utf-8"
        )

    tutorial_assets = root / "docs" / "assets" / "tutorials"
    assert not tuple(tutorial_assets.glob("*.png"))
    assert not (tutorial_assets / "source.html").exists()

    for command in (
        "tests/test_speculative_decode.py",
        "scripts/train_spec_draft.py",
        "bench/bench_speculative_decode.py",
        "tests/test_peft_lora.py",
        "tests/test_native_trainer_smoke.py",
        "tests/test_device_map_generate.py",
        "scripts/run_zero_training_smoke.sh",
    ):
        assert command in guide
        assert command in guide_zh

def test_complete_adapter_user_index_stays_discoverable() -> None:
    root = Path(__file__).resolve().parents[1]
    index = (root / "docs" / "COMPLETE_ADAPTER_GUIDE.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    readme_zh = (root / "README_ZH.md").read_text(encoding="utf-8")
    docs_readme = (root / "docs" / "README.md").read_text(encoding="utf-8")
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")

    assert "COMPLETE_ADAPTER_GUIDE.md" in readme
    assert "COMPLETE_ADAPTER_GUIDE.md" in readme_zh
    assert "COMPLETE_ADAPTER_GUIDE.md" in docs_readme
    assert "COMPLETE_ADAPTER_GUIDE.md" in agents

    topical_docs = (
        "INFERENCE_WORKFLOWS.md",
        "TRAINING_WORKFLOWS.md",
        "TRAIN_TEMP_CUDA.md",
        "QUANTIZATION_USAGE.md",
        "APPLE_USAGE.md",
        "WINDOWS_CPU.md",
        "ADVANCED_USAGE_ZH.md",
        "AI_ASSISTED_SETUP.md",
    )
    for name in topical_docs:
        assert name in index

    public_guides = (
        index,
        readme_zh,
        *((root / "docs" / name).read_text(encoding="utf-8") for name in topical_docs),
    )
    for internal_phrase in (
        "教学覆盖合同",
        "这些内容不能写成",
        "每个新增教程必须包含六项内容",
        "本页不再维护第二套 AI 指令",
    ):
        for guide in public_guides:
            assert internal_phrase not in guide
    assert "smallest safe example" in agents
    assert "observable PASS gate" in agents

    for removed_duplicate in (
        "COMPLETE_ADAPTER_GUIDE_ZH.md",
        "INFERENCE_WORKFLOWS_ZH.md",
        "TRAINING_WORKFLOWS_ZH.md",
        "QUANTIZATION_USAGE_ZH.md",
        "APPLE_USAGE_ZH.md",
    ):
        assert not (root / "docs" / removed_duplicate).exists()

    commands_by_doc = {
        "INFERENCE_WORKFLOWS.md": (
            "batch_convert_rwkv7_to_hf.py",
            "sync_hf_adapter_code.py",
            "test_hf_api_contract.py",
            "test_batch_cache.py",
            "test_dynamic_batch_cache.py",
            "test_chunked_prefill.py",
        ),
        "TRAINING_WORKFLOWS.md": (
            "test_peft_lora.py",
            "test_native_peft_save_load_merge.py",
            "test_native_trainer_resume_smoke.py",
            "test_hf_rl_training_smoke.py",
        ),
        "QUANTIZATION_USAGE.md": (
            "test_quantized_inference.py",
            "test_native_bnb_quant_smoke.py",
            "test_native_mm8_persist.py",
        ),
        "APPLE_USAGE.md": (
            "convert_hf_to_mlx.py",
            "mlx_session_batch_smoke.py",
            "mlx_dynamic_serving_bench.py",
            "export_rwkv7_coreml.py",
        ),
        "WINDOWS_CPU.md": (
            "examples/cpu_tiny_demo.py",
            "scripts/run_cpu_demo.ps1",
            "examples/generate.py",
        ),
    }
    for document, commands in commands_by_doc.items():
        text = (root / "docs" / document).read_text(encoding="utf-8")
        assert "AI_ASSISTED_SETUP.md" in text
        assert "AI 会返回完整命令、退出码和验收结果" in text
        for command in commands:
            assert command in text


def test_ai_instructions_have_one_canonical_source() -> None:
    root = Path(__file__).resolve().parents[1]
    ai_guide = (root / "docs" / "AI_ASSISTED_SETUP.md").read_text(encoding="utf-8")
    for task_id in (
        "first-run",
        "windows-cpu",
        "inference",
        "cache",
        "speculative",
        "training",
        "train-temp-alignment",
        "trl",
        "multi-gpu-inference",
        "deepspeed",
        "quantization",
        "apple",
    ):
        assert f"`{task_id}`" in ai_guide
    for field in ("TASK_ID:", "MODEL:", "DEVICE:", "DTYPE:", "RESULT_DIR:"):
        assert field in ai_guide

    topical_docs = (
        "INFERENCE_WORKFLOWS.md",
        "TRAINING_WORKFLOWS.md",
        "TRAIN_TEMP_CUDA.md",
        "QUANTIZATION_USAGE.md",
        "APPLE_USAGE.md",
        "ADVANCED_USAGE.md",
        "ADVANCED_USAGE_ZH.md",
        "WINDOWS_CPU.md",
    )
    for document in topical_docs:
        text = (root / "docs" / document).read_text(encoding="utf-8")
        assert "AI_ASSISTED_SETUP.md" in text
        assert "TASK_ID:" not in text


def test_train_temp_tutorial_has_user_acceptance_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs" / "TRAIN_TEMP_CUDA.md").read_text(encoding="utf-8")

    for required in (
        "前置条件和支持环境",
        "最小安全输入",
        "可直接复制的 API",
        "精确通过标准",
        "失败恢复和当前限制",
        "AI_ASSISTED_SETUP.md",
        "train_temp_causal_cross_entropy",
        "compare_convergence_cohort.json",
    ):
        assert required in text
    assert "TASK_ID:" not in text


def test_windows_cpu_tutorial_has_user_acceptance_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs" / "WINDOWS_CPU.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    readme_zh = (root / "README_ZH.md").read_text(encoding="utf-8")

    for required in (
        "前置条件和支持环境",
        "最小安全模型和输入",
        "可直接复制的命令和 API",
        "精确且可观察的通过标准",
        "失败恢复方法和当前限制",
        "AI_ASSISTED_SETUP.md",
        "CPU INFERENCE PASS",
        "CPU TRAINING PASS",
        "CPU SAVE/RELOAD PASS",
        "CPU DEMO PASS",
        "final_loss < initial_loss",
        "run_cpu_demo.ps1",
        "examples/cpu_tiny_demo.py",
    ):
        assert required in text
    assert "TASK_ID:" not in text
    assert "docs/WINDOWS_CPU.md" in readme
    assert "docs/WINDOWS_CPU.md" in readme_zh
    assert (root / "scripts" / "run_cpu_demo.ps1").is_file()
