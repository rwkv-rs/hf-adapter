from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "gradio" / "native_hf_v3a_compat.py"
PATCH = ROOT / "examples" / "gradio" / "rwkv-gradio-3-native-hf.patch"
GUIDE = ROOT / "docs" / "GRADIO_NATIVE_HF.md"


def _load_example():
    spec = importlib.util.spec_from_file_location("native_hf_v3a_compat_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_native_hf_gradio_state_expands_prompt_cache() -> None:
    module = _load_example()

    class FakeCache:
        def __init__(self) -> None:
            self.repeat = None

        def clone(self):
            return FakeCache()

        def batch_repeat_interleave(self, count: int) -> None:
            self.repeat = count

    source = module.NativeHFState(batch_size=1, cache=FakeCache())
    destination = module.NativeHFState(batch_size=8)
    module.copy_state_to_batch(destination, source)
    assert destination.cache is not source.cache
    assert destination.cache.repeat == 8


def test_native_hf_gradio_decode_uses_fast_token_api() -> None:
    module = _load_example()

    class FakeModel:
        def __init__(self) -> None:
            self.fast_calls = 0
            self.forward_calls = 0

        def rwkv7_forward_token(
            self,
            token_ids,
            *,
            past_key_values,
            return_dict,
            copy_logits,
        ):
            self.fast_calls += 1
            assert token_ids.shape == (2, 1)
            assert past_key_values == "prompt-cache"
            assert return_dict is False
            assert copy_logits is False
            return torch.arange(10, dtype=torch.float32).view(2, 1, 5), "decode-cache"

        def __call__(self, **_kwargs):
            self.forward_calls += 1
            raise AssertionError("single-token cached decode must not use model.forward")

    bridge = object.__new__(module.RWKV7)
    bridge.model = FakeModel()
    state = module.NativeHFState(batch_size=2, cache="prompt-cache")
    logits = bridge._forward(
        state,
        input_ids=torch.tensor([[3], [4]], dtype=torch.long),
    )

    assert logits.shape == (2, 5)
    assert bridge.model.fast_calls == 1
    assert bridge.model.forward_calls == 0
    assert state.cache == "decode-cache"


def test_space_patch_routes_token_decode_to_native_hf() -> None:
    text = PATCH.read_text(encoding="utf-8")
    assert 'APP3_BACKEND = os.environ.get("APP3_BACKEND", "v3a")' in text
    assert 'model_path = os.environ.get("APP3_HF_MODEL_PATH", "").strip()' in text
    assert "DECODE_USES_TOKEN_IDS" in text
    assert "copy_state_to_batch" in text
    assert 'os.environ.get("APP3_SHARE", "0")' in text
    assert 'os.environ.get("APP3_SERVER_PORT", "7860")' in text
    assert "+accelerate" in text


def test_gradio_guide_has_complete_user_contract_and_single_ai_entry() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    for heading in (
        "前置条件和支持环境",
        "最小安全模型和输入",
        "可直接复制的安装和启动命令",
        "精确且可观察的通过标准",
        "失败恢复和当前限制",
        "让 AI 执行",
    ):
        assert heading in text
    assert "AI_ASSISTED_SETUP.md" in text
    assert "TASK_ID=gradio-native-hf" in text
