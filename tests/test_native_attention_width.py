from __future__ import annotations

import copy
import inspect
import tempfile

import pytest
import torch

from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM
from rwkv7_hf.native_jit import extract, prefill
from rwkv7_hf.native_quant_mm4 import quantize_model_mm4
from rwkv7_hf.native_quant_mm8 import quantize_model_mm8


def build_width_split_model() -> NativeRWKV7ForCausalLM:
    torch.manual_seed(333)
    config = NativeRWKV7Config(
        vocab_size=31,
        hidden_size=8,
        attention_hidden_size=16,
        num_heads=4,
        head_dim=4,
        num_hidden_layers=2,
        intermediate_size=16,
        decay_low_rank_dim=3,
        a_low_rank_dim=3,
        gate_low_rank_dim=3,
        v_low_rank_dim=3,
        use_cache=True,
    )
    return NativeRWKV7ForCausalLM(config)


def test_legacy_config_defaults_attention_width_to_heads_times_head_dim() -> None:
    config = NativeRWKV7Config(
        vocab_size=31,
        hidden_size=8,
        num_heads=2,
        head_dim=4,
        num_hidden_layers=1,
    )
    assert config.attention_hidden_size == 8
    assert NativeRWKV7Config.from_dict(config.to_dict()).attention_hidden_size == 8


def test_tiny_legacy_config_infers_a_valid_single_head() -> None:
    config = NativeRWKV7Config(hidden_size=8, num_hidden_layers=1, vocab_size=16)
    assert config.attention_hidden_size == 8
    assert config.num_heads == 1
    assert config.head_dim == 8


def test_attention_width_must_match_recurrent_heads() -> None:
    with pytest.raises(ValueError, match=r"num_heads \* head_dim"):
        NativeRWKV7Config(
            hidden_size=8,
            attention_hidden_size=12,
            num_heads=2,
            head_dim=4,
        )


def test_width_split_forward_backward_cache_generate_and_reload() -> None:
    model = build_width_split_model()
    attention = model.model.layers[0].attn
    assert attention.r_proj.weight.shape == (16, 8)
    assert attention.k_proj.weight.shape == (16, 8)
    assert attention.v_proj.weight.shape == (16, 8)
    assert attention.o_proj.weight.shape == (8, 16)
    assert attention.g_norm.weight.shape == (16,)
    assert attention.value_dim == 16
    assert attention.head_v_dim == 4

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    result = model(input_ids=input_ids, labels=input_ids, use_cache=True)
    assert result.logits.shape == (2, 3, 31)
    assert result.loss is not None and torch.isfinite(result.loss)
    result.loss.backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )

    state, xpa, xpf, v_first = result.past_key_values
    assert state[0].shape == (2, 4, 4, 4)
    assert xpa[0].shape == (2, 8)
    assert xpf[0].shape == (2, 8)
    assert v_first.shape == (2, 16)

    model.eval()
    with torch.no_grad():
        full = model(input_ids=input_ids, use_cache=True)
        prefill = model(input_ids=input_ids[:, :2], use_cache=True)
        decode = model(
            input_ids=input_ids[:, 2:],
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
        generated = model.generate(
            input_ids=input_ids[:1, :2],
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=0,
        )
    torch.testing.assert_close(decode.logits[:, -1], full.logits[:, -1])
    assert torch.equal(decode.logits[:, -1].argmax(-1), full.logits[:, -1].argmax(-1))
    assert generated.shape == (1, 4)
    assert model.rwkv7_native_model_last_decode_backend() == "native_jit"

    with tempfile.TemporaryDirectory(prefix="rwkv7_width_split_") as model_dir:
        model.save_pretrained(model_dir)
        reloaded = NativeRWKV7ForCausalLM.from_pretrained(model_dir).eval()
        assert reloaded.config.attention_hidden_size == 16
        with torch.no_grad():
            reloaded_logits = reloaded(input_ids=input_ids, use_cache=False).logits
        torch.testing.assert_close(reloaded_logits, full.logits)


def test_width_split_layerwise_prefill_matches_token_eager() -> None:
    model = build_width_split_model().eval()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    packs, _, _, _ = extract(model)
    with torch.no_grad():
        reference = model(input_ids=input_ids, use_cache=True).logits
        logits, state, xpa, xpf = prefill(
            model,
            input_ids,
            packs,
            logits_to_keep=0,
        )
    torch.testing.assert_close(logits, reference, atol=1e-6, rtol=1e-6)
    assert state[0].shape == (2, 4, 4, 4)
    assert xpa[0].shape == (2, 8)
    assert xpf[0].shape == (2, 8)


@pytest.mark.parametrize("mode", ["mm8", "mm4"])
def test_width_split_native_quantized_projection_shapes(mode: str) -> None:
    model = copy.deepcopy(build_width_split_model()).eval()
    if mode == "mm8":
        replaced = quantize_model_mm8(model, min_params=1, fused=False)
    else:
        replaced = quantize_model_mm4(model, min_params=1, fused=False)
    assert replaced > 0
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3]])).logits
    assert logits.shape == (1, 3, 31)
    assert torch.isfinite(logits).all()


def test_width_split_peft_backward_when_available() -> None:
    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(
        build_width_split_model(),
        peft.LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=4,
            target_modules=["r_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    output = model(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        labels=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        use_cache=False,
    )
    output.loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


def test_width_split_trainer_step_when_available() -> None:
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Trainer"):
        pytest.skip("Transformers Trainer is unavailable")
    model = build_width_split_model()
    model.config.use_cache = False
    rows = [
        {"input_ids": torch.tensor([1, 2, 3, 4]), "labels": torch.tensor([1, 2, 3, 4])},
        {"input_ids": torch.tensor([4, 3, 2, 1]), "labels": torch.tensor([4, 3, 2, 1])},
    ]

    def collate(items):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in items]),
            "labels": torch.stack([item["labels"] for item in items]),
        }

    with tempfile.TemporaryDirectory(prefix="rwkv7_width_split_trainer_") as output_dir:
        kwargs = {
            "output_dir": output_dir,
            "max_steps": 1,
            "per_device_train_batch_size": 2,
            "save_strategy": "no",
            "report_to": [],
            "disable_tqdm": True,
            "dataloader_pin_memory": False,
            "remove_unused_columns": False,
        }
        signature = inspect.signature(transformers.TrainingArguments.__init__).parameters
        kwargs["use_cpu" if "use_cpu" in signature else "no_cuda"] = True
        trainer = transformers.Trainer(
            model=model,
            args=transformers.TrainingArguments(**kwargs),
            train_dataset=rows,
            data_collator=collate,
        )
        result = trainer.train()
    assert trainer.state.global_step == 1
    assert torch.isfinite(torch.tensor(float(result.training_loss)))
