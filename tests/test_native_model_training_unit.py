#!/usr/bin/env python3
# coding=utf-8
"""CPU unit coverage for the experimental native RWKV-7 CausalLM training API.

This intentionally uses a tiny random config so it can run without converted
weights, FLA, CUDA, or model files. The checkpoint-level equivalence tests stay
in ``tests/test_native_model.py``; this file verifies the HF/PEFT-facing API
surface that training stacks expect from a CausalLM fallback.
"""
from __future__ import annotations

import inspect
import tempfile

import torch
import torch.nn.functional as F

from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM

try:
    from transformers import Trainer, TrainingArguments
except Exception:  # pragma: no cover - optional in minimal CPU environments.
    Trainer = None
    TrainingArguments = None

try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover - optional across Transformers versions.
    DynamicCache = None

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except Exception:  # pragma: no cover - optional in minimal CPU environments.
    LoraConfig = None
    PeftModel = None
    get_peft_model = None


def build_tiny_model() -> NativeRWKV7ForCausalLM:
    torch.manual_seed(1234)
    cfg = NativeRWKV7Config(
        vocab_size=23,
        hidden_size=8,
        num_hidden_layers=2,
        head_dim=4,
        intermediate_size=16,
        decay_low_rank_dim=3,
        gate_low_rank_dim=3,
        a_low_rank_dim=3,
        v_low_rank_dim=3,
        use_cache=True,
    )
    return NativeRWKV7ForCausalLM(cfg)


def trainable_snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def collate_rows(rows):
    return {
        "input_ids": torch.stack([row["input_ids"] for row in rows], dim=0),
        "labels": torch.stack([row["labels"] for row in rows], dim=0),
    }


def make_cpu_training_args(output_dir: str):
    kwargs = {
        "output_dir": output_dir,
        "max_steps": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-3,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,
        "disable_tqdm": True,
        "dataloader_pin_memory": False,
        "gradient_checkpointing": True,
        "optim": "adamw_torch",
    }
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "logging_strategy" in params:
        kwargs["logging_strategy"] = "no"
    if "use_cpu" in params:
        kwargs["use_cpu"] = True
    elif "no_cuda" in params:
        kwargs["no_cuda"] = True
    return TrainingArguments(**kwargs)


def main() -> int:
    model = build_tiny_model()
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [6, 5, 4, 3, 2],
        ],
        dtype=torch.long,
    )
    labels = input_ids.clone()
    labels[0, 2] = -100

    assert model.get_input_embeddings() is model.model.embeddings
    assert model.get_output_embeddings() is model.lm_head

    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    assert out.loss is not None
    assert out.past_key_values is None
    assert out.logits.shape == (2, 5, 23)
    cached_loss = model(input_ids=input_ids, labels=labels, use_cache=True)
    assert cached_loss.loss is not None
    assert cached_loss.past_key_values is not None
    assert cached_loss.past_key_values.get_seq_length() == input_ids.shape[1]
    if DynamicCache is not None:
        empty_dynamic_loss = model(
            input_ids=input_ids,
            labels=labels,
            past_key_values=DynamicCache(config=model.config),
            use_cache=True,
        )
        assert empty_dynamic_loss.loss is not None
        assert torch.allclose(empty_dynamic_loss.loss, out.loss)
        assert torch.allclose(empty_dynamic_loss.logits, out.logits)
        assert empty_dynamic_loss.past_key_values.get_seq_length() == input_ids.shape[1]
    cached_loss_legacy = model(input_ids=input_ids, labels=labels, use_cache=True, return_legacy_cache=True)
    assert cached_loss_legacy.loss is not None
    assert isinstance(cached_loss_legacy.past_key_values, tuple)
    assert cached_loss_legacy.past_key_values.get_seq_length() == input_ids.shape[1]
    all_ignored = model(input_ids=input_ids, labels=torch.full_like(labels, -100), use_cache=False)
    assert all_ignored.loss is not None
    assert all_ignored.loss.requires_grad
    assert torch.isfinite(all_ignored.loss)
    assert all_ignored.loss.item() == 0.0
    one_token_labels = model(input_ids=input_ids[:, :1], labels=labels[:, :1], use_cache=False)
    assert one_token_labels.loss is not None
    assert one_token_labels.logits.shape == (2, 1, 23)
    assert one_token_labels.loss.requires_grad
    assert torch.isfinite(one_token_labels.loss)
    assert one_token_labels.loss.item() == 0.0
    one_d = model(input_ids=input_ids[0], labels=labels[0])
    assert one_d.loss is not None
    assert one_d.logits.shape == (1, 5, 23)
    expected = F.cross_entropy(
        out.logits[:, :-1, :].contiguous().view(-1, 23).float(),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    assert torch.allclose(out.loss, expected), (out.loss.item(), expected.item())
    out.loss.backward()
    assert model.get_input_embeddings().weight.grad is not None
    assert model.get_output_embeddings().weight.grad is not None
    assert torch.isfinite(model.get_input_embeddings().weight.grad).all()
    assert torch.isfinite(model.get_output_embeddings().weight.grad).all()

    if Trainer is not None and TrainingArguments is not None:
        trainer_model = build_tiny_model()
        trainer_model.config.use_cache = False
        trainer_rows = [
            {"input_ids": input_ids[0].clone(), "labels": input_ids[0].clone()},
            {"input_ids": input_ids[1].clone(), "labels": input_ids[1].clone()},
        ]
        trainer_before = trainable_snapshot(trainer_model)
        with tempfile.TemporaryDirectory(prefix="native_trainer_unit_") as out_dir:
            trainer = Trainer(
                model=trainer_model,
                args=make_cpu_training_args(out_dir),
                train_dataset=trainer_rows,
                data_collator=collate_rows,
            )
            trainer_result = trainer.train()
            trainer_step = int(trainer.state.global_step)
        trainer_delta = max(
            float((trainer_before[name] - param.detach()).abs().max().item())
            for name, param in trainer_model.named_parameters()
            if name in trainer_before
        )
        assert trainer_step == 1
        assert torch.isfinite(torch.tensor(float(trainer_result.training_loss)))
        assert trainer_delta > 0.0
        assert getattr(trainer_model, "is_gradient_checkpointing", True)

    frozen_model = build_tiny_model()
    for param in frozen_model.parameters():
        param.requires_grad_(False)
    frozen_model.enable_input_require_grads()
    hooked = frozen_model(input_ids=input_ids, labels=labels, use_cache=False)
    assert hooked.loss.requires_grad
    hooked.loss.backward()
    frozen_model.disable_input_require_grads()
    unhooked = frozen_model(input_ids=input_ids, labels=labels, use_cache=False)
    assert not unhooked.loss.requires_grad

    if LoraConfig is not None and PeftModel is not None and get_peft_model is not None:
        peft_model = get_peft_model(
            build_tiny_model(),
            LoraConfig(
                task_type="CAUSAL_LM",
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"],
            ),
        )
        peft_model.train()
        before = trainable_snapshot(peft_model)
        assert before, "expected trainable LoRA parameters"
        peft_out = peft_model(input_ids=input_ids, labels=labels, use_cache=False)
        assert peft_out.loss is not None
        peft_out.loss.backward()
        grad_count = sum(
            1
            for p in peft_model.parameters()
            if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
        )
        assert grad_count > 0
        opt = torch.optim.AdamW((p for p in peft_model.parameters() if p.requires_grad), lr=1e-3)
        opt.step()
        after = {n: p for n, p in peft_model.named_parameters() if p.requires_grad}
        updated = sum(1 for n in before if not torch.equal(before[n], after[n]))
        assert updated > 0
        native_base = peft_model.base_model.model
        assert native_base._native_model_has_adapter_layers()
        assert native_base._native_model_requires_eager_decode()
        peft_model.eval()
        with torch.no_grad():
            peft_full = peft_model(input_ids=input_ids[:, :4], use_cache=True)
            peft_prefill = peft_model(input_ids=input_ids[:, :3], use_cache=True)
            peft_decode = peft_model(
                input_ids=input_ids[:, 3:4],
                past_key_values=peft_prefill.past_key_values,
                use_cache=True,
            )
        assert native_base.rwkv7_native_model_last_decode_backend() == "eager"
        assert torch.allclose(peft_decode.logits, peft_full.logits[:, 3:4], atol=1e-6)
        with tempfile.TemporaryDirectory(prefix="native_peft_unit_") as adapter_dir:
            peft_model.save_pretrained(adapter_dir)
            reloaded = PeftModel.from_pretrained(build_tiny_model(), adapter_dir).eval()
            with torch.no_grad():
                reloaded_logits = reloaded(input_ids=input_ids[:, :4], use_cache=False).logits
                generated = reloaded.generate(
                    input_ids=input_ids[:1, :2],
                    max_new_tokens=1,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=0,
                )
            assert torch.allclose(reloaded_logits, peft_full.logits, atol=1e-6)
            assert generated.shape == (1, 3)
            merged = reloaded.merge_and_unload().eval()
            with torch.no_grad():
                merged_logits = merged(input_ids=input_ids[:, :4], use_cache=False).logits
            assert torch.allclose(merged_logits, peft_full.logits, atol=1e-5)

    tuple_out = model(input_ids=input_ids, labels=labels, return_dict=False)
    assert len(tuple_out) == 3
    assert tuple_out[0].shape == ()
    assert tuple_out[1].shape == (2, 5, 23)
    assert tuple_out[2].get_seq_length() == input_ids.shape[1]
    tuple_legacy = model(input_ids=input_ids, labels=labels, return_dict=False, return_legacy_cache=True)
    assert len(tuple_legacy) == 3
    assert tuple_legacy[0].shape == ()
    assert tuple_legacy[1].shape == (2, 5, 23)
    assert isinstance(tuple_legacy[2], tuple)
    assert tuple_legacy[2].get_seq_length() == input_ids.shape[1]
    tuple_no_cache = model(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)
    assert len(tuple_no_cache) == 2
    assert tuple_no_cache[0].shape == ()
    assert tuple_no_cache[1].shape == (2, 5, 23)

    with torch.no_grad():
        cached = model(input_ids=input_ids[:, :3], use_cache=True)
        # use_cache=True keeps full-sequence logits (HF default behavior);
        # only logits_to_keep truncates. input (2,3) -> (2,3,23).
        assert cached.logits.shape == (2, 3, 23)
        assert cached.past_key_values is not None
        cached_from_labels = model(input_ids=input_ids[:, :3], labels=labels[:, :3], use_cache=True)
        next_token = input_ids[:, 3:4]
        decode_from_forward = model(next_token, past_key_values=cached.past_key_values, use_cache=True)
        decode_from_loss = model(next_token, past_key_values=cached_from_labels.past_key_values, use_cache=True)
        assert torch.allclose(decode_from_loss.logits, decode_from_forward.logits)
        padded_ids = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long)
        padded_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
        padded_labels = padded_ids.clone()
        padded_labels[padded_mask == 0] = -100
        masked = model(input_ids=padded_ids, attention_mask=padded_mask, labels=padded_labels)
        compact = model(input_ids=padded_ids[:1, :2], labels=padded_labels[:1, :2])
        assert masked.loss is not None
        assert torch.isfinite(masked.loss)
        assert torch.allclose(masked.logits[0, -1], compact.logits[0, -1], atol=1e-6)

    try:
        model(input_ids=input_ids, labels=labels[:, :4])
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("mismatched labels should raise ValueError")

    try:
        model(input_ids=input_ids[:, :1], labels=labels[:, :1], past_key_values=cached.past_key_values)
    except ValueError as exc:
        assert "past_key_values" in str(exc)
    else:
        raise AssertionError("labels with past_key_values should raise ValueError")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
