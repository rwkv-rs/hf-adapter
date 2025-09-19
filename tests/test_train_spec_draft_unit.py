#!/usr/bin/env python3
# coding=utf-8
"""CPU unit coverage for the speculative-decoding draft-alignment recipe.

Mirrors the philosophy of ``test_native_model_training_unit.py``: tiny random
models, no FLA, no CUDA, no real weights. It verifies the ``align_draft``
pipeline (LoRA distillation toward a frozen target, then merge) and, above
all, guards the 准则:

  * the target's parameters must not change;
  * the returned draft is a plain module (LoRA merged away);
  * distillation loss is finite and does not diverge.

It does NOT exercise ``rwkv7_speculative_generate`` — by design that verify
path is never touched by draft training (see scripts/train_spec_draft.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from scripts.train_spec_draft import align_draft, distill_loss


class _Out:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class _TinyLM(nn.Module):
    """Minimal CausalLM-ish module with named projections peft can target."""

    def __init__(self, vocab: int = 16, dim: int = 8, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab, dim, _weight=torch.randn(vocab, dim, generator=g))
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids=None, return_dict=True, **_):
        h = self.value(torch.tanh(self.key(self.embed(input_ids))))
        logits = self.head(h)
        return _Out(logits) if return_dict else logits


def main() -> int:
    torch.manual_seed(0)
    target = _TinyLM(vocab=16, dim=8, seed=1)
    draft = _TinyLM(vocab=16, dim=8, seed=2)

    ids = torch.randint(0, 16, (2, 8))

    # 1. distill_loss is finite + differentiable through the draft only.
    t_logits = target(input_ids=ids, return_dict=True).logits
    d_logits = draft(input_ids=ids, return_dict=True).logits
    loss0 = distill_loss(d_logits, t_logits, ce_weight=1.0, kl_weight=1.0)
    assert torch.isfinite(loss0), loss0
    loss0.backward()
    assert draft.head.weight.grad is not None
    assert target.head.weight.grad is None  # not yet frozen at module level here

    # 2. align_draft: LoRA on draft, target frozen, then merged.
    draft2 = _TinyLM(vocab=16, dim=8, seed=2)  # fresh, so LoRA starts from the same init
    target_before = {n: p.detach().clone() for n, p in target.named_parameters()}
    aligned, losses = align_draft(
        target, draft2, [ids],
        epochs=3, lr=1e-2,
        lora_r=4, lora_alpha=8,
        target_modules=["key", "value"],
        ce_weight=1.0, kl_weight=1.0,
        merge=True, device="cpu", seed=0,
    )

    # 准则 guard: target untouched.
    for n, p in target.named_parameters():
        assert torch.equal(p.detach(), target_before[n]), f"target param {n} changed"

    # Merged away: no LoRA wrappers left, plain module.
    assert not aligned.__class__.__name__.startswith("Peft"), aligned.__class__.__name__

    # Loss finite, recorded one row per step, and did not diverge.
    assert len(losses) == 3, losses
    assert all(torch.isfinite(torch.tensor(x)) for x in losses), losses
    assert losses[-1] <= losses[0] + 1e-3, losses

    # 3. The aligned draft still produces a finite logits tensor (save/load contract).
    with torch.no_grad():
        out = aligned(input_ids=ids, return_dict=True).logits
    assert out.shape == (2, 8, 16) and torch.isfinite(out).all(), out.shape

    # 4. Validation: alignment must RAISE draft->target token agreement.
    #    argmax agreement is the proxy for greedy-verify acceptance; the verify
    #    path (rwkv7_speculative_generate) is never touched, per the 准则.
    def _agreement(t: _TinyLM, d: torch.nn.Module, seq: torch.Tensor) -> float:
        with torch.no_grad():
            tl = t(input_ids=seq, return_dict=True).logits[:, :-1]
            dl = d(input_ids=seq, return_dict=True).logits[:, :-1]
        return float((tl.argmax(-1) == dl.argmax(-1)).float().mean().item())

    target_v = _TinyLM(vocab=16, dim=8, seed=1)
    draft_off = _TinyLM(vocab=16, dim=8, seed=2)  # different init -> low agreement
    held_out = torch.randint(0, 16, (1, 24))
    train_seqs = [torch.randint(0, 16, (2, 12)) for _ in range(12)]
    before = _agreement(target_v, draft_off, held_out)
    aligned2, _ = align_draft(
        target_v, draft_off, train_seqs,
        epochs=3, lr=1e-2, lora_r=8, lora_alpha=16,
        target_modules=["key", "value", "head"],
        ce_weight=1.0, kl_weight=1.0,
        merge=True, device="cpu", seed=0,
    )
    after = _agreement(target_v, aligned2, held_out)
    print("agreement before/after", round(before, 3), round(after, 3))
    assert after > before, (before, after)

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
