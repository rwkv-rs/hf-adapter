#!/usr/bin/env python3
# coding=utf-8
"""Train a small RWKV draft model to align with a larger RWKV target for
HF-compatible speculative decoding.

准则 (guideline): this script ONLY produces a draft checkpoint that loads
through the EXISTING ``--draft-model`` / ``draft_model=`` switch of
``rwkv7_speculative_generate``. It does NOT modify the target, does NOT modify
the verify path, and does NOT touch ``rwkv7_speculative_generate`` itself.
The trained draft is a drop-in replacement for the off-the-shelf small RWKV
draft; removing the ``--draft-model`` argument restores the original behavior
with zero loss. The existing 0.1B -> 0.4B path, its tests, and its
``bench/results.jsonl`` rows stay intact as a permanent safe fallback.

Recipe: LoRA-align the draft's next-token distribution to the target's (KL +
cross-entropy-to-target-argmax over a prompt corpus), then merge LoRA back
into the draft and ``save_pretrained``. LoRA (not full fine-tune) is preferred
so the draft keeps its general generality and only nudges toward the target.
Reference: DeepSeek DeepSpec / SpecForge draft-training recipe
(https://github.com/deepseek-ai/DeepSpec) — the training+eval framework is
reused; the draft stays a small RWKV, because DSpark/DFlash/Eagle3 transformer
drafts do not fit RWKV recurrent state.

Usage (GPU, real models)::

    python scripts/train_spec_draft.py \
        --target /path/to/rwkv7-0.4b-hf \
        --draft  /path/to/rwkv7-0.1b-hf \
        --prompts prompts.txt \
        --output /path/to/rwkv7-0.1b-draft-aligned \
        --device cuda --dtype fp16

After training, measure acceptance with the EXISTING bench (the verify path is
unchanged)::

    python bench/bench_speculative_decode.py \
        --target-model /path/to/rwkv7-0.4b-hf \
        --draft-model /path/to/rwkv7-0.1b-draft-aligned \
        ... --results bench/results.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

# FLA backward trips torch.compile/Triton unless Dynamo is disabled (matches tests/test_peft_lora.py).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import torch.nn.functional as F


DEFAULT_TARGET_MODULES = ["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"]


def _module_has_hf_device_map(model: torch.nn.Module) -> bool:
    """Return True for Accelerate/HF device_map-dispatched modules.

    Calling ``.to(device)`` on such modules can error or invalidate the dispatch
    hooks, so movement must be skipped once ``from_pretrained(device_map=...)``
    has placed the model. PEFT wraps models (``PeftModel -> LoraModel -> base``),
    while RWKV HF wrappers expose ``model``; walk only those common containers.
    """
    stack = [model]
    seen: set[int] = set()
    for _ in range(8):
        if not stack:
            break
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        if getattr(obj, "hf_device_map", None) is not None:
            return True
        for attr in ("base_model", "model", "module"):
            child = getattr(obj, attr, None)
            if child is not None and id(child) not in seen:
                stack.append(child)
    return False


def _move_module_to_device_if_safe(model: torch.nn.Module, device: str) -> bool:
    """Move ``model`` unless it is already managed by HF/Accelerate device_map."""
    if _module_has_hf_device_map(model):
        return False
    model.to(device)
    return True


def _module_input_device(model: torch.nn.Module, fallback: str) -> torch.device:
    """Best-effort input device for normal and device_map-dispatched modules."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def _param_identity_fingerprint(model: torch.nn.Module) -> dict[str, tuple[int, tuple[int, ...], torch.dtype, torch.device]]:
    """Lightweight target guard: metadata only, no full tensor clone."""
    return {
        n: (p.data_ptr(), tuple(p.shape), p.dtype, p.device)
        for n, p in model.named_parameters()
    }


def _forward_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Run a model and return full-sequence logits as float.

    Works for both HF CausalLM models (return_dict -> .logits) and plain
    nn.Module stubs returning a tensor or a namedtuple with ``logits``.
    """
    out = model(input_ids=input_ids, return_dict=True)
    if hasattr(out, "logits"):
        return out.logits.float()
    return out.float()


def distill_loss(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    ce_weight: float = 1.0,
    kl_weight: float = 1.0,
) -> torch.Tensor:
    """Align the draft's next-token distribution to the target's.

    Computed on shifted positions (``[:, :-1]`` predict the next token). The
    draft is pulled toward the target's *full* distribution (KL) and toward the
    target's greedy argmax (CE); both keep the output correct under the
    speculative-decoding greedy verify because matching the target's argmax is
    exactly what raises acceptance.
    """
    d = draft_logits[:, :-1, :]
    # Teacher distribution is fixed: detach so gradient flows to the draft only.
    t = target_logits[:, :-1, :].detach()
    log_d = F.log_softmax(d, dim=-1)
    p_t = F.softmax(t, dim=-1)
    loss = torch.zeros((), device=d.device, dtype=torch.float32)
    if kl_weight:
        loss = loss + kl_weight * (p_t * (torch.log(p_t.clamp_min(1e-12)) - log_d)).sum(-1).mean()
    if ce_weight:
        tgt = p_t.argmax(-1)
        loss = loss + ce_weight * F.nll_loss(
            log_d.reshape(-1, log_d.size(-1)), tgt.reshape(-1)
        )
    return loss


def align_draft(
    target: torch.nn.Module,
    draft: torch.nn.Module,
    prompt_token_ids: list[torch.Tensor],
    *,
    epochs: int = 1,
    lr: float = 1e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules: list[str] | None = None,
    task_type: str | None = None,
    ce_weight: float = 1.0,
    kl_weight: float = 1.0,
    merge: bool = True,
    device: str = "cpu",
    seed: int | None = None,
    max_grad_norm: float = 1.0,
    log_every: int = 0,
):
    """LoRA-align ``draft`` to ``target`` over prompts; return (draft, losses).

    The target is frozen; the draft gets a PEFT LoRA adapter trained by
    ``distill_loss``, then (by default) merged back in-place so the returned
    draft is a plain module ``save_pretrained``-able and loadable via the
    existing speculative-decoding draft switch. The verify path is never
    touched.
    """
    from peft import LoraConfig, get_peft_model

    if seed is not None:
        torch.manual_seed(seed)

    if target is draft:
        raise ValueError("target and draft must be separate model instances")

    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    if target_modules is None:
        target_modules = list(DEFAULT_TARGET_MODULES)
    # task_type=None (plain PeftModel) on purpose: the script only trains,
    # merges, and saves — it never generates. A plain PeftModel injects LoRA
    # into any module (stub or real RWKV) without requiring HF generation API.
    lora_cfg = LoraConfig(
        task_type=task_type,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
    )
    draft = get_peft_model(draft, lora_cfg)
    draft.train()
    _move_module_to_device_if_safe(draft, device)
    _move_module_to_device_if_safe(target, device)

    target_param_ids = {id(p) for p in target.parameters()}
    trainable_params = [p for p in draft.parameters() if p.requires_grad]
    overlap = [n for n, p in draft.named_parameters() if p.requires_grad and id(p) in target_param_ids]
    if overlap:
        raise RuntimeError(
            "draft optimizer would update target parameters: " + ", ".join(overlap[:8])
        )
    if not trainable_params:
        raise RuntimeError("no trainable draft LoRA parameters were created")
    target_param_meta = _param_identity_fingerprint(target)

    opt = torch.optim.AdamW(trainable_params, lr=lr)

    losses: list[float] = []
    for _ep in range(int(epochs)):
        for ids in prompt_token_ids:
            target_ids = ids.to(_module_input_device(target, device)).long()
            draft_ids = ids.to(_module_input_device(draft, device)).long()
            with torch.no_grad():
                t_logits = _forward_logits(target, target_ids).to(draft_ids.device)
            d_logits = _forward_logits(draft, draft_ids)
            loss = distill_loss(d_logits, t_logits, ce_weight=ce_weight, kl_weight=kl_weight)
            if not torch.isfinite(loss):
                # Skip unstable steps instead of poisoning params with NaN grads.
                losses.append(float(loss.detach().cpu()))
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            if log_every and (len(losses) % log_every == 0):
                print("step", len(losses), "loss", round(losses[-1], 5))

    if merge:
        draft = draft.merge_and_unload()
    draft.eval()

    # Guard the 准则 without cloning the full target: it must remain frozen, have
    # no gradients, and keep the same parameter storage metadata. This catches
    # accidental optimizer inclusion / module replacement while avoiding an extra
    # full target copy that can OOM 0.4B+ real runs.
    for n, p in target.named_parameters():
        if p.requires_grad:
            raise RuntimeError(f"target parameter unexpectedly trainable: {n}")
        if p.grad is not None:
            raise RuntimeError(f"target parameter received a gradient: {n}")
        meta = target_param_meta.get(n)
        if meta is None:
            raise RuntimeError(f"target parameter disappeared or was replaced: {n}")
        if meta != (p.data_ptr(), tuple(p.shape), p.dtype, p.device):
            raise RuntimeError(f"target parameter storage metadata changed: {n}")
    return draft, losses


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", required=True, help="Target HF model directory (large RWKV)")
    ap.add_argument("--draft", required=True, help="Draft HF model directory (small RWKV); will be aligned")
    ap.add_argument("--prompts", required=True, help="Text file, one prompt per line")
    ap.add_argument("--output", required=True, help="Output dir for the aligned (merged) draft HF model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--max-len", type=int, default=512, help="Per-prompt token cap")
    ap.add_argument(
        "--gen-tokens",
        type=int,
        default=64,
        help="Regenerate target continuations of this length and train on the full target "
             "trajectory (DeepSpec recipe). Default 64 is REQUIRED for acceptance gains: "
             "training only on prompt prefixes (set to 0) empirically LOWERS acceptance "
             "(0.73->0.59) due to train/generation distribution mismatch.",
    )
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--results", default=None, help="Optional bench/results.jsonl path to append a traceability row")
    ap.add_argument("--log-every", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtypes[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, trust_remote_code=True, torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
    )
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft, trust_remote_code=True, torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
    )
    # Match the attn-mode conventions used by the spec-decode path / peft smoke.
    for m in (target, draft):
        m.config.attn_mode = "fused_recurrent"
        m.config.use_cache = False
        m.config.fuse_cross_entropy = False
        for layer in getattr(m.model, "layers", []):
            attn = getattr(layer, "attn", None)
            if hasattr(attn, "mode"):
                attn.mode = "fused_recurrent"

    with open(args.prompts, encoding="utf-8") as fh:
        prompts = [ln.strip() for ln in fh if ln.strip()]
    if not prompts:
        raise SystemExit(f"no prompts found in {args.prompts}")
    prompt_ids = []
    for p in prompts:
        enc = tok(p, return_tensors="pt")
        ids = enc["input_ids"][:, : args.max_len]  # keep [1, seq]; 1D makes FLA forward silently NaN
        if int(ids.numel()) >= 2:
            prompt_ids.append(ids)

    if args.gen_tokens > 0:
        # Spec-decode acceptance depends on the draft matching the target DURING
        # autoregressive generation. Training only on prompt prefixes overfits the
        # draft to "predict-like-target-given-a-prompt" and empirically LOWERS
        # acceptance. Instead, regenerate each target trajectory and train on the
        # target's own token sequence (DeepSpec recipe). The verify path is still
        # untouched -- this only changes the training data.
        target.eval()
        target.config.use_cache = True
        regen = []
        for p in prompts:
            enc = tok(p, return_tensors="pt")
            base = enc["input_ids"].to(_module_input_device(target, args.device))
            with torch.inference_mode():
                full = target.generate(
                    base,
                    max_new_tokens=args.gen_tokens,
                    do_sample=False,
                    pad_token_id=getattr(tok, "pad_token_id", None) or 0,
                )
            seq = full[0][: args.max_len].cpu().long().unsqueeze(0)
            if int(seq.numel()) >= 2:
                regen.append(seq)
        target.config.use_cache = False
        print(f"regenerated {len(regen)} target trajectories (gen_tokens={args.gen_tokens})")
        prompt_ids = regen

    aligned, losses = align_draft(
        target, draft, prompt_ids,
        epochs=args.epochs, lr=args.lr,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        ce_weight=args.ce_weight, kl_weight=args.kl_weight,
        merge=True, device=args.device, max_grad_norm=args.max_grad_norm, log_every=args.log_every,
    )

    os.makedirs(args.output, exist_ok=True)
    aligned.save_pretrained(args.output)
    tok.save_pretrained(args.output)
    print("saved_aligned_draft", args.output)
    if losses:
        print("final_loss", round(losses[-1], 6), "steps", len(losses))

    if args.results:
        row = {
            "type": "spec_draft_trained",
            "target": os.path.basename(os.path.normpath(args.target)),
            "draft": os.path.basename(os.path.normpath(args.draft)),
            "output": args.output,
            "epochs": args.epochs,
            "lr": args.lr,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "ce_weight": args.ce_weight,
            "kl_weight": args.kl_weight,
            "steps": len(losses),
            "final_loss": (round(losses[-1], 6) if losses else None),
            "device": args.device,
            "dtype": args.dtype,
            "note": "traceability only; acceptance is measured by bench_speculative_decode.py with draft=trained",
        }
        with open(args.results, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("wrote_results_row", args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
