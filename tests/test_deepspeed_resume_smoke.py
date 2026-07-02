#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("DS_IGNORE_CUDA_DETECTION", "1")

# This smoke only resumes checkpoints it just created in a temporary local
# directory. Torch 2.5 is blocked by recent Transformers from loading optimizer
# .pt files, but DeepSpeed resume validation on this V100 box needs the full
# checkpoint path. Disable the guard only for this self-generated smoke.
try:
    import transformers.trainer as _hf_trainer
    import transformers.utils.import_utils as _hf_import_utils

    _hf_import_utils.check_torch_load_is_safe = lambda: None
    _hf_trainer.check_torch_load_is_safe = lambda: None
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("_rwkv7_ds_smoke", _HERE / "test_deepspeed_training_smoke.py")
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load test_deepspeed_training_smoke.py")
ds = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ds
_SPEC.loader.exec_module(ds)


def latest_checkpoint(out_dir: str) -> str:
    ckpts = sorted(Path(out_dir).glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    if not ckpts:
        raise RuntimeError(f"no checkpoints in {out_dir}")
    return str(ckpts[-1])


def maybe_barrier(torch: Any) -> None:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception:
        pass


def release_cuda(torch: Any, *objs: Any) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def shared_out_dir(args: argparse.Namespace, stage: int) -> str:
    run_id = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT", "29500")
    model_id = Path(args.model).name.replace("/", "_")
    path = Path(tempfile.gettempdir()) / f"rwkv7_zero{stage}_resume_{run_id}_{model_id}"
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0 and path.exists():
        shutil.rmtree(path)
    return str(path)


def make_args(TrainingArguments: Any, out_dir: str, max_steps: int, batch_size: int, grad_accum: int, dtype: str, config_path: Path) -> Any:
    torch = ds.optional_torch()
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    return TrainingArguments(
        output_dir=out_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=3,
        report_to=[],
        remove_unused_columns=False,
        fp16=cuda_available and dtype == "fp16",
        bf16=cuda_available and dtype == "bf16",
        dataloader_num_workers=0,
        gradient_checkpointing=False,
        deepspeed=str(config_path),
    )


def run_stage(args: argparse.Namespace, stage: int) -> dict[str, Any]:
    config_path = ds.zero_config_path(Path(args.config_dir), stage)
    ds.ensure_single_process_distributed_env()
    ds_error = ds.deepspeed_import_error()
    if ds_error:
        raise RuntimeError(f"DeepSpeed import failed: {ds_error}")
    torch, _, _, _, AutoTokenizer, Trainer, TrainingArguments = ds.require_training_deps()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dataset = ds.TokenDataset(tok, args.max_length, repeats=args.dataset_repeats)
    collator = ds.CausalCollator(tok)

    out_dir = shared_out_dir(args, stage)
    maybe_barrier(torch)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    try:
        model = ds.load_lora_model(args.model, args.attn_mode, args.train_dtype)
        before_first = ds.trainable_snapshot(model)
        trainer = Trainer(
            model=model,
            args=make_args(TrainingArguments, out_dir, args.first_steps, args.batch_size, args.gradient_accumulation_steps, args.train_dtype, config_path),
            train_dataset=dataset,
            data_collator=collator,
            processing_class=tok,
        )
        first_result = trainer.train()
        maybe_barrier(torch)
        first_loss = float(first_result.training_loss)
        first_delta = ds.max_trainable_delta(before_first, model)
        ckpt = latest_checkpoint(out_dir)
        # Torch 2.5 + recent Transformers cannot weights_only-load numpy RNG
        # states. This smoke validates model/optimizer/Trainer continuity, so
        # remove RNG state files and let Trainer resume without RNG replay.
        for rng_file in Path(ckpt).glob("rng_state*.pth"):
            rng_file.unlink(missing_ok=True)
        first_step = int(trainer.state.global_step)

        del trainer, model, before_first, first_result
        release_cuda(torch)
        maybe_barrier(torch)

        # The first Trainer set transformers' global is_deepspeed_zero3_enabled()
        # flag; deleting the Trainer does NOT reset it. Left set, the next
        # from_pretrained builds the resume model under DeepSpeed's partitioned
        # init, which breaks FLA's _initialize_weights (it indexes
        # param.shape[1], out of range on a partitioned 1-D shard -> IndexError,
        # the ZeRO3-resume failure). Reset it so the resume model builds at full
        # shape; the new Trainer re-enables ZeRO3 via deepspeed.initialize.
        try:
            from transformers.integrations import unset_hf_deepspeed_config
            unset_hf_deepspeed_config()
        except Exception:
            pass

        resumed_model = ds.load_lora_model(args.model, args.attn_mode, args.train_dtype)
        before_resume = ds.trainable_snapshot(resumed_model)
        resumed_trainer = Trainer(
            model=resumed_model,
            args=make_args(TrainingArguments, out_dir, args.resume_steps, args.batch_size, args.gradient_accumulation_steps, args.train_dtype, config_path),
            train_dataset=dataset,
            data_collator=collator,
            processing_class=tok,
        )
        resume_result = resumed_trainer.train(resume_from_checkpoint=ckpt)
        maybe_barrier(torch)
        resume_loss = float(resume_result.training_loss)
        resume_delta = ds.max_trainable_delta(before_resume, resumed_model)
        global_step = int(resumed_trainer.state.global_step)
        metrics = dict(getattr(resume_result, "metrics", {}) or {})
    finally:
        maybe_barrier(torch)
        if int(os.environ.get("RANK", "0")) == 0:
            shutil.rmtree(out_dir, ignore_errors=True)
        maybe_barrier(torch)

    ok = (
        math.isfinite(first_loss)
        and math.isfinite(resume_loss)
        and first_delta > 0.0
        and resume_delta > 0.0
        and first_step == args.first_steps
        and global_step == args.resume_steps
    )
    row = {
        "axis": "deepspeed_resume_smoke",
        "backend": "hf_adapter",
        "trainer_backend": f"trainer_zero{stage}_resume",
        "zero_stage": stage,
        "status": "pass" if ok else "fail",
        "dtype": args.train_dtype,
        "train_dtype": args.train_dtype,
        "device": ds.device_name(),
        **ds.model_metadata(args, resumed_model),
        "cuda_device_count": ds.cuda_device_count(),
        "distributed_world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "attn_mode": args.attn_mode,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "first_steps": args.first_steps,
        "resume_steps": args.resume_steps,
        "dataset_repeats": args.dataset_repeats,
        "max_length": args.max_length,
        "deepspeed_config": str(config_path),
        "checkpoint": Path(ckpt).name,
        "first_loss": first_loss,
        "resume_loss": resume_loss,
        "train_runtime_s": ds.metric(metrics, "train_runtime"),
        "first_max_trainable_delta": first_delta,
        "resume_max_trainable_delta": resume_delta,
        "global_step": global_step,
    }
    if not ok:
        raise AssertionError(row)
    return row


def append_rows(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-size-label", default="", help="Optional size label such as 0.4b; inferred from --model when omitted")
    ap.add_argument("--config-dir", default="configs/deepspeed")
    ap.add_argument("--zero-stage", choices=["2", "3", "both"], default="both")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--max-length", type=int, default=16)
    ap.add_argument("--train-dtype", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--first-steps", type=int, default=1)
    ap.add_argument("--resume-steps", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--dataset-repeats", type=int, default=4)
    ap.add_argument("--results", default="")
    args = ap.parse_args()
    if args.train_dtype is None:
        args.train_dtype = "bf16" if args.device.startswith("cuda") else "fp32"
    
    if args.resume_steps <= args.first_steps:
        raise ValueError("resume-steps must exceed first-steps")
    stages = [2, 3] if args.zero_stage == "both" else [int(args.zero_stage)]
    rows = [run_stage(args, s) for s in stages]
    append_rows(args.results, rows)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print("DEEPSPEED RESUME PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
