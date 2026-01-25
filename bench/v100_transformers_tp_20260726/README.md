# Two-V100 Transformers tensor-parallel acceptance — 2026-07-26

This artifact validates dense fp16 B1 and B8 `tp_plan="auto"` on two
Tesla V100-PCIE-32GB GPUs with the 24-layer, hidden-1024 converted checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  tests/test_tensor_parallel_generate.py \
  --model /home/wzu/rwkv7-pr84-model \
  --dtype fp16 --batch-size 1 --max-new-tokens 4 \
  --results results.jsonl
```

The retained artifact repeats the command with `--batch-size 8`.

Result: **PASS**.

- all six representative matrix shapes matched the expected half-shards;
- cached greedy output matched the single-GPU reference and both ranks;
- minimum logits cosine was `0.9999982119`; B1/B8 maximum absolute fp16
  reduction-order differences were `0.0625/0.15625`;
- local TP peak was `458.7/598.0 MiB` versus `881.6/977.7 MiB` for the B1/B8
  single-GPU references (`0.52031x/0.611611x`);
- the recurrent state remained explicitly replicated at 16 heads per rank;
- decode used eager module calls, so no packed single-device kernel bypassed
  Transformers TP hooks.

Raw row: [`results.jsonl`](results.jsonl).
