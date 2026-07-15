# Native quant FFN deep-fusion implementation plan

1. Add optional residual epilogues to MM8 Triton scalar, batched GEMV, and
   Blackwell dot kernels with safe reference fallbacks.
2. Add `MM8Linear.rwkv7_forward_add()` and an independent, default-off
   `RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN_DOWN_ADD` native-graph dispatch.
3. Add CPU/dispatch tests and a synthetic complete-FFN benchmark.
4. Run the synthetic matrix on RTX 5070 and reject any shape with incorrect or
   negative value from promotion.
5. Sync 1.5B/2.9B/7.2B checkpoints under `D:\models\rwkv7` using resumable
   transfers and validate file sizes.
6. Run local MM4/MM8 1.5B and memory-permitting 2.9B end-to-end matrices.
7. Sync the branch to the V100 host and run complete paired 1.5B/2.9B/7.2B
   evidence. Keep dense, up-only, and deep-fused rows separate because the
   first cross-card microbench showed opposite RTX 5070 and V100 behavior.
   Use `bench/run_native_quant_e2e_matrix.py` for resumable fresh processes,
   shared fp16 baselines, and two-GPU shards.
8. Update benchmark/status documentation, commit with Wang Yue DCO, push, and
   move PR #21 to ready only when evidence and CI are complete.
