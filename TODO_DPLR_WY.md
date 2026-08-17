# DPLR/WY compiled Prefill TODO

DPLR/WY is an opt-in Native Prefill research route. It must not change the
default HF path until full-model correctness and exact-card end-to-end timing
beat the accepted Native route.

## Current implementation

- `triton_wy`: compiled bridge through the fused recurrent scan.
- `triton_dense3`: explicit summary → prefix → apply/output scaffold.
- `triton_wy_compact`: compact summary/prefix factors with the existing
  apply/output stage.
- Native state layout remains `[B,H,N,N]`; token inputs support FP16/BF16/FP32
  and recurrent state accumulation remains FP32 where required.

The critical synthetic target is `B=1,H=16,N=64,T=512,chunk=64,fp16`.
Correctness requires output cosine `>=0.9999` against the recurrent reference,
plus full-model greedy and cache-handoff parity through repo-code HF loading.

## Next work

1. Fuse compact apply/output so the compact route avoids dense `[N,N]`
   materialization and unnecessary launch boundaries.
2. Re-profile summary, prefix, apply/output, state preparation, and surrounding
   projections on the exact target card.
3. Run full-model 0.4B/B1/P512 correctness and repeated timing against the
   current Native Prefill route.
4. Promote only if the opt-in route is correctness-clean and non-negative end
   to end across the declared shape matrix.
5. Record any promoted result as a new artifact in
   `bench/CURRENT_ARTIFACTS.json`, replacing the previous line rather than
   accumulating exploratory directories.

## Guardrails

- Do not present synthetic kernel throughput as full-model throughput.
- Do not make wrapper/Python micro-optimization the main performance plan.
- Do not default-enable `triton_dense3` or `triton_wy_compact` without the
  full-model gate.
- Do not start native vLLM/SGLang work in this repository.
- Keep FLA/PyTorch only as compatibility/reference paths; Native remains the
  performance target.
