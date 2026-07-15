# Documentation freshness audit — 2026-07-15

## Scope

The audit scanned every project Markdown file visible in the checkout while
excluding generated cache content: 151 files total (9 root documents, 27 under
`docs/`, and 115 under `bench/`). It also ran the repository local-link checker.

The purpose is not to rewrite old measurements. It is to prevent dated plans,
negative experiments and exact-card snapshots from being mistaken for current
repository status.

## Classification used

| Class | Examples | Treatment |
|---|---|---|
| Canonical current | `HF_STATUS.md`, `HF_TODO.md`, `BENCHMARK.md`, `docs/ACCEPTANCE.md`, `docs/HARDWARE_MATRIX.md` | Resolve contradictions and update accepted status |
| Current engineering reference | `docs/BACKENDS.md`, fused-backend roadmaps, runtime architecture | Keep implementation direction; do not promote unmeasured claims |
| Dated validation/evidence | validation matrices and `bench/<topic>_<date>/` artifacts | Preserve exact rows and scope; add a snapshot boundary where needed |
| Historical plan/investigation | `docs/plans/`, `docs/archive/`, dated live notes and branch TODOs | Preserve rationale but label as historical/superseded |

## Stale or ambiguous items corrected

1. `README.md` no longer says V100 ZeRO-3 resume is an unclosed initial gap;
   resume is recorded through 2.9B on 2×V100, while larger matrices remain
   open.
2. `README.md` no longer describes the whole project as only a wrapper-first
   prototype. It distinguishes the HF compatibility shell, native fused hot
   paths and the still-open upstream-Transformers step.
3. `HF_TODO.md` no longer lists the completed RTX 5070 optimized-Qwen matrix as
   active work. It now lists only broader batches, model pairs and cards.
4. `docs/ACCEPTANCE.md` now includes the V100 1.5B/full-FLA-Qwen3.5-2B B1/B8
   raw and active-parameter work gates.
5. The current V100 result is linked consistently from the README, status,
   benchmark, hardware, performance, acceptance and V100 validation pages.
6. The old V100 forced-Torch Qwen plan, intermediate 5070 core-FLA design,
   completed 5070 full-FLA design, Apple live-gap note, MATH500 parity branch
   note and old 4090 summary now have explicit historical/snapshot banners.
7. A100/A800 validation pages are explicitly exact-card snapshots rather than
   cross-repository current verdicts.
8. The strict 149-item Apple audit is distinguished from bounded M5
   production-close claims; its machine-generated counts were not hand-edited.
9. `docs/README.md` now defines document lifecycle and precedence so the same
   ambiguity is less likely to recur.

## Intentionally unchanged

- Dated benchmark JSONL/log conclusions remain immutable evidence. Later wins
  do not alter what an older run measured.
- Attribution and contribution ledgers were not normalized or rewritten.
- Card-local negative quant rows remain in platform histories even when a newer
  selected speed policy passes different shapes.
- Open H100, AMD/ROCm, Turing, full-memory quant, production PP/TP and broader
  training/quality work remains open; the sweep does not convert missing
  evidence into a pass.

## Ongoing rule

When a new artifact changes status:

1. preserve the dated artifact;
2. update `BENCHMARK.md` first;
3. update `HF_STATUS.md`, `HF_TODO.md` and the relevant canonical topic page;
4. mark superseded plans or live notes as historical rather than deleting them;
5. run `python tests/test_markdown_links.py` and
   `python tests/test_document_freshness.py`.
