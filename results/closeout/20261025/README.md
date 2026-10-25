# Acceptance resumption closeout: 2026-10-25

**Status record only. `release_ready` remains false; no new validation results are
claimed by this document.** The repair run described here was still executing on
the RTX 4080 when this record was written; per-unit outcomes must be taken from
the manifest and logs under the evidence root, not from this file.

## Provenance

- Frozen source: `5f4a4a25018f7567fd1b35b93fb1edf57ff86ac7`
- Pinned FLA: `80e494f6c588e091fc8316b612870df29375c5b8`
- Frozen wheel pair re-hashed this session and byte-identical to the
  2026-09-08 record (`13402378...` / `0ff2d45e...`).
- Raw evidence root on the RTX 4080:
  `/home/wzu/codex-run/results/rwkv7-freeze-5f4a4a25/4080` (1.1 GB, intact).

## What changed since the 2026-09-08 pause

1. **The RTX 4080 host is reachable again.** The 2026-09-08 closeout was blocked
   on "4080 offline, 4090 SSH timeout"; the 4080 block is lifted. The 4090
   remains unreachable.
2. **The freeze model directories were reconstructed byte-identically.** The
   2026-09-23 model cleanup deleted `codex-run/models/native-v010-530d935`,
   which the `freeze-5f4a4a25` clean model directories symlinked for all data
   files. Every file was restored and verified against the SHA-256 set recorded
   in the 2026-08-31 lane provenance (`models.json`, preserved in the rerun
   evidence directory): 39/39 files across 0.1b/0.4b/1.5b match. Weights come
   from the published Hub v0.9.0 snapshots (single files verified directly; the
   1.5b single file was re-merged from its six published shards and matches the
   recorded digest). Small metadata files were regenerated with the
   `native-v010` converter at `530d935b` under transformers 5.8.0 and verified
   file-by-file. See `status.json` for the per-file source map.
3. **A repair run is executing** under
   `/home/wzu/codex-run/run-freeze-repair-20261025.sh`:
   - the five open reference units (`1.5b-b8` × hellaswag, winogrande,
     arc_easy, arc_challenge, openbookqa), and
   - the targeted graph-disabled retry of optimized `1.5b-b8-wikitext`
     (`RWKV7_NATIVE_PREFILL_GRAPH=0`), the retry environment already prescribed
     by the 2026-09-08 record, followed by
   - the corrected three-way validation (`validate_lm_eval_three_way.py`
     `--require-model-routes`).
   Rerun evidence (manifest-before snapshots, stashed interrupted rows) is kept
   under `lm-eval-rerun-evidence/reference-1.5b-b8-20261025/` on the host.
4. **Environment deviation, recorded on purpose:** the rerun resolves datasets
   through `HF_ENDPOINT=https://hf-mirror.com` because `huggingface.co` is no
   longer reachable from this host; the 2026-08-31 formal run resolved datasets
   online. Dataset caches were re-fetched through the mirror before launch. No
   other environment change: same pinned venv (`rwkv7-native-v010-4c86709`),
   same CUDA 13.0 toolkit overlay, same frozen wheels.

## Open items (unchanged or newly measured)

- **training-adaptive `b4-t128` gradient gate.** The 2026-08-31 report shows
  10 of 12 cases passing; both `b4-t128` cases fail only when the atomic
  optimized training program is selected (global gradient relative_l2 0.0283
  against the 0.025 contract, cosine 0.9996 passes). This is a numerical
  property of the frozen kernels; a code fix requires an explicit source thaw
  and a new version per `RELEASE_SOURCE_FREEZE.json` policy. A rerun to confirm
  reproducibility is queued behind the repair run.
- **fla three-way trailing OOM.** Both report JSONs in
  `fla/{inference,training}-three-way.json` were already written with
  `status: passed` before the process died allocating in `_linear_reference`
  with CUDA-graph private pools resident. A graph-disabled rerun of the
  trailing stage is queued; the written reports are untouched originals.
- **RTX 4090 device matrix** has never run for 1.0.0.

## Scope of this closeout

- Documentation only. No frozen file was modified; this record adds
  `results/closeout/20261025/` alongside the 2026-09-08 record.
- No publication step (GitHub release, PyPI, Hub tagging) was started; those
  remain gated on the repair run outcome and the items above.
