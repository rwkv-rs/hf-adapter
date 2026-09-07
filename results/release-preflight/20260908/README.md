# Release preflight — 2026-09-08

**Not a final release bundle. `release_ready: false`.** The user resumed the
existing acceptance task, but RTX 4080 is offline and RTX 4090 SSH times out.
No GPU jobs or recurring monitors were started. V100 remains excluded from
GPU acceptance; its existing jump host was used for isolated CPU packaging.

## Verified in this attempt

| Check | Result | Scope |
|---|---|---|
| Full local pytest | 470 passed; 409 deprecation warnings | CPU, Python 3.11.15 / Torch 2.12.1 / Transformers 5.12.1 |
| Source freeze | 155/155 hashes unchanged | No model, package, kernel or threshold changes |
| Main CI | full-tests, CPU smoke, HF ecosystem successful | Main `2d0426e0`; tree identical to frozen `5f4a4a25` |
| Wheel reconstruction | Both original SHA256 values match | CPU-only Linux build; not a new kernel version |
| Wheel/package audit | Passed | 149 owned files; 102 migrated NVIDIA files; API v4; metadata/RECORD/license |
| Source archives | Passed | New sdists match frozen checkout and wheel contents |
| Twine | 4/4 archives passed strict checks | Packaging only; nothing uploaded |
| Installed-wheel smoke | Passed | CPU forward/cache, greedy/beam, AutoConfig/AutoModel, save/reload, loss/backward |
| Converter CLI | Passed | Synthetic tiny official-format checkpoint, FP32, `--low-memory` |
| Package-free directory | Passed | Neither project package visible; AutoModel/AutoTokenizer, exact fixture logits, cached generation |

The smoke environments isolate project packages but share existing dependency
files. The package-free process has neither `rwkv7_hf`, `rwkv7_hf_tools` nor
`rwkv7_kernels` importable, and no source checkout on `sys.path`. This is a
local synthetic fixture, **not** official-model accuracy, an independently
provisioned GPU environment or a Hub redownload.

The installed fixture's cached/full logits max-abs difference is `0.0` and
its finite causal loss is `4.1649651527404785`. The converted fixture's
package-free logits also match exactly. Those numbers describe these tiny
CPU fixtures only, not the three real-model lm_eval matrix.

## Identical wheel bytes recovered

| Artifact | SHA256 | Size (bytes) |
|---|---|---:|
| `rwkv7_hf-1.0.0-py3-none-any.whl` | `13402378207a59ac90fa69d947402a7177de69e8c77d66ab14aa7803cae25854` | 47897 |
| `rwkv7_kernels-1.0.0-py3-none-any.whl` | `0ff2d45e911a5cd741fd71ac956898d862fc0e7b62c984c3e36e2d755da86d11` | 530324 |

Source: `5f4a4a25018f7567fd1b35b93fb1edf57ff86ac7`,
`SOURCE_DATE_EPOCH=1788148092`, Python 3.12.2, `umask 022`.
See [the build recipe](../../../docs/REPRODUCIBILITY.md) and
`build-environment.txt`. Use a fresh source directory; an initial extraction
under `umask 002` produced different ZIP file-mode metadata and therefore
different hashes. It was not promoted. The corrected build changes no source
bytes. No historical sdist identity is claimed.

Both correct wheels and the audited sdists are retained outside Git. This
bundle deliberately contains no weights, wheels, checkpoints, sample logs,
tokens or connection credentials. `attempts.json` preserves the failed
dependency-install/build/smoke attempts and their resolutions instead of
reporting only the final successes.

## Evidence files

- `status.json`: scope, source/freeze identity, connectivity and pending gates.
- `cpu-tests.json`, `cpu-tests.txt`: full local suite, with the local venv path
  redacted in the text log; JSON retains the original log SHA256.
- `publication-preflight.json`: live main CI links and 0.9.0 publication state.
- `artifacts.json`, `build-environment.txt`: build environment and all four
  archive hashes. The wheel hashes match the previous acceptance pair.
- `wheel-audit.json`, `source-archive-audit.json`, `twine-check.txt`: existing
  release validators' packaging results, not hand-authored GPU gate results.
- `cpu-wheel-smoke.json`, `conversion-smoke.json`, `package-free-smoke.json`:
  targeted CPU fixture results and explicit limitations.
- `attempts.json`: infrastructure failures and unpromoted attempts.
- `MANIFEST.sha256`: every file in this compact preflight directory except
  the manifest itself.

The wheel/source audits use the unchanged `audit_release_wheels.py` and
`verify_release_assets.py` validators. The full test command is recorded in
`cpu-tests.json`; run it from the frozen source with the recorded environment.

## Still required

1. Restore RTX 4080 access and retrieve original evidence. Verify exact wheel,
   model-directory code, loaded modules, routes, weights and dataset identity.
2. Finish the one optimized OOM retry and five incomplete reference units,
   preserving old attempts, then run the unchanged three-way metric validator.
3. Close final-wheel 4080 correctness/HF/training/finetune/speed evidence;
   complete RTX 4090 with the same wheel pair, after 4080.
4. Close all six Hub tags/download smokes, GitHub release provenance/assets,
   both PyPI artifacts and post-publication byte audits.

The [historical September 1 snapshot](../../closeout/20260908/README.md) is
unchanged. This preflight cannot turn its 47/48 optimized exits into 48/48,
nor does the existing 48/48 FLA exit count prove metric equivalence on its own.
