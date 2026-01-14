# RTX 4090 non-regression for PR #70 — 2026-07-23

This is the cross-card control for `wangyue/5090-full-regression-fixes`
(`be289cb`) against `main` at `3788176`.

## Environment

- NVIDIA GeForce RTX 4090, SM89, 24 GiB
- driver `550.142`
- PyTorch `2.5.1+cu124`
- Triton `3.1.0`
- Transformers `5.14.1`
- bitsandbytes `0.49.2`
- PEFT `0.19.1`

The official g1d 0.1B checkpoint was downloaded from ModelScope and converted
to the native HF format with each tree's own remote code.

## Dispatch isolation

The production policy selected the RTX 4090 Ada profile:

- the new 5090-only `(2048, 24, 8, 128)` prefill graph shape is absent;
- native graph state remains FP32;
- deterministic sparse splits remain `0`;
- 5090 low-memory sparse FFN relayout remains disabled.

This confirms that the PR does not alias `4090` to the exact `5090` gates.

## Tests

- Changed/shared unit surface: **42 passed**.
- Whole repository: **615 passed, 12 skipped, 1 unrelated failure**.
- The one failure is the Apple rendered acceptance document depending on
  benchmark artifacts omitted from the archive-based test checkout.  The
  unmodified `main` control fails the exact same assertion with the exact same
  `39 / 149` versus `50 / 149` difference, so it is not introduced by PR #70.

See [`selected_pytest.log`](selected_pytest.log),
[`full_pytest.log`](full_pytest.log), and
[`main_apple_control.log`](main_apple_control.log).

## Real-model B1/B8 smoke

Both lanes pass prefill, cached native-graph decode, greedy generation and
finite-logit checks:

| Batch | Prefill | Decode | Prefill backend | Decode backend | Peak VRAM |
|---:|---:|---:|---|---|---:|
| 1 | 23,210.1 tok/s | 1,170.0 tok/s | native prefill graph | native graph | 491.2 MiB |
| 8 | 142,397.4 tok/s | 6,852.4 tok/s | native prefill graph | native graph | 549.6 MiB |

See [`smoke_4090.log`](smoke_4090.log).

## Main versus PR performance control

Fresh processes were alternated `main -> PR -> main -> PR`; every process used
nine timed samples after graph warmup.

| Batch | Prefill main | Prefill PR | Ratio | Decode main | Decode PR | Ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23,457.0 | 23,465.0 | `1.0003x` | 1,215.4 | 1,207.8 | `0.9937x` |
| 8 | 144,428.3 | 144,503.7 | `1.0005x` | 7,102.7 | 7,102.9 | `1.0000x` |

The maximum observed delta is the B1 decode `-0.63%`, inside the 3% quick
non-regression tolerance and with unchanged effective backends.  Raw samples
are in [`compare_4090.jsonl`](compare_4090.jsonl).

## External quantization smoke

BnB W8 and W4 both pass load, forward, cached eager decode and greedy generate:

- W8: 72 quantized linears, finite logits, 319.5 MiB peak.
- W4: 72 quantized linears, finite logits, 387.9 MiB peak.

Eager decode is expected for these external quant modules.  See
[`bnb_4090.log`](bnb_4090.log).

## Verdict

**RTX 4090 quick cross-card gate passes.**  PR #70 preserves dispatch,
correctness and B1/B8 performance on the tested official 0.1B model.
