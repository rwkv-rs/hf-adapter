# Pinned official-to-Native inference alignment

`final_report.json` compares separate-process captures of the official Space
commit `cc57df475465c6cacd42ecd4f2f05a588ee5473b` and the Native HF route on the
same g1h 7.2B weights. The seven official source files are verified by
`official_source_manifest.json` before import.

The prompt is consumed token-by-token from explicit zero FP16 state at B1 and
B8, followed by 16 greedy decode steps. Both rows pass exact greedy and top-1
agreement. Logits require cosine at least `0.9999` and max absolute difference
at most `0.125`; recurrent state allows max `1.0`; xpa/xpf allow max `0.125`.
All tensors must be finite and elapsed state must match exactly.

Observed B8 values include logits cosine `0.999999785`, logits max difference
`0.09375`, prefill-state cosine `0.999999380`, and prefill xpf max difference
`0.078125`. The report stores the SHA256 of both large capture files; the
captures remain on the validation host and are not committed.

Run `scripts/compare_official_native_inference.py --help` for the three-stage
`capture-native`, `capture-official`, and `compare` interface. Source pin,
shape, prompt IDs, state shapes, finite values, numerical thresholds, elapsed
state, logits top-1, and greedy tokens all fail closed.
