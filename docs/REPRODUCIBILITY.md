# Reproducibility artifacts

Local files are the source of truth even when W&B reporting is enabled.

Each evaluation or training run writes:

- resolved configuration and exact command;
- source Git SHA, model revision and dataset revision;
- Python, PyTorch, Transformers, TRL and PEFT versions;
- JSONL metrics and final evaluation;
- checkpoint inventory and SHA256;
- stdout/stderr paths and exit status;
- optional W&B run ID and URL, never a token.

Release bundles live below `results/`. Large task sample logs can be
attached to a release artifact while the manifest and summary stay in Git.
Committed result summaries must be immutable and must identify the exact GPU.

The performance history is intentionally not mixed with the reference line. It
is retained on `perf/native-kernels-v0.8`.
