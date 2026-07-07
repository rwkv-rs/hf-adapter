# Hugging Face v0.9.0 V100 release verification

Status: **passed**.

- All six exact Hub revisions passed metadata and self-contained reference-code checks.
- 0.1B, 0.4B and 1.5B were loaded directly from the Hub with `trust_remote_code=True`; forward, cache and generation passed.
- 2.9B, 7.2B and 13.3B were loaded with the v0.9.0 reference files and server-local single-file safetensors. Their SHA256 and byte size exactly match each Hub repository's `conversion_manifest.json` source conversion. Forward, cached decode, finite logits and generation passed.
- Environment: Python 3.11.15, PyTorch 2.5.1+cu124, Transformers 4.52.4, Tesla V100-PCIE-32GB.
- Source revision: `b8438cab0dc7d11238942efca4c05135d53fcaf8`.

`validation.json` is the machine-readable gate. Model weights, samples, caches and verbose logs are intentionally excluded.
