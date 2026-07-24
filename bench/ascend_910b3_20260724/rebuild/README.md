# Rebuilt Ascend 910B3 acceptance

This directory is a second, clean execution of the real 7.2B HF gates after
the original ephemeral server was destroyed. It ran on one Ascend 910B3 with
CANN 8.5.0, torch 2.9.0+cpu, torch_npu 2.9.0 and Transformers 4.57.6.

## Results

| Gate | Result |
|---|---|
| Native HF BF16 load/forward/cache/generate/chunked prefill | pass |
| Independent pinned-FLA CPU oracle vs HF/NPU, 208 tensors | pass |
| Greedy `Hello`, 3 tokens | exact `[45, 308, 459]` |
| Minimum logits cosine / maximum normalized RMSE | 0.99996638 / 0.00790427 |
| Minimum recurrent-state cosine / maximum normalized RMSE | 0.99984443 / 0.01771485 |
| Ragged B2 compact/cache/chunked equivalence | pass |
| Ragged B2 global logits/state minimum cosine | 0.99999988 / 0.99999970 |
| Resident / peak allocated HBM in the oracle candidate capture | 14,433,095,680 / 14,485,243,392 bytes |

The independent reference capture hash is
`35f24f1e0116fcbee548c69ee38652b897f14e6a09dedfad79753a0380e324f9`;
the NPU candidate capture hash is
`8b83cb5af96a250565c713a5dd9f9a3ad0dff7d123659f00fc59fb86d6fe4cbf`.
Those canonical tensor-map hashes match the original run. The rebuilt
candidate safetensors file hash is recorded inside its JSON because safetensors
container metadata/order can change without changing the canonical tensor map.

The two approximately 66 MiB tensor captures are intentionally not committed.
The committed reference/candidate JSON files pin every tensor hash, every model
and tokenizer file hash, and the relevant FLA source hashes; the comparison
fails closed for a missing, changed, or incomplete tensor capture.

## Reproduction order

1. `build_ascend_hf_reference.py`
2. `capture_ascend_hf_candidate.py`
3. `compare_ascend_hf_reference.py`
4. `smoke_ascend_hf_ragged_b2.py`

Exact commands are documented in
[`../../../docs/hardware/HUAWEI_ASCEND.md`](../../../docs/hardware/HUAWEI_ASCEND.md).
