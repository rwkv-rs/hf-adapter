<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  metadata: ../reference/provenance.yaml
  license: MIT
-->

# Hugging Face native tensor parallelism

The Native RWKV-7 model exposes a Transformers `base_model_tp_plan`. This is
weight tensor parallelism driven by `torch.distributed`; it is distinct from an
Accelerate `device_map`, which places complete layers on different devices.

## Run

First refresh an existing converted model so its remote-code files contain the
TP plan:

```bash
python scripts/sync_hf_adapter_code.py /path/to/rwkv7-hf
```

Then launch one process per GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  tests/test_tensor_parallel_generate.py \
  --model /path/to/rwkv7-hf --dtype fp16 --max-new-tokens 4
```

Application code uses the standard Transformers load argument:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    dtype=torch.float16,
    tp_plan="auto",
).eval()
```

Initialize `torch.distributed` and select the local CUDA device before loading
when writing a custom launcher. Do not combine `tp_plan` and `device_map`.

## Current partition

| Component | Transformers style | Runtime result |
|---|---|---|
| Embedding table | `embedding_rowwise` | vocabulary rows are sharded; embedding output is reduced |
| R/K/V projections | `colwise_gather_output` | output rows are sharded; complete head vectors are gathered before WKV |
| Attention output | `rowwise_split_input` | input columns are sharded; partial residual outputs are reduced |
| RWKV low-rank projections | `colwise_gather_output` | both numbered linears are sharded and gathered |
| FFN key/value | `colwise` → `rowwise` | intermediate activation stays rank-local between the two linears |
| LM head | `colwise_gather_output` | vocabulary rows are sharded; complete logits are gathered |

The recurrent WKV state is currently **replicated on every TP rank**. That is a
deliberate correctness boundary: the existing WKV kernels consume complete head
tensors. A future head-local WKV implementation can shard the state and replace
the gathered attention projections, but must first pass the same logits, cache,
generation, and per-rank shape gates. Packed single-device JIT/graph paths fail
closed to eager module calls under TP so that Transformers communication hooks
cannot be bypassed.

All dimensions partitioned above must be divisible by the TP world size. Dense
fp16 inference is the validated lane. Quantized TP and TP training require
separate evidence and are not implied by this result.

## Accepted evidence

The two-V100 fp16 run in
[`../../bench/v100_transformers_tp_20260726/`](../../bench/v100_transformers_tp_20260726/)
checks B1/B8 local shard shapes, complete logits, cached greedy generation, rank
agreement, recurrent-state ownership, eager fail-closed routing, and local peak
VRAM. Both batches produced exact greedy parity; minimum logits cosine was
`0.99999821`, and local peak ratios were `0.52031x/0.611611x` versus the B1/B8
single-GPU references.
