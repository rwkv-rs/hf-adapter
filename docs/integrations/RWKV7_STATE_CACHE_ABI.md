<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  reference_implementation: rwkv7_hf/native_model.py::NativeRWKV7Cache
  metadata: ../reference/provenance.yaml
  license: MIT
-->

# RWKV-7 serving state-cache ABI

## Why this is not a KV cache

RWKV-7 decode carries a fixed-size recurrent state per request. Its memory
does not grow with sequence length. Paged-attention block allocation and KV
block tables are therefore the wrong storage abstraction.

The Hugging Face `NativeRWKV7Cache` is a semantic reference. A serving engine
should implement a persistent state-slot pool tied to scheduler request IDs.

## Logical request state

For one request:

```python
RWKV7RequestState = {
    "recurrent_state": list[L],  # each [H,N,N], normally FP32
    "attn_prev":       list[L],  # each [D], activation dtype
    "ffn_prev":        list[L],  # each [D], activation dtype
    "v_first":         [A],      # activation dtype
    "seen_tokens":     int64,
}
```

`v_first` is overwritten by layer 0 for each new token, but remains in the
logical ABI for compatibility, CUDA Graph replay, mid-layer handoff, and
pipeline parallel execution.

Approximate state bytes per request:

```text
L * H * N * N * sizeof(state_dtype)
+ 2 * L * D * sizeof(activation_dtype)
+ A * sizeof(activation_dtype)
+ metadata
```

There is no `sequence_length` multiplier.

## Recommended physical layout

Use layer-major tensors so a layer kernel sees contiguous scheduled rows:

```text
S_pool       [L, capacity, H, N, N]
xpa_pool     [L, capacity, D]
xpf_pool     [L, capacity, D]
vfirst_pool  [capacity, A]
seen_tokens  [capacity] int64
allocated    [capacity] bool/bitmap
generation   [capacity] uint64
```

`generation` prevents a stale asynchronous operation or cache handle from
writing into a slot that has been freed and reassigned.

An alternative slot-major layout is valid if measured kernels benefit, but
the external ABI must not expose layout assumptions.

## Required allocator interface

```python
class RWKV7StatePool:
    def allocate(self, request_ids) -> slot_ids:
        """Return zero-initialized, uniquely owned slots."""

    def free(self, slot_ids, generations) -> None:
        """Invalidate handles before making slots reusable."""

    def reset(self, slot_ids) -> None:
        """Restore exact new-request state."""

    def clone(self, source_slots, destination_slots) -> None:
        """Deep-copy state for beam/fork/prefix copy-on-write."""

    def gather(self, layer_id, slot_ids):
        """Return the scheduled layer state or indexed views."""

    def scatter(self, layer_id, slot_ids, updated):
        """Commit updates to exactly those live slots."""

    def snapshot(self, slot_id):
        """Return an immutable/offloaded prefix-cache entry."""

    def restore(self, entry, destination_slot):
        """Validate fingerprints and copy state into a live slot."""
```

Production kernels may consume pool pointers and `slot_ids` directly, removing
the gather/scatter copies.

## Scheduler contract

For every execution batch, the scheduler supplies:

```text
request_ids
slot_ids
slot_generations
token_ids or packed token ranges
operation: prefill | decode
active mask for padded graph rows
```

Before launch:

- every request has exactly one mutable destination slot;
- slot generation matches the scheduler handle;
- request order and slot order are explicit;
- no duplicate destination slot exists unless the operation is read-only.

After launch:

- only active destination slots changed;
- each active decode request advanced by exactly one token;
- each prefill request advanced by its valid token count;
- failed/cancelled launches do not expose partially committed state.

For failure atomicity, compute into temporary scheduled buffers and commit at
the end, or use a generation/transaction marker around direct writes.

## Continuous/dynamic batching

Example:

```text
time 0 scheduled requests: A B C D
time 1 scheduled requests: A C D E
slot ids:                  7 2 9 4
```

The model must gather or index slots `[7,2,9,4]`; it must not assume that the
previous dense row order is preserved.

Required operations:

- insert a new request without resetting existing requests;
- remove completed/cancelled requests;
- compact or reorder scheduled rows;
- support mixed prompt lengths;
- fork a state for beam/speculative candidates;
- return a slot to the pool without cross-request leakage.

`NativeRWKV7Cache.select_batch`, `batch_repeat_interleave`, `reorder_cache`,
and `reset` show the expected semantics, but allocate new tensors and are not
the target serving implementation.

## CUDA Graph buckets

Graph capture normally requires fixed row shapes. Maintain buckets such as:

```text
1, 2, 4, 8, 16, 32, ...
```

If `active_rows < bucket_rows`:

- fill unused `slot_ids` with a dedicated scratch slot;
- set `active_mask=False`;
- prevent scratch rows from changing live state;
- exclude scratch logits from sampling;
- do not increment `seen_tokens`.

Never duplicate a live slot to fill graph rows: two graph rows would race on
the same state.

Graph runners that bind persistent state pointers must be invalidated when:

- pool storage is reallocated;
- dtype/layout version changes;
- quantization/kernel profile changes;
- slot mapping is embedded in captured arguments.

## Prefix-state cache

### Key

Use a collision-safe key over:

```text
model/checkpoint content fingerprint
config and state-layout version
tokenizer/vocabulary fingerprint
adapter/LoRA identity
quantization format/group size
exact prefix token IDs
```

String prompts are insufficient because tokenization settings can differ.

### Value

```text
immutable state snapshot
seen_tokens
prefix length
state dtype/layout
device/offload location
checksum or validation metadata
last-access/size for eviction
```

An entry represents state after consuming the prefix. Restore it into a newly
owned slot, then process only the suffix.

### Ownership

Prefix entries are immutable. Two live requests may reference one immutable
CPU snapshot, but they must not mutate the same GPU state slot. Implement:

- eager clone on hit; or
- copy-on-write with a refcount and private destination on first mutation.

### Cacheability

Do not cache:

- a state from a partially failed execution;
- a state with uncommitted quant/adapter changes;
- padded tokens as if they were real tokens;
- a state whose layout/fingerprint is unknown.

## Chunked prefill

For request `r`, with incoming state `S0` and chunks `c0...cn`:

1. build each chunk's local projections/summary without crossing request
   boundaries;
2. combine summaries in request token order to obtain chunk start states;
3. apply each chunk from its start state;
4. commit only the request-final state to its slot;
5. return logits for the last valid token, plus any logprobs requested by the
   engine.

Mixed requests may have different chunk counts. Use `cu_seqlens` plus
request-local chunk offsets; a global prefix scan without segment boundaries
will corrupt state.

Partial final chunks must be masked or handled by the sequential oracle.

## `seen_tokens`

Maintain one counter per request slot. It is metadata, not a recurrent tensor.

- new request: `0`;
- prefill: add valid prompt tokens actually consumed;
- decode: add `1`;
- prefix restore: set cached prefix length, then add suffix length;
- clone/fork: copy value;
- reset: `0`.

Do not use one scalar for a dynamic batch. Requests with different lengths are
normal.

## Cropping and rollback

A recurrent state cannot be cropped to an arbitrary earlier positive prefix
without a stored snapshot. To rollback:

- restore an earlier prefix snapshot; or
- rerun prefill for the desired prefix.

Reset to zero length is always supported.

Speculative decoding should retain a state snapshot at the verification
boundary or compute candidate state in scratch slots, then commit only
accepted tokens.

## Offload

CPU/NVMe offload can store immutable prefix snapshots. Record:

- source dtype/layout;
- pinned-memory status;
- completion event for asynchronous copies;
- checkpoint/adapter/quant fingerprints.

Do not schedule a restored slot until the copy completion event is visible to
the model stream.

## Pipeline parallel state ownership

Each PP rank owns `S/xpa/xpf` only for its layer range. The rank containing
layer 0 produces `v_first` for the current token; it must travel with the
residual activation to all later stages that need it.

At request cancellation, every stage must free the corresponding local slot.
Use a shared logical request handle plus per-stage physical slot mapping.

## Telemetry

Expose at least:

```text
allocated/free slots
bytes per slot and total state bytes
allocation failures
prefix hits/misses/evictions
prefix bytes restored
state gather/scatter bytes
clone/copy-on-write count
stale-generation rejection count
active rows and graph bucket rows
state-pool high-water mark
```

Cache hit rate alone is insufficient; report saved prefill tokens and restored
bytes.

## Correctness tests

1. zero state matches a fresh native HF cache;
2. one-token update matches every state component;
3. row reorder preserves request histories;
4. insertion/removal does not change surviving-request logits;
5. duplicate/fork produces independent future states;
6. reset removes all prior-request influence;
7. inactive graph rows do not mutate live slots;
8. prefix hit equals full prefill;
9. chunk sizes produce equivalent final state;
10. stale slot generations are rejected;
11. cancellation cannot commit late kernel output;
12. long mixed-length decode remains equal to independent B1 runs.
