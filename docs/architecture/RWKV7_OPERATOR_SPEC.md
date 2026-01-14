<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  reference_implementation: rwkv7_hf/native.py
  upstream_math: https://github.com/BlinkDL/RWKV-LM
  metadata: ../reference/provenance.yaml
  license: MIT
-->

# RWKV-7 operator specification

## Purpose

This document separates the RWKV-7 block mathematics from Transformers,
FLA, vLLM, SGLang, CUDA Graph, and quantization wrappers. It is the contract
for implementing a new runtime.

The official RWKV-LM implementation remains the source of the RWKV-7
algorithm. `rwkv7_hf/native.py` is the canonical tensor-layout oracle for this
adapter.

## Symbols and layouts

| Symbol | Meaning |
|---|---|
| `B` | active request rows |
| `T` | tokens in a dense sequence |
| `L` | number of blocks |
| `D` | residual/hidden width |
| `H` | recurrent attention heads |
| `N` | head dimension |
| `A` | attention width, exactly `H*N` |
| `F` | FFN intermediate width |
| `Rw/Ra/Rg/Rv` | low-rank dimensions for W/A/G/V |

Reference decode tensors:

| Tensor | Shape | Reference dtype |
|---|---:|---|
| residual `x` | `[B,D]` | activation dtype |
| attention previous input `xpa[i]` | `[B,D]` | activation dtype |
| FFN previous input `xpf[i]` | `[B,D]` | activation dtype |
| recurrent matrix `S[i]` | `[B,H,N,N]` | FP32 eager reference |
| cross-layer value `v_first` | `[B,A]` | activation dtype |

The implementation supports `A != D`. Do not infer attention width from the
residual width.

## Model structure

```text
token ids
  -> embedding [D]
  -> L x RWKV7 block
  -> final LayerNorm
  -> independent lm_head [vocab,D]
  -> logits
```

The embedding and output head are not tied.

Each block contains:

```text
optional pre_norm (layer 0)
attention LayerNorm
RWKV-7 time mix / recurrent operator
residual add
FFN LayerNorm
channel mix / FFN
residual add
```

## State initialization

For a new request:

```python
for i in range(L):
    S[i]   = zeros([H, N, N], float32)
    xpa[i] = zeros([D], activation_dtype)
    xpf[i] = zeros([D], activation_dtype)
v_first = zeros([A], activation_dtype)
seen_tokens = 0
```

The leading batch/slot dimension is added by the serving runtime.

## Attention/time-mix token step

Inputs for layer `i`:

```text
h        [B,D]       normalized current residual
x_prev   [B,D]       xpa[i]
S        [B,H,N,N]   recurrent state
v_first  [B,A]       current token's layer-0 value
```

### 1. Time mixing

```python
delta = x_prev - h
xr = h + delta * x_r
xw = h + delta * x_w
xk = h + delta * x_k
xv = h + delta * x_v
xa = h + delta * x_a
xg = h + delta * x_g
```

The six learned mix vectors have `D` elements and broadcast across `B`.

### 2. Projections

Using linear storage `[out_features,in_features]`:

```python
r = linear(xr, W_r)                         # [B,A]
w_raw = Ww_up(tanh(Ww_down(xw)))            # [B,A]
k = linear(xk, W_k)                         # [B,A]
v = linear(xv, W_v)                         # [B,A]
a = sigmoid(Wa_up(Wa_down(xa)))             # [B,A]
g = Wg_up(sigmoid(Wg_down(xg)))             # [B,A]
```

`Ww_up`, `Wa_up`, and layer `i>0` `Wv_up` have bias in the native module.
The G path is bias-free.

### 3. Key adaptation and first-layer value coupling

```python
kk = l2_normalize(
    reshape(k * k_k, [B,H,N]),
    axis=-1,
)
k = k * (1 + (a - 1) * k_a)

if i == 0:
    v_first = v
else:
    v_mix = sigmoid(Wv_up(Wv_down(xv)))
    v = v + (v_first - v) * v_mix
```

`v_first` is logically a **current-token cross-layer value**. Layer 0 replaces
it for every token before later layers consume it. It is carried in the HF
cache for a uniform interface and graph replay; a pipeline-parallel runtime
must transfer it from the stage containing layer 0 to later stages for the
same token.

### 4. Decay

```python
decay = exp(-exp(-0.5) * sigmoid(float32(w_raw)))
```

The reference constant in code is approximately `0.606531`. Keep the sigmoid
and exponent in FP32 unless an optimized path has passed long-horizon state
and greedy-token tests.

### 5. DPLR recurrent update

Reshape `r,k,v,kk,a` to `[B,H,N]`. Define:

```python
VK = outer(v, k)                 # [B,H,N,N]
AB = outer(-kk, kk * a)          # [B,H,N,N]
```

The state update is:

```python
S_new = S * decay[:, :, None, :] + matmul(S, AB) + VK
```

`decay` scales the **last/column dimension** of `S`. The eager correctness
path computes recurrent state and the `AB/VK` accumulation in FP32.

Equivalent per-head form:

```text
S' = S D + S (-kk)(kk*a)^T + v k^T
```

where `D` is diagonal from `decay`.

### 6. Recurrent readout

```python
o = matmul(S_new_as_activation_dtype, r[..., None])
o = reshape(o, [B,A])
o = group_norm(o, num_groups=H, eps=N * 1e-5)

skip = sum(
    reshape(r,[B,H,N])
    * reshape(k,[B,H,N])
    * r_k[None,:,:],
    axis=-1,
    keepdims=True,
)
o = o + reshape(skip * reshape(v,[B,H,N]), [B,A])
o = linear(o * g, W_o)           # [B,D]
```

Outputs:

```text
attention output  [B,D]
xpa_new           h, [B,D]
S_new             [B,H,N,N]
v_first_new       [B,A]
```

Do not update `xpa` with the post-attention residual. It stores the normalized
input `h` passed into time mix.

## FFN/channel-mix token step

Inputs:

```text
h2       [B,D]   FFN-normalized residual
xpf[i]   [B,D]   previous FFN input
```

Operation:

```python
delta = xpf - h2
k = h2 + delta * x_k
k = relu(linear(k, W_key)) ** 2  # [B,F]
f = linear(k, W_value)           # [B,D]
xpf_new = h2
```

The fused ReLU-square epilogue belongs only to the FFN-key projection. A
generic quantized Linear must remain a plain Linear or the activation can be
applied twice.

## Complete block step

```python
residual = pre_norm(x) if layer_id == 0 else x

h = attn_norm(residual)
attn_out, xpa, S, v_first = attention_step(
    h, xpa, S, v_first
)
x = residual + attn_out

residual = x
h2 = ffn_norm(x)
ffn_out, xpf = ffn_step(h2, xpf)
x = residual + ffn_out
```

Process layers strictly in increasing order for each token.

## Attention mask semantics

For a padded token row:

- do not update `S`, `xpa`, `xpf`, or `v_first`;
- do not advance `seen_tokens` for that request;
- preserve the prior hidden/logit state according to the caller contract.

A serving engine should normally use packed sequences rather than padding.

## Prefill formulations

### Sequential oracle

Run the complete token step from `t=0` to `T-1`. This path is slow but defines
correct state and logits.

### Chunked DPLR/WY

Each token transition has affine form:

```text
S_t = S_(t-1) P_t + Q_t
```

with:

```text
P_t = D_t + AB_t
Q_t = VK_t
```

A chunk summary represents:

```text
S_end = S_start P_chunk + Q_chunk
```

The compact implementation stores diagonal and low-rank factors rather than
materializing dense transition matrices for every token. In this repository,
the explicit compact summary fields are:

```text
transition_diag   [B,chunks,H,N]
transition_left   [B,chunks,H,N,C]
transition_right  [B,chunks,H,N,C]
additive_left     [B,chunks,H,N,C]
additive_right    [B,chunks,H,N,C]
```

where `C` is chunk size/rank. See
`rwkv7_hf/mlx_dplr_prefill.py` for the readable
summary → prefix-combine → chunk-apply oracle, and
`rwkv7_hf/self_chunk_rwkv7.py` for CUDA/Triton inference kernels.

Chunk implementations must return:

- all required per-token recurrent readouts;
- exact request-final `S`;
- last valid token logits;
- state independent of chunk size within tolerance.

## Numerical contract

Correctness should be tested at three levels:

1. **operator:** one layer, one token, fixed state;
2. **transition:** final state after multiple chunks;
3. **behavior:** long greedy trace.

Recommended comparison:

- FP32 reference for formula debugging;
- native HF at the target FP16/BF16 dtype;
- official RWKV-LM reference when checkpoint/layout changes.

Do not rely on cosine similarity alone. Record:

```text
max absolute error
relative L2 error
logit cosine
top-1 equality
state max absolute error per component/layer
greedy trace equality or first divergence position
```

## Kernel ABI checklist

Every fused implementation must document:

- input/output shapes and contiguous strides;
- accumulator and output dtype;
- whether state updates are in place;
- aliasing restrictions;
- supported `B/T/D/H/N/F`;
- alignment and padding;
- graph-capture safety;
- stream semantics;
- fused epilogue, if any;
- fallback behavior;
- hardware capability and measured profile.
