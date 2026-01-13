<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  reference_implementation:
    - rwkv7_hf/native_quant.py
    - rwkv7_hf/native_quant_mm8.py
    - rwkv7_hf/native_quant_mm4.py
    - rwkv7_hf/native_quant_a8w8.py
    - rwkv7_hf/native_quant_marlin.py
    - rwkv7_hf/sm70_quant.py
  metadata: ../reference/provenance.yaml
  license: MIT
-->

# RWKV-7 W8/W4 porting contract for vLLM

## Scope and terminology

This document specifies the quantized Linear representations used by the
adapter and how to expose them through a serving-engine quantization method.

The current native packers are RTN/affine quantizers. They are **not** an
OBQ/GPTQ calibration implementation. The Marlin route uses a
GPTQ-compatible symmetric packed representation and Marlin compute kernel,
but that does not make the quantization algorithm GPTQ.

The canonical converted checkpoint is dense safetensors. Most native routes
pack weights at model load. A serving engine may:

1. load dense weights and pack once; or
2. define a versioned prepacked checkpoint format with explicit metadata.

Never infer a packed layout from a filename alone.

## Common Linear contract

For a dense PyTorch/HF Linear:

```text
weight          [O,I]
input           [...,I]
output          [...,O]
bias            [O] or absent
```

A vLLM quant method should expose:

```python
class RWKVQuantLinearMethod:
    def create_weights(self, layer, input_size, output_size, params_dtype):
        ...

    def process_weights_after_loading(self, layer):
        ...

    def apply(self, layer, x, bias=None):
        ...
```

Use separate methods/layout IDs for incompatible packed representations.

## Format 1: native signed row-wise W8

Reference: `rwkv7_hf/native_quant.py`.

Packing:

```python
scale[o] = max(abs(weight[o,:]), eps) / 127
q[o,i] = clamp(round(weight[o,i] / scale[o]), -127, 127)
```

Buffers:

```text
q_weight  int8     [O,I]
scales    float32  [O]
bias      original dtype [O], optional
```

Dequantization:

```text
weight_hat[o,i] = float(q_weight[o,i]) * scales[o]
```

This is a simple correctness/fallback format. The Triton GEMV dequantizes in
registers and must not materialize a full dense weight on the production path.

Suggested layout ID:

```text
rwkv7.rowwise_w8.v1
```

## Format 2: native signed row-wise W4

Reference: `rwkv7_hf/native_quant.py`.

Packing:

```python
scale[o] = max(abs(weight[o,:]), eps) / 7
q[o,i] = clamp(round(weight[o,i] / scale[o]), -7, 7)
```

Two signed two's-complement nibbles are stored in one byte:

```text
low nibble  = input feature 2*j
high nibble = input feature 2*j+1
```

Buffers:

```text
q_weight  uint8    [O,ceil(I/2)]
scales    float32  [O]
in_features int
bias      original dtype [O], optional
```

Nibble decode:

```python
u4 = nibble & 0xF
q4 = u4 - 16 if u4 >= 8 else u4
weight_hat = q4 * scale[o]
```

If `I` is odd, the unused final high nibble is zero. Kernels must mask it.

Suggested layout ID:

```text
rwkv7.rowwise_w4.v1
```

## Format 3: RWKV MM8 affine

Reference: `rwkv7_hf/native_quant_mm8.py`.

The packer receives transposed dense weight:

```text
W = dense_weight.T, shape [I,O]
```

Buffers:

```text
w_u8  uint8             [I,O]
mx                         [O]
rx                         [O]
my                         [I,1]
ry                         [I,1]
```

`mx/rx/my/ry` use the source weight dtype. The dequantized matrix is:

```text
W_hat = (w_u8 + 0.5) * ry * rx + my + mx
output = input @ W_hat
```

The quantizer subtracts minima and normalizes ranges in an order selected from
`I > O`, matching the official RWKV fp16i8 representation. See
`quantize_mm8` for exact rounding and scale factors.

Suggested layout ID:

```text
rwkv7.mm8_affine.v1
```

## Format 4: RWKV MM4 affine

Reference: `rwkv7_hf/native_quant_mm4.py`.

The packer also receives:

```text
W = dense_weight.T, shape [I,O]
```

The output-feature dimension is padded to even and packed:

```text
packed uint8 [I,O_padded/2]
low nibble  = output feature 2*j
high nibble = output feature 2*j+1
```

Buffers:

```text
packed   uint8              [I,O_padded/2]
mx                           [O_padded]
rx_s                         [O_padded]
my                           [I,1]
ry_s                         [I,1]
m_orig                       original O
```

Dequantization:

```text
W_hat = (u4 + 0.5) * ry_s * rx_s + my + mx
W_hat = W_hat[:, :m_orig]
output = input @ W_hat
```

This unsigned affine MM4 layout is different from the signed row-wise W4
layout above.

Suggested layout ID:

```text
rwkv7.mm4_affine.v1
```

## Format 5: SM7x DP4A W8/W4

Reference: `rwkv7_hf/sm70_quant.py`.

These are measured Volta/Turing paths:

- W8 row-wise weights plus DP4A kernels;
- W4 row-wise packed weights;
- W4 group-wise weights with group size `128` or `256`;
- optional fused FFN-key ReLU-square and residual-add epilogues.

The group-wise W4 kernel contract includes:

```text
packed weight uint8 [O,I/2]
scale         fp16  [O,I/group_size]
group_size          128 or 256
activation/output   fp16
I divisible by group_size and by 8
```

Dispatch is intentionally restricted to:

- compute capability 7.0; or
- compute capability 7.5 with a validated Tesla T4 identity.

Other SM7x products must fall back until validated.

Suggested layout IDs:

```text
rwkv7.sm7x_rowwise_w8.v1
rwkv7.sm7x_rowwise_w4.v1
rwkv7.sm7x_group_w4.v1
```

## Format 6: dynamic A8W8

Reference: `rwkv7_hf/native_quant_a8w8.py`.

Weights:

```text
q_weight_t    int8     [I,O]
weight_scale  float32  [O]
```

Each input row is dynamically quantized:

```text
activation_scale[row] = max(abs(x[row,:])) / 127
q_activation          = round/clamp to int8
```

INT8 dot products are rescaled by
`activation_scale[row] * weight_scale[o]`.

The reference native policy often applies this only to selected matrices such
as `lm_head`; a vLLM implementation should benchmark prefill and decode
separately before widening module coverage.

Suggested layout ID:

```text
rwkv7.dynamic_a8w8.v1
```

## Format 7: symmetric group-wise Marlin W4

Reference: `rwkv7_hf/native_quant_marlin.py` and
`rwkv7_hf/csrc/marlin/`.

Requirements in the retained implementation:

```text
weight/activation dtype  BF16
group_size               32, 64, or 128
I % group_size           0
I % 8                    0
O % 64                   0
```

For each output/group:

```python
amax = max(abs(values))
scale = 2 * amax / 15
u4 = clamp(round(values / scale + 8), 0, 15)
```

Eight nibbles are initially packed into `int32`, then repacked into Marlin's
U4B8 layout. Scales are permuted for the kernel.

Persistent buffers include:

```text
qweight
scales
workspace int32
empty metadata tensors required by the op
bias, optional
```

The exact shape of `qweight` after Marlin repack is an implementation detail of
the vendored op. A portable checkpoint must record:

```text
layout ID
Marlin format/type ID
I/O
group size
scale dtype/order
repack implementation/version
```

Do not load a generic GPTQ tensor directly into this op without verifying
nibble bias, scale permutation, zero-point semantics, and repack version.

Suggested layout ID:

```text
rwkv7.marlin_u4b8_bf16.v1
```

## Fused epilogues

Quantized Linear and activation are separate unless the method advertises a
specific ABI.

Supported conceptual methods:

```python
linear.forward(x)                 # plain Linear
linear.forward_relu2(x)           # relu(linear(x)) ** 2
linear.forward_add(x, residual)   # linear(x) + residual
linear.forward_into(x, out)       # write to graph-stable output
```

`forward_relu2` is valid only for `model.layers.*.ffn.key`. Never globally
replace plain `forward` with the fused activation.

## Module-selection policy

Quantizing every large Linear minimizes weight footprint but may reduce speed.
Treat module coverage as part of the quantization profile:

```text
lm_head
attention r/k/v/o projections
FFN key
FFN value
low-rank projections
```

Record the exact module allowlist or policy in checkpoint metadata.

Typical integration order:

1. `lm_head` for a low-risk speed/footprint path;
2. FFN key/value with fused epilogues;
3. attention projections;
4. low-rank projections only after measured benefit.

## BN/TN and launch schedules

BN/TN are kernel scheduling parameters, not quantization formats:

- `BN`: output columns owned by a block/CTA;
- `TN`: output values accumulated/written by a thread or writer.

They depend on GPU, matrix shape, row count, dtype, group size, and kernel
implementation. The retained 5090 Marlin path validates production BN/TN
inside CUDA and fails closed on mismatch. The SM70 path uses separate measured
tables.

A vLLM port should key schedules by:

```text
GPU/runtime fingerprint
layout ID and group size
M/I/O
epilogue
kernel version
```

Unknown keys select the kernel's safe default, not the nearest card profile.

## Loader metadata

A prepacked checkpoint needs a manifest per tensor/module:

```json
{
  "format": "rwkv7.marlin_u4b8_bf16.v1",
  "module": "model.layers.10.ffn.key",
  "input_features": 4096,
  "output_features": 16384,
  "group_size": 128,
  "activation_dtype": "bfloat16",
  "scale_dtype": "bfloat16",
  "epilogue": "relu2",
  "source_weight_sha256": "...",
  "packer_version": "..."
}
```

Validate metadata before allocating GPU buffers.

## Hardware routing

Use a registry such as:

```python
route = registry.resolve(
    backend="cuda",
    capability=(major, minor),
    device_name=device_name,
    activation_dtype=dtype,
    quant_format=format_id,
    group_size=group_size,
    rows=M,
    in_features=I,
    out_features=O,
    epilogue=epilogue,
)
```

The route returns:

```text
kernel
workspace bytes
alignment/padding
graph safety
validated profile ID
fallback
```

Keep correctness fallback available on every supported device.

## Acceptance gates

For every promoted format/card/model/batch pair:

1. packed footprint is lower than dense W16;
2. runtime peak VRAM is recorded separately from packed footprint;
3. dequantized/reference Linear output passes tolerance;
4. full-model prompt and decode logits pass;
5. same-next-token and long greedy trace are recorded;
6. prefill and decode speed are separately measured;
7. end-to-end W8/W4 is not slower than matching W16 for a production speed
   claim;
8. cold compile/load time is reported separately;
9. no dense shadow copy remains resident unless explicitly counted;
10. unsupported shapes fail closed to a correct path.

See [`../validation/VLLM_ACCEPTANCE.md`](../validation/VLLM_ACCEPTANCE.md).
