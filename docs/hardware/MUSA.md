# Moore Threads MUSA

This document records the MUSA implementation and exact-card validation boundary
for the RWKV-7 HF adapter. The imported path has now been exercised on one MTT
S70, but the evidence remains intentionally narrow and does not establish broad
MUSA platform support.

## Sources of truth

- Implementation and retained measurements: [KakaruHayate/RWKV-MUSA](https://github.com/KakaruHayate/RWKV-MUSA), commit `6b48752`.
- Runtime and compiler contracts: MUSA SDK 4.2.0 Runtime API, Driver API, and MCC user manual.
- Python APIs: only calls exercised by RWKV-MUSA with torch_musa 2.5.0 are used here. CUDA or ROCm behavior is not inferred.

The imported WKV source is Apache-2.0 code derived from BlinkDL/RWKV-LM through
RWKV-MUSA. Attribution is retained in the source header and repository
provenance documents.

## Imported HF adapter path

The canonical `NativeRWKV7ForCausalLM` remains the public model. On a reported
`musa` device it uses native/no-FLA PyTorch behavior and may use the optional
MUSA WKV forward kernel when all of these conditions hold:

- `torch.musa.is_available()` is true;
- device type is `musa`;
- WKV operands use fp16 IO;
- recurrent state remains fp32;
- head size is exactly 64;
- `RWKV7_MUSA_WKV` is not disabled.

If any condition is absent, or if the optional extension cannot compile, the
adapter uses its existing pure-PyTorch recurrence. The base package remains
importable without torch_musa. The MUSA extension is compiled lazily and does
not modify torch or torch_musa.

## Retained exact-device facts

RWKV-MUSA reports these implementation facts for MTT S70 (`mp_21`, 7 GB), MUSA
SDK 4.2.0 and torch_musa 2.5.0. The same kernel contract has now also been
exercised through this HF adapter on one S70:

- fp16 IO with fp32 recurrence compute/state is the validated WKV route;
- the imported kernel uses block synchronization and no warp shuffle;
- the validated kernel head size is 64;
- bf16 was not validated and is not included in this port;
- Triton/FLA and torch quantization are not enabled by this port;
- CUDA graph, CUDA kernel, ROCm, and other-device policies are not applied to MUSA.

The retained HF evidence is
[`../../bench/musa_s70_validation_20260728/`](../../bench/musa_s70_validation_20260728/README.md).
Standalone output/state parity, model load/forward/cache/generate, eager/WKV
long-trace token equality and differentiable eager fallback passed. In three
paired processes for the exact 0.1B, batch-1, prompt-128/decode-32 fp16 shape,
the WKV route had `1.214072x` median prefill throughput, `1.000000x` median
decode throughput and the same `406.0 MiB` peak allocated memory as eager.
Therefore this result supports a narrow prefill observation only; it does not
support a decode-speed or memory-reduction claim.

## Real-device acceptance gate

Run on the MUSA host before changing the hardware matrix from **Open**. Keep the
fp32 compatibility row, then run paired inference rows with the optional kernel
disabled and enabled. The adapter converts the WKV operands to fp16 IO before
calling the kernel, so record the selected route rather than inferring it from
the model dtype:

```bash
PYTHON_BIN=python DEVICE=musa DTYPE=fp32 \
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
RESULTS=bench/results_musa_fp32.jsonl \
bash scripts/run_hardware_smoke.sh

RWKV7_MUSA_WKV=0 PYTHON_BIN=python DEVICE=musa DTYPE=fp16 \
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
RESULTS=bench/results_musa_fp16_eager.jsonl \
bash scripts/run_hardware_smoke.sh

PYTHON_BIN=python DEVICE=musa DTYPE=fp16 \
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
RESULTS=bench/results_musa_fp16_wkv.jsonl \
bash scripts/run_hardware_smoke.sh

RWKV7_MUSA_WKV=0 python tests/test_hf_api_contract.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device musa --dtype fp16

python tests/test_hf_api_contract.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device musa --dtype fp16

python tests/test_peft_lora.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device musa --attn-mode native
```

The PEFT row must remain on the differentiable pure-PyTorch recurrence because
the imported WKV extension is inference-only and is disabled while autograd is
enabled.

Also record:

- exact MTT device, driver, MUSA SDK, torch_musa, PyTorch and Transformers versions;
- pure-PyTorch versus MUSA-WKV logits, greedy tokens, state maximum error and chunk handoff;
- prefill and decode separately, including batch and sequence lengths;
- physical footprint, peak MUSA memory, and explicit selected/fallback route for every row;
- both `RWKV7_MUSA_WKV=0` and enabled/default rows in the same environment.

Do not claim bf16, Triton/FLA, quantization, graph capture, multi-device, or a
different MUSA architecture until that exact capability is documented and
validated.
