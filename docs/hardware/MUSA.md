# Moore Threads MUSA

This document records the imported MUSA implementation boundary for the RWKV-7
HF adapter. It does not promote a live-card HF support result: this branch was
prepared without a local MUSA development environment.

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

RWKV-MUSA reports these results for MTT S70 (`mp_21`, 7 GB), MUSA SDK 4.2.0,
torch_musa 2.5.0. They are source evidence, not newly reproduced results:

- fp16 IO with fp32 recurrence compute/state is the validated WKV route;
- the imported kernel uses block synchronization and no warp shuffle;
- the validated kernel head size is 64;
- bf16 was not validated and is not included in this port;
- Triton/FLA and torch quantization are not enabled by this port;
- CUDA graph, CUDA kernel, ROCm, and other-device policies are not applied to MUSA.

The RWKV-MUSA repository retains forward/state/gradient checks and S70
performance logs. This HF branch does not copy those numbers into
`BENCHMARK.md`, because the current adapter revision has not yet been run on the
card.

## Real-device acceptance gate

Run on the MUSA host before changing the hardware matrix from **Open**:

```bash
PYTHON_BIN=python DEVICE=musa DTYPE=fp32 \
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
RESULTS=bench/results_musa.jsonl \
bash scripts/run_hardware_smoke.sh

python tests/test_hf_api_contract.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device musa --dtype fp32

python tests/test_peft_lora.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device musa --attn-mode native
```

Also record:

- exact MTT device, driver, MUSA SDK, torch_musa, PyTorch and Transformers versions;
- pure-PyTorch versus MUSA-WKV logits, greedy tokens, state maximum error and chunk handoff;
- prefill and decode separately, including batch and sequence lengths;
- physical footprint, peak MUSA memory, selected/fallback route;
- a kernel-disabled row using `RWKV7_MUSA_WKV=0`.

Do not claim bf16, Triton/FLA, quantization, graph capture, multi-device, or a
different MUSA architecture until that exact capability is documented and
validated.
