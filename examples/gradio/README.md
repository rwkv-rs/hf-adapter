# RWKV-Gradio-3 Native HF bridge

`native_hf_v3a_compat.py` exposes the small model/state surface used by the
official `BlinkDL/RWKV-Gradio-3` Space while delegating inference to
`NativeRWKV7ForCausalLM`. The adjacent patch adds backend selection and token-ID
decode support to Space commit `cc57df4`.

This integration is opt-in. Follow
[`docs/GRADIO_NATIVE_HF.md`](../../docs/GRADIO_NATIVE_HF.md) for pinned commands,
observable pass criteria, recovery, current performance, and the single AI
entry point.
