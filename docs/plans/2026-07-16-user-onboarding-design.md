# Ordinary-user onboarding design

Historical design record. Current user instructions live in
`docs/USER_GUIDE.md` and `docs/USER_GUIDE_ZH.md`.

## Goal

Make the first successful RWKV-7 HF generation possible without requiring a
user to understand benchmark scripts, `device_map`, internal cache APIs, or
backend environment variables.

## Chosen approach

Use three layers:

1. Put a five-minute path at the top of the root README.
2. Keep installation, checkpoint conversion, platform choices, Python API, and
   troubleshooting in a dedicated user guide with an equivalent Chinese entry.
3. Provide one executable `examples/generate.py` command whose `auto` policy
   selects CUDA/MPS/CPU and falls back to the native backend when FLA is absent.

This is preferred over documentation alone because users can validate the
documented path, and preferred over a one-click checkpoint installer because
it avoids hard-coding a model catalog or initiating multi-gigabyte downloads
without an explicit user command.

## Safety and compatibility

- Do not require `accelerate` for single-device inference.
- Keep `trust_remote_code=True` explicit and explain its trust boundary.
- Default to deterministic generation and a small token count.
- Do not claim W8/W4 is universally faster; link to exact-card evidence.
- Keep benchmark and contributor documentation intact below the user entry.

## Verification

- Unit-test device, dtype, and backend selection without a model download.
- Run `examples/generate.py --help` in the base environment.
- Run clean-install packaging and documentation-freshness tests.
- Validate Markdown relative links before publishing.
