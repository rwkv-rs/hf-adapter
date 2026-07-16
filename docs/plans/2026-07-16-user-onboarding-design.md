# Ordinary-user and AI-assisted onboarding design

Historical design record. Current user instructions live in
`docs/USER_GUIDE.md` and `docs/USER_GUIDE_ZH.md`.

## Goal

Make the first successful RWKV-7 HF generation possible without requiring a
user to understand benchmark scripts, `device_map`, internal cache APIs, or
backend environment variables.

## Chosen approach

Use five layers:

1. Put a five-minute path at the top of the root README.
2. Keep installation, checkpoint conversion, platform choices, Python API, and
   troubleshooting in a dedicated user guide with an equivalent Chinese entry.
3. Provide one executable `examples/generate.py` command whose `auto` policy
   selects CUDA/MPS/CPU and falls back to the native backend when FLA is absent.
4. Provide `examples/check_environment.py` as a no-download doctor with explicit
   PASS/FAIL output before installation is called complete.
5. Provide an AI runbook with a copy-ready prompt, approval boundaries, a small
   state machine, and command-based acceptance criteria.

This is preferred over prose alone because users and agents can validate the
documented path. It is preferred over an unattended checkpoint installer
because it avoids initiating large downloads without explicit approval. Windows
and Bash commands are separated so shell syntax is never left for a beginner to
translate.

## Safety and compatibility

- Do not require `accelerate` for single-device inference.
- Keep `trust_remote_code=True` explicit and explain its trust boundary.
- Default to deterministic generation and a small token count.
- Do not claim W8/W4 is universally faster; link to exact-card evidence.
- Keep benchmark and contributor documentation intact below the user entry.
- Do not ask users to share tokens, SSH credentials, or other secrets with an
  AI assistant for public checkpoint setup.
- Do not allow an AI assistant to substitute prose for a failed command; the
  failed state must be rerun successfully.

## Verification

- Unit-test device, dtype, and backend selection without a model download.
- Run `examples/generate.py --help` in the base environment.
- Run `examples/check_environment.py` and unit-test model-directory diagnosis.
- Run clean-install packaging and documentation-freshness tests.
- Validate Markdown relative links before publishing.
