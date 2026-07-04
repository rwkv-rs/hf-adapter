# coding=utf-8
"""Correctness-first MLX RWKV-7 recurrent reference backend.

This module is the next Apple Silicon layer after :mod:`rwkv7_hf.mlx_bridge`:
it can run the native RWKV-7 recurrent equations directly on MLX arrays loaded
from a converted HuggingFace checkpoint.  The implementation intentionally
mirrors ``rwkv7_hf.native`` and stays optional/import-safe on non-Apple hosts.

It is **not** the final production Metal/WKV kernel.  The purpose is to pin down
the full MLX weight layout, recurrent state-cache semantics, chunked prefill,
and greedy decode behavior before replacing the inner WKV update with a fused
Metal/MLX kernel.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .mlx_bridge import load_selected_hf_tensors_as_mlx, require_mlx, summarize_mlx_arrays


EXP_HALF = 0.606531  # = exp(-0.5), RWKV-7 decay base used by the native torch path.


def _mx():
    return require_mlx()


def _as_list(values: Iterable[int] | Any) -> list[int]:
    if isinstance(values, list):
        return [int(v) for v in values]
    if isinstance(values, tuple):
        return [int(v) for v in values]
    try:
        return [int(v) for v in values.tolist()]
    except Exception:
        return [int(v) for v in values]


def _layer_indices(arrays: dict[str, Any]) -> list[int]:
    found: set[int] = set()
    pattern = re.compile(r"^model\.layers\.(\d+)\.")
    for key in arrays:
        match = pattern.match(key)
        if match:
            found.add(int(match.group(1)))
    if not found:
        raise ValueError("no model.layers.* tensors found in MLX weight bundle")
    return sorted(found)


@dataclass
class MLXRWKV7State:
    """MLX recurrent state cache for RWKV-7 decode.

    Layout mirrors the native PyTorch cache:

    - ``recurrent_state[layer]``: ``[B, H, N, N]`` fp32 WKV state
    - ``attn_x_prev[layer]``: ``[B, hidden]`` previous attention input
    - ``ffn_x_prev[layer]``: ``[B, hidden]`` previous FFN input
    - ``v_first``: ``[B, hidden]`` first-layer value stream

    ``select_batch`` / ``reorder_cache`` give the MLX path the same dynamic
    batching seam used by HF serving caches.
    """

    recurrent_state: list[Any]
    attn_x_prev: list[Any]
    ffn_x_prev: list[Any]
    v_first: Any
    seen_tokens: int = 0

    @property
    def batch_size(self) -> int:
        return int(self.v_first.shape[0])

    @property
    def num_layers(self) -> int:
        return len(self.recurrent_state)

    def clone(self) -> "MLXRWKV7State":
        mx = _mx()
        cloned = MLXRWKV7State(
            [mx.array(x) for x in self.recurrent_state],
            [mx.array(x) for x in self.attn_x_prev],
            [mx.array(x) for x in self.ffn_x_prev],
            mx.array(self.v_first),
            seen_tokens=int(self.seen_tokens),
        )
        mx.eval(cloned.v_first, *cloned.recurrent_state, *cloned.attn_x_prev, *cloned.ffn_x_prev)
        return cloned

    def select_batch(self, indices: Iterable[int] | Any) -> "MLXRWKV7State":
        mx = _mx()
        idx = mx.array(_as_list(indices), dtype=mx.int32)
        selected = MLXRWKV7State(
            [mx.take(x, idx, axis=0) for x in self.recurrent_state],
            [mx.take(x, idx, axis=0) for x in self.attn_x_prev],
            [mx.take(x, idx, axis=0) for x in self.ffn_x_prev],
            mx.take(self.v_first, idx, axis=0),
            seen_tokens=int(self.seen_tokens),
        )
        mx.eval(selected.v_first, *selected.recurrent_state, *selected.attn_x_prev, *selected.ffn_x_prev)
        return selected

    def reorder_cache(self, indices: Iterable[int] | Any) -> "MLXRWKV7State":
        return self.select_batch(indices)

    def compact(self, indices: Iterable[int] | Any) -> "MLXRWKV7State":
        return self.select_batch(indices)

    def detach(self) -> "MLXRWKV7State":
        # MLX arrays are eager/lazy value arrays, not torch autograd tensors; a
        # clone gives callers an explicit cache boundary.
        return self.clone()


@dataclass
class MLXGenerateOutput:
    """Tokenizer-integrated MLX generation result."""

    prompt: str
    prompt_ids: list[int]
    generated_ids: list[int]
    text: str
    prefill_s: float
    decode_s: float
    prompt_tokens: int
    generated_tokens: int

    @property
    def prefill_tok_s(self) -> float | None:
        return self.prompt_tokens / self.prefill_s if self.prefill_s > 0 else None

    @property
    def decode_tok_s(self) -> float | None:
        return self.generated_tokens / self.decode_s if self.decode_s > 0 else None

    def telemetry(self) -> dict[str, Any]:
        return {
            "prompt_tokens": int(self.prompt_tokens),
            "generated_tokens": int(self.generated_tokens),
            "prefill_s": round(float(self.prefill_s), 6),
            "decode_s": round(float(self.decode_s), 6),
            "prefill_tok_s": round(float(self.prefill_tok_s), 6) if self.prefill_tok_s is not None else None,
            "decode_tok_s": round(float(self.decode_tok_s), 6) if self.decode_tok_s is not None else None,
            "generated_preview": [int(x) for x in self.generated_ids[:16]],
        }


@dataclass
class MLXSessionStepOutput:
    """One incremental decode step from an :class:`MLXGenerationSession`."""

    step_index: int
    generated_ids: list[int]
    text: str
    decode_s: float
    total_generated_tokens: int
    seen_tokens: int

    @property
    def generated_tokens(self) -> int:
        return len(self.generated_ids)

    @property
    def decode_tok_s(self) -> float | None:
        return self.generated_tokens / self.decode_s if self.decode_s > 0 else None

    def telemetry(self) -> dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "generated_tokens": int(self.generated_tokens),
            "total_generated_tokens": int(self.total_generated_tokens),
            "seen_tokens": int(self.seen_tokens),
            "decode_s": round(float(self.decode_s), 6),
            "decode_tok_s": round(float(self.decode_tok_s), 6) if self.decode_tok_s is not None else None,
            "generated_preview": [int(x) for x in self.generated_ids[:16]],
        }


class MLXGenerationSession:
    """Stateful tokenizer-backed MLX generation helper.

    The plain ``generate_text`` helper is useful for one-shot demos.  Serving
    style callers need a stricter seam: prefill a prompt once, hold the RWKV
    recurrent state cache, then decode in multiple chunks without recomputing
    the prompt.  This class exposes that shape for Apple/MLX smoke tests and
    future Metal-backed serving integration.

    The session is intentionally single-prompt/tokenizer-backed.  Dynamic batch
    select/reorder remains covered at the lower ``MLXRWKV7State`` layer where
    batched cache tensors are explicit.
    """

    def __init__(
        self,
        *,
        model: "MLXRWKV7Model",
        tokenizer: Any,
        prompt: str,
        prompt_ids: list[int],
        logits: Any,
        state: MLXRWKV7State,
        prefill_s: float,
        skip_special_tokens: bool = False,
    ):
        if state.batch_size != 1:
            raise ValueError("MLXGenerationSession currently expects one prompt / batch row")
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.prompt_ids = [int(x) for x in prompt_ids]
        self.logits = logits
        self.state = state
        self.prefill_s = float(prefill_s)
        self.decode_s = 0.0
        self.generated_ids: list[int] = []
        self.step_count = 0
        self.skip_special_tokens = bool(skip_special_tokens)

    @classmethod
    def from_prompt(
        cls,
        model: "MLXRWKV7Model",
        tokenizer: Any,
        prompt: str,
        *,
        skip_special_tokens: bool = False,
    ) -> "MLXGenerationSession":
        """Encode and prefill a prompt, returning a reusable decode session."""

        encoded = tokenizer(prompt, add_special_tokens=False)
        prompt_ids = [int(tok) for tok in encoded.input_ids]
        if not prompt_ids:
            raise ValueError("prompt produced no token ids")
        t0 = time.perf_counter()
        logits, state = model.prefill([prompt_ids])
        prefill_s = time.perf_counter() - t0
        return cls(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_ids=prompt_ids,
            logits=logits,
            state=state,
            prefill_s=prefill_s,
            skip_special_tokens=skip_special_tokens,
        )

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_ids)

    @property
    def generated_tokens(self) -> int:
        return len(self.generated_ids)

    @property
    def text(self) -> str:
        return self.tokenizer.decode(self.generated_ids, skip_special_tokens=self.skip_special_tokens)

    @property
    def prefill_tok_s(self) -> float | None:
        return self.prompt_tokens / self.prefill_s if self.prefill_s > 0 else None

    @property
    def decode_tok_s(self) -> float | None:
        return self.generated_tokens / self.decode_s if self.decode_s > 0 else None

    def decode(self, max_new_tokens: int) -> MLXSessionStepOutput:
        """Decode ``max_new_tokens`` more tokens from the cached state."""

        mx = _mx()
        n = int(max_new_tokens)
        if n < 0:
            raise ValueError("max_new_tokens must be non-negative")
        t0 = time.perf_counter()
        generated = []
        next_token = mx.argmax(self.logits[:, -1, :], axis=-1).astype(mx.int32)
        for _ in range(n):
            generated.append(next_token)
            self.logits, self.state = self.model.decode_step(next_token, self.state)
            next_token = mx.argmax(self.logits[:, -1, :], axis=-1).astype(mx.int32)
        if generated:
            out = mx.stack(generated, axis=1)
            mx.eval(out, self.logits)
            step_ids = _as_list(out.reshape(-1))
        else:
            mx.eval(self.logits)
            step_ids = []
        elapsed = time.perf_counter() - t0
        self.generated_ids.extend(step_ids)
        self.decode_s += elapsed
        self.step_count += 1
        return MLXSessionStepOutput(
            step_index=self.step_count,
            generated_ids=step_ids,
            text=self.tokenizer.decode(step_ids, skip_special_tokens=self.skip_special_tokens),
            decode_s=elapsed,
            total_generated_tokens=self.generated_tokens,
            seen_tokens=int(self.state.seen_tokens),
        )

    def output(self) -> MLXGenerateOutput:
        """Return a cumulative one-shot-style generation output."""

        return MLXGenerateOutput(
            prompt=self.prompt,
            prompt_ids=list(self.prompt_ids),
            generated_ids=list(self.generated_ids),
            text=self.text,
            prefill_s=self.prefill_s,
            decode_s=self.decode_s,
            prompt_tokens=self.prompt_tokens,
            generated_tokens=self.generated_tokens,
        )

    def telemetry(self) -> dict[str, Any]:
        out = self.output().telemetry()
        out.update(
            {
                "session_steps": int(self.step_count),
                "seen_tokens": int(self.state.seen_tokens),
            }
        )
        return out


class MLXGenerationSessionBatch:
    """Small interleaved session manager for MLX serving smoke tests.

    This is not a fused batched MLX kernel.  It is a production-shaped API
    scaffold: multiple independent prompts are prefetched once, then advanced
    round-by-round while preserving each prompt's recurrent state cache.  The
    helper lets Apple validation exercise concurrent session bookkeeping before
    the inner decode loop is replaced by a fused MLX/Metal backend.
    """

    def __init__(self, sessions: list[MLXGenerationSession]):
        if not sessions:
            raise ValueError("MLXGenerationSessionBatch requires at least one session")
        model = sessions[0].model
        tokenizer = sessions[0].tokenizer
        for idx, session in enumerate(sessions):
            if session.model is not model:
                raise ValueError(f"session {idx} uses a different MLX model instance")
            if session.tokenizer is not tokenizer:
                raise ValueError(f"session {idx} uses a different tokenizer instance")
        self.model = model
        self.tokenizer = tokenizer
        self.sessions = list(sessions)
        self.round_count = 0

    @classmethod
    def from_prompts(
        cls,
        model: "MLXRWKV7Model",
        tokenizer: Any,
        prompts: list[str],
        *,
        skip_special_tokens: bool = False,
    ) -> "MLXGenerationSessionBatch":
        if isinstance(prompts, str):
            raise TypeError("prompts must be a list of strings, not a single string")
        if not prompts:
            raise ValueError("prompts must contain at least one prompt")
        sessions = [
            MLXGenerationSession.from_prompt(
                model,
                tokenizer,
                prompt,
                skip_special_tokens=skip_special_tokens,
            )
            for prompt in prompts
        ]
        return cls(sessions)

    @property
    def batch_size(self) -> int:
        return len(self.sessions)

    def decode_round(self, tokens_per_session: int | list[int]) -> list[MLXSessionStepOutput]:
        """Advance all sessions once and return per-session step outputs."""

        if isinstance(tokens_per_session, int):
            steps = [int(tokens_per_session)] * self.batch_size
        else:
            steps = [int(x) for x in tokens_per_session]
            if len(steps) != self.batch_size:
                raise ValueError(f"expected {self.batch_size} token counts, got {len(steps)}")
        if any(step < 0 for step in steps):
            raise ValueError("all token counts must be non-negative")
        outputs = [session.decode(step) for session, step in zip(self.sessions, steps)]
        self.round_count += 1
        return outputs

    def outputs(self) -> list[MLXGenerateOutput]:
        return [session.output() for session in self.sessions]

    def telemetry(self) -> dict[str, Any]:
        prompt_tokens = [session.prompt_tokens for session in self.sessions]
        generated_tokens = [session.generated_tokens for session in self.sessions]
        seen_tokens = [int(session.state.seen_tokens) for session in self.sessions]
        decode_s = [round(float(session.decode_s), 6) for session in self.sessions]
        return {
            "batch_size": int(self.batch_size),
            "session_rounds": int(self.round_count),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "seen_tokens": seen_tokens,
            "decode_s": decode_s,
            "decode_tok_s": [
                round(float(session.decode_tok_s), 6) if session.decode_tok_s is not None else None
                for session in self.sessions
            ],
            "generated_previews": [[int(x) for x in session.generated_ids[:16]] for session in self.sessions],
        }


class MLXRWKV7Model:
    """Minimal MLX-native RWKV-7 recurrent model loaded from HF safetensors."""

    def __init__(self, config: dict[str, Any], arrays: dict[str, Any]):
        self.config = dict(config)
        self.arrays = dict(arrays)
        self.layer_ids = _layer_indices(self.arrays)
        self.num_hidden_layers = int(self.config.get("num_hidden_layers", len(self.layer_ids)))
        self.hidden_size = int(self.config["hidden_size"])
        self.num_heads = int(self.config.get("num_heads", self.config.get("n_head", 0)))
        self.head_dim = int(self.config.get("head_dim", self.hidden_size // self.num_heads))
        self.vocab_size = int(self.config["vocab_size"])
        self.intermediate_size = int(self.config.get("intermediate_size", self.hidden_size * 4))
        self.norm_eps = float(self.config.get("norm_eps", 1e-5))
        if self.num_heads * self.head_dim != self.hidden_size:
            raise ValueError(
                f"invalid RWKV-7 shape: num_heads({self.num_heads}) * head_dim({self.head_dim}) "
                f"!= hidden_size({self.hidden_size})"
            )
        if len(self.layer_ids) != self.num_hidden_layers:
            raise ValueError(f"config has {self.num_hidden_layers} layers but tensors contain {len(self.layer_ids)}")

    @classmethod
    def from_hf(cls, model_dir: str | Path, *, dtype: str | None = "fp16") -> "MLXRWKV7Model":
        root = Path(model_dir)
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        arrays = load_selected_hf_tensors_as_mlx(root, tensor_regex=r".*", dtype=dtype)
        return cls(config, arrays)

    @classmethod
    def from_arrays(cls, config: dict[str, Any], arrays: dict[str, Any]) -> "MLXRWKV7Model":
        return cls(config, arrays)

    def telemetry(self) -> dict[str, Any]:
        return {
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            **summarize_mlx_arrays(self.arrays),
        }

    def _get(self, key: str):
        try:
            return self.arrays[key]
        except KeyError as exc:
            raise KeyError(f"missing MLX RWKV-7 tensor {key!r}") from exc

    def _linear(self, x, weight_key: str, bias_key: str | None = None):
        y = x @ self._get(weight_key).T
        if bias_key is not None and bias_key in self.arrays:
            y = y + self._get(bias_key)
        return y

    def _layer_norm(self, x, prefix: str):
        mx = _mx()
        xf = x.astype(mx.float32)
        mean = mx.mean(xf, axis=-1, keepdims=True)
        var = mx.mean((xf - mean) * (xf - mean), axis=-1, keepdims=True)
        y = (xf - mean) * mx.rsqrt(var + self.norm_eps)
        y = y.astype(x.dtype)
        return y * self._get(f"{prefix}.weight") + self._get(f"{prefix}.bias")

    def _group_norm_heads(self, x, layer: int):
        mx = _mx()
        B = int(x.shape[0])
        xf = x.astype(mx.float32).reshape(B, self.num_heads, self.head_dim)
        mean = mx.mean(xf, axis=-1, keepdims=True)
        var = mx.mean((xf - mean) * (xf - mean), axis=-1, keepdims=True)
        y = (xf - mean) * mx.rsqrt(var + self.head_dim * 1e-5)
        y = y.reshape(B, self.hidden_size).astype(x.dtype)
        prefix = f"model.layers.{layer}.attn.g_norm"
        return y * self._get(f"{prefix}.weight") + self._get(f"{prefix}.bias")

    def _normalize_last_dim(self, x, eps: float = 1e-12):
        mx = _mx()
        xf = x.astype(mx.float32)
        denom = mx.sqrt(mx.maximum(mx.sum(xf * xf, axis=-1, keepdims=True), eps))
        return (xf / denom).astype(x.dtype)

    def init_state(self, batch_size: int, *, dtype: Any | None = None) -> MLXRWKV7State:
        mx = _mx()
        if dtype is None:
            dtype = self._get("model.embeddings.weight").dtype
        B = int(batch_size)
        state = [
            mx.zeros((B, self.num_heads, self.head_dim, self.head_dim), dtype=mx.float32)
            for _ in range(self.num_hidden_layers)
        ]
        xpa = [mx.zeros((B, self.hidden_size), dtype=dtype) for _ in range(self.num_hidden_layers)]
        xpf = [mx.zeros((B, self.hidden_size), dtype=dtype) for _ in range(self.num_hidden_layers)]
        v_first = mx.zeros((B, self.hidden_size), dtype=dtype)
        mx.eval(v_first, *state, *xpa, *xpf)
        return MLXRWKV7State(state, xpa, xpf, v_first, seen_tokens=0)

    def _attn_step(self, layer: int, x, x_prev, v_first, state):
        mx = _mx()
        B = int(x.shape[0])
        hidden = self.hidden_size
        H = self.num_heads
        N = self.head_dim
        prefix = f"model.layers.{layer}.attn"
        xx = x_prev - x
        xr = x + xx * self._get(f"{prefix}.x_r").reshape(1, hidden)
        xw = x + xx * self._get(f"{prefix}.x_w").reshape(1, hidden)
        xk = x + xx * self._get(f"{prefix}.x_k").reshape(1, hidden)
        xv = x + xx * self._get(f"{prefix}.x_v").reshape(1, hidden)
        xa = x + xx * self._get(f"{prefix}.x_a").reshape(1, hidden)
        xg = x + xx * self._get(f"{prefix}.x_g").reshape(1, hidden)

        r = self._linear(xr, f"{prefix}.r_proj.weight")
        w = self._linear(
            mx.tanh(self._linear(xw, f"{prefix}.w_lora.lora.0.weight")),
            f"{prefix}.w_lora.lora.2.weight",
            f"{prefix}.w_lora.lora.2.bias",
        )
        k = self._linear(xk, f"{prefix}.k_proj.weight")
        v = self._linear(xv, f"{prefix}.v_proj.weight")
        a = mx.sigmoid(
            self._linear(
                self._linear(xa, f"{prefix}.a_lora.lora.0.weight"),
                f"{prefix}.a_lora.lora.2.weight",
                f"{prefix}.a_lora.lora.2.bias",
            )
        )
        g = self._linear(
            mx.sigmoid(self._linear(xg, f"{prefix}.g_lora.lora.0.weight")),
            f"{prefix}.g_lora.lora.2.weight",
        )

        kk = self._normalize_last_dim((k * self._get(f"{prefix}.k_k").reshape(1, hidden)).reshape(B, H, N)).reshape(
            B, hidden
        )
        k = k * (1 + (a - 1) * self._get(f"{prefix}.k_a").reshape(1, hidden))
        if layer == 0:
            v_first = v
        else:
            v_mix = mx.sigmoid(
                self._linear(
                    self._linear(xv, f"{prefix}.v_lora.lora.0.weight"),
                    f"{prefix}.v_lora.lora.2.weight",
                    f"{prefix}.v_lora.lora.2.bias",
                )
            )
            v = v + (v_first - v) * v_mix
        w = mx.exp(-EXP_HALF * mx.sigmoid(w.astype(mx.float32)))

        vk = v.reshape(B, H, N, 1) @ k.reshape(B, H, 1, N)
        ab = (-kk).reshape(B, H, N, 1) @ (kk * a).reshape(B, H, 1, N)
        state = state * w.reshape(B, H, 1, N) + state @ ab.astype(mx.float32) + vk.astype(mx.float32)
        out = state.astype(x.dtype) @ r.reshape(B, H, N, 1)
        out = out.reshape(B, hidden)
        out = self._group_norm_heads(out, layer)
        sk = (
            r.reshape(B, H, N)
            * k.reshape(B, H, N)
            * self._get(f"{prefix}.r_k").reshape(1, H, N)
        ).sum(axis=-1, keepdims=True)
        out = out + (sk * v.reshape(B, H, N)).reshape(B, hidden)
        out = self._linear(out * g, f"{prefix}.o_proj.weight")
        return out, x, state, v_first

    def _ffn_step(self, layer: int, x, x_prev):
        mx = _mx()
        prefix = f"model.layers.{layer}.ffn"
        xx = x_prev - x
        k = x + xx * self._get(f"{prefix}.x_k").reshape(1, self.hidden_size)
        k = mx.maximum(self._linear(k, f"{prefix}.key.weight"), 0)
        k = k * k
        return self._linear(k, f"{prefix}.value.weight"), x

    def _embedding(self, token_ids):
        mx = _mx()
        ids = token_ids.astype(mx.int32).reshape(-1)
        return self._get("model.embeddings.weight")[ids]

    def _step_token(self, token_ids, state: MLXRWKV7State):
        mx = _mx()
        x = self._embedding(token_ids)
        for layer in range(self.num_hidden_layers):
            residual = self._layer_norm(x, f"model.layers.{layer}.pre_norm") if layer == 0 else x
            h = self._layer_norm(residual, f"model.layers.{layer}.attn_norm")
            a, state.attn_x_prev[layer], state.recurrent_state[layer], state.v_first = self._attn_step(
                layer,
                h,
                state.attn_x_prev[layer],
                state.v_first,
                state.recurrent_state[layer],
            )
            x = residual + a
            residual = x
            h2 = self._layer_norm(x, f"model.layers.{layer}.ffn_norm")
            f, state.ffn_x_prev[layer] = self._ffn_step(layer, h2, state.ffn_x_prev[layer])
            x = residual + f
        state.seen_tokens += 1
        mx.eval(x, state.v_first, *state.recurrent_state, *state.attn_x_prev, *state.ffn_x_prev)
        return x, state

    def _logits_from_hidden(self, x):
        x = self._layer_norm(x, "model.norm")
        return self._linear(x, "lm_head.weight")

    def forward(self, input_ids: Iterable[Iterable[int]] | Any, state: MLXRWKV7State | None = None, *, collect_all: bool = True):
        """Run recurrent forward over ``input_ids`` shaped ``[B, T]``.

        Returns ``(logits, state)``.  With ``collect_all=True``, logits are
        shaped ``[B, T, vocab]``.  Otherwise only the last-token logits are
        returned as ``[B, 1, vocab]``.
        """

        mx = _mx()
        ids = mx.array(input_ids, dtype=mx.int32)
        if ids.ndim == 1:
            ids = ids.reshape(1, -1)
        if ids.ndim != 2:
            raise ValueError("MLXRWKV7Model.forward expects input ids shaped [batch, seq]")
        B, T = int(ids.shape[0]), int(ids.shape[1])
        if T <= 0 or B <= 0:
            raise ValueError("MLXRWKV7Model.forward requires a non-empty batch and sequence")
        if state is None:
            state = self.init_state(B)
        elif state.batch_size != B:
            raise ValueError(f"state batch size {state.batch_size} does not match input batch size {B}")
        logits = []
        last = None
        for t in range(T):
            last, state = self._step_token(ids[:, t], state)
            if collect_all:
                logits.append(self._logits_from_hidden(last))
        if collect_all:
            out = mx.stack(logits, axis=1)
        else:
            out = self._logits_from_hidden(last).reshape(B, 1, self.vocab_size)
        mx.eval(out)
        return out, state

    def prefill(self, input_ids: Iterable[Iterable[int]] | Any, state: MLXRWKV7State | None = None):
        return self.forward(input_ids, state=state, collect_all=False)

    def decode_step(self, token_ids: Iterable[int] | Any, state: MLXRWKV7State):
        mx = _mx()
        ids = mx.array(token_ids, dtype=mx.int32).reshape(-1, 1)
        return self.forward(ids, state=state, collect_all=False)

    def decode_greedy(self, logits: Any, state: MLXRWKV7State, *, max_new_tokens: int):
        """Continue decoding from an existing prefill ``logits`` + ``state``.

        This is the serving-shaped path: callers prefill a prompt once, keep the
        recurrent state cache, and then decode one token at a time without
        recomputing the prompt.
        """

        mx = _mx()
        generated = []
        next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
        for _ in range(int(max_new_tokens)):
            generated.append(next_token)
            logits, state = self.decode_step(next_token, state)
            next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
        if not generated:
            return mx.zeros((int(logits.shape[0]), 0), dtype=mx.int32), state
        out = mx.stack(generated, axis=1)
        mx.eval(out)
        return out, state

    def chunked_prefill(self, input_ids: Iterable[Iterable[int]] | Any, *, chunk_size: int):
        mx = _mx()
        ids = mx.array(input_ids, dtype=mx.int32)
        if ids.ndim == 1:
            ids = ids.reshape(1, -1)
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        state = self.init_state(int(ids.shape[0]))
        logits = None
        for start in range(0, int(ids.shape[1]), int(chunk_size)):
            logits, state = self.forward(ids[:, start : start + int(chunk_size)], state=state, collect_all=False)
        if logits is None:
            raise ValueError("chunked_prefill requires non-empty input")
        return logits, state

    def generate_greedy(self, input_ids: Iterable[Iterable[int]] | Any, *, max_new_tokens: int):
        logits, state = self.prefill(input_ids)
        return self.decode_greedy(logits, state, max_new_tokens=max_new_tokens)

    def generate_text(
        self,
        tokenizer: Any,
        prompt: str,
        *,
        max_new_tokens: int,
        skip_special_tokens: bool = False,
    ) -> MLXGenerateOutput:
        """Encode ``prompt`` with an HF tokenizer and greedily decode on MLX.

        This is a lightweight reusable API for Apple-local demos and smoke
        harnesses.  It intentionally returns generated text only (not prompt +
        completion) so callers can decide how to display or postprocess.
        """

        session = MLXGenerationSession.from_prompt(
            self,
            tokenizer,
            prompt,
            skip_special_tokens=skip_special_tokens,
        )
        session.decode(int(max_new_tokens))
        return session.output()


def load_mlx_rwkv7_model(model_dir: str | Path, *, dtype: str | None = "fp16") -> MLXRWKV7Model:
    return MLXRWKV7Model.from_hf(model_dir, dtype=dtype)


def load_mlx_generation_session(
    model_dir: str | Path,
    prompt: str,
    *,
    dtype: str | None = "fp16",
    skip_special_tokens: bool = False,
) -> MLXGenerationSession:
    """Load a converted HF directory and prefill a tokenizer-backed MLX session."""

    from transformers import AutoTokenizer

    model = load_mlx_rwkv7_model(model_dir, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return MLXGenerationSession.from_prompt(
        model,
        tokenizer,
        prompt,
        skip_special_tokens=skip_special_tokens,
    )


def generate_text_from_hf(
    model_dir: str | Path,
    prompt: str,
    *,
    max_new_tokens: int,
    dtype: str | None = "fp16",
    skip_special_tokens: bool = False,
) -> MLXGenerateOutput:
    """Load a converted HF directory and run tokenizer-integrated MLX generate."""

    from transformers import AutoTokenizer

    model = load_mlx_rwkv7_model(model_dir, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return model.generate_text(
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        skip_special_tokens=skip_special_tokens,
    )
