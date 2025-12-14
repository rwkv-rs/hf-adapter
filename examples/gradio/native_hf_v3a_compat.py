"""Small Gradio-3 compatibility surface backed by the repository Native HF model."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM


MODEL_PATH = ""
WKV_MODE = "fp32io16"
EMB_DEVICE = "cuda"
RKV_MODE = "off"
CMIX_SPARSE = "no-fc"
LOWRANK_WEIGHT = "transpose"
ORIG_LINEAR_GROUPS = set()
PRECOMPUTE_EMB_LN0 = False
SYNC_INIT = False
C = 0
USES_INTERNAL_CUDA_GRAPH = True
DECODE_USES_TOKEN_IDS = True


def log(message) -> None:
    print(message, flush=True)


def load_extensions(_mode: str) -> None:
    return None


def select_path(_batch: int, _tokens: int):
    return "native_hf"


@dataclass
class NativeHFState:
    batch_size: int
    cache: object | None = None


def copy_state_to_batch(dst: NativeHFState, src: NativeHFState) -> None:
    if src.cache is None:
        raise RuntimeError("Native HF prompt state is empty")
    cache = src.cache.clone()
    if int(dst.batch_size) != int(src.batch_size):
        if int(src.batch_size) != 1:
            raise ValueError("Native HF batch expansion requires a batch-one prompt cache")
        cache.batch_repeat_interleave(int(dst.batch_size))
    dst.cache = cache


def _dtype_from_env() -> torch.dtype:
    value = os.environ.get("APP3_DTYPE", "bfloat16").strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported APP3_DTYPE={value!r}; use bf16 or fp16")


DTYPE = _dtype_from_env()


class RWKV7:
    def __init__(self) -> None:
        global C
        model_path = Path(os.environ.get("APP3_HF_MODEL_PATH", MODEL_PATH)).expanduser()
        if not model_path.is_dir():
            raise FileNotFoundError(
                "APP3_BACKEND=native_hf requires APP3_HF_MODEL_PATH to be a converted HF model directory"
            )
        os.environ.setdefault("RWKV7_NATIVE_MODEL_BACKEND", "native_graph")
        os.environ.setdefault("RWKV7_FAST_PREFILL", "1")
        self.device = torch.device("cuda")
        self.dtype = DTYPE
        self.model = NativeRWKV7ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(device=self.device, dtype=self.dtype).eval()
        self.emb_cpu = False
        self.n_embd = int(self.model.config.hidden_size)
        self.args = SimpleNamespace(
            vocab_size=int(self.model.config.vocab_size),
            n_embd=self.n_embd,
        )
        C = self.n_embd

    def zero_state(self, batch_size: int) -> NativeHFState:
        return NativeHFState(batch_size=int(batch_size))

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(token_ids.to(self.device, dtype=torch.long))

    def _forward(self, state: NativeHFState, **inputs) -> torch.Tensor:
        with torch.inference_mode():
            token_ids = inputs.get("input_ids")
            if (
                state.cache is not None
                and token_ids is not None
                and token_ids.dim() == 2
                and int(token_ids.shape[1]) == 1
            ):
                logits, state.cache = self.model.rwkv7_forward_token(
                    token_ids,
                    past_key_values=state.cache,
                    return_dict=False,
                    copy_logits=False,
                )
                output = logits[:, -1]
            else:
                result = self.model(
                    **inputs,
                    past_key_values=state.cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                state.cache = result.past_key_values

                output = result.logits[:, -1]

        # The Gradio sampler applies occurrence penalties in place. Cloning
        # after leaving inference mode returns a normal mutable tensor on new
        # PyTorch releases while keeping model execution inference-only.
        return output.clone()

    def forward(self, tokens, state: NativeHFState) -> torch.Tensor:
        token_ids = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        return self._forward(state, input_ids=token_ids)

    def forward_from_x(self, x: torch.Tensor, state: NativeHFState, _path=None) -> torch.Tensor:
        if int(x.shape[0]) != int(state.batch_size):
            raise ValueError(
                f"Native HF decode batch mismatch: embeddings={int(x.shape[0])}, state={state.batch_size}"
            )
        return self._forward(state, inputs_embeds=x)

    def forward_tokens(self, token_ids: torch.Tensor, state: NativeHFState, _path=None) -> torch.Tensor:
        if int(token_ids.shape[0]) != int(state.batch_size):
            raise ValueError(
                f"Native HF decode batch mismatch: tokens={int(token_ids.shape[0])}, state={state.batch_size}"
            )
        return self._forward(state, input_ids=token_ids.to(self.device, dtype=torch.long))
