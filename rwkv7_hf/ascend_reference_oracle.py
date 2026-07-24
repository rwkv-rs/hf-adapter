"""Independent CPU oracle for the Huawei Ascend RWKV-7 acceptance gate.

The implementation is a direct, dependency-light transcription of the pure
PyTorch layer formula and naive recurrence at the pinned FLA revision below.
It deliberately does not import or call ``native.py``, ``native_model.py`` or a
Hugging Face model ``forward``.  Official checkpoint tensors are read directly
from the safetensors index so the candidate backend cannot become its own
reference.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F

FLA_REPOSITORY = "https://github.com/fla-org/flash-linear-attention.git"
FLA_COMMIT = "d1ce07369d581813553f30a750af3b6b5f9af6a9"
FLA_SOURCE_SHA256 = {
    "fla/layers/rwkv7.py": "de4109b25df8162810f8b45abfc48cc8149b93a02a06cfb51ea3b36717404ffa",
    "fla/models/rwkv7/modeling_rwkv7.py": "9475cb196b820fbd8ad9ab6f8cf31e53976feda17c2c48fb7cce1cbd78cdc537",
    "fla/ops/rwkv7/fused_recurrent.py": "0a2e76abc14e35aa2a479757d1a1f7549f2876c3980709bb4be0a1fcb4dfddff",
}
RWKV7_7P2_CHECKPOINT_SHA256 = {
    "config.json": "309b9eac6a4948c927d8ce62e533c777050ed5d2e45b607bf88da0726459e46c",
    "model.safetensors.index.json": "0b209368afda04594706c38e1151c811dd4ccee0398d789e34c4d8f3d0f75d2c",
    "model-00001-of-00003.safetensors": "68b42441667378b2709e0de134c1851216f9bd1eb482bf3a8bf1b2f13a3e41aa",
    "model-00002-of-00003.safetensors": "7fca283c35c77173af3ca6de598167cbf34acfc86f2808915f496cd035b0258f",
    "model-00003-of-00003.safetensors": "cd5c4b8dc1028fddf5e7cf6f52e8667e762daa06903463a56081b03fb5a6d7c4",
}
RWKV7_7P2_TOKENIZER_SHA256 = {
    "added_tokens.json": "a349cae6cdaa680cf6fc0d2929b16f2a9edb43eb7027b2e344b1ca8063854fb9",
    "hf_rwkv_tokenizer.py": "b63afcc86288f29301f27f373e19f9cf72c188e7d8164fa03ecfb20015505b6b",
    "rwkv_vocab_v20230424.txt": "e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89",
    "special_tokens_map.json": "e5cbcda832aba36b2e5ae58039e0ab55534e9716a6dbc1a3d743879e4a134154",
    "tokenizer_config.json": "e6feaaf6331c743ad14b8d146a7c05e16cf44b5c1fd6d6f07459545d7ecd6bdb",
}
REFERENCE_FORMAT_VERSION = 1
EXP_HALF = 0.6065306597126334
DEFAULT_ACCEPTANCE_THRESHOLDS = {
    "logits_min_cosine": 0.999,
    "logits_max_normalized_rmse": 0.02,
    "state_min_cosine": 0.999,
    "state_max_normalized_rmse": 0.02,
    "greedy_token_ids_exact": True,
}


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(root: str | Path, expected: Mapping[str, str]) -> dict[str, str]:
    root = Path(root)
    observed: dict[str, str] = {}
    errors = []
    for relative, wanted in expected.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        actual = sha256_file(path)
        observed[relative] = actual
        if actual != wanted:
            errors.append(f"sha256 mismatch {relative}: {actual} != {wanted}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return observed


def verify_fla_checkout(path: str | Path) -> dict[str, object]:
    root = Path(path)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot identify FLA checkout {root}") from exc
    if commit != FLA_COMMIT:
        raise RuntimeError(f"FLA commit {commit} != pinned {FLA_COMMIT}")
    return {
        "repository": FLA_REPOSITORY,
        "commit": commit,
        "files_sha256": verify_files(root, FLA_SOURCE_SHA256),
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_map_sha256(tensors: Mapping[str, torch.Tensor], names: Iterable[str] | None = None) -> str:
    selected = sorted(tensors if names is None else names)
    digest = hashlib.sha256()
    for name in selected:
        if name not in tensors:
            raise KeyError(name)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_sha256(tensors[name]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass
class OracleState:
    recurrent: list[torch.Tensor]
    attn_shift: list[torch.Tensor]
    ffn_shift: list[torch.Tensor]
    v_first: torch.Tensor
    valid_tokens: torch.Tensor
    processed_width: int = 0

    def clone(self) -> "OracleState":
        return OracleState(
            [value.clone() for value in self.recurrent],
            [value.clone() for value in self.attn_shift],
            [value.clone() for value in self.ffn_shift],
            self.v_first.clone(),
            self.valid_tokens.clone(),
            self.processed_width,
        )


@dataclass
class OracleOutput:
    logits: torch.Tensor
    state: OracleState


class SafetensorStore:
    """Read indexed checkpoint tensors without constructing an adapter model."""

    def __init__(self, model_dir: str | Path):
        self.root = Path(model_dir)
        self.config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        index_path = self.root / "model.safetensors.index.json"
        self.index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        self._contexts: list[object] = []
        self._handles: dict[str, object] = {}

    def __enter__(self) -> "SafetensorStore":
        from safetensors import safe_open

        for filename in sorted(set(self.index.values())):
            context = safe_open(self.root / filename, framework="pt", device="cpu")
            self._contexts.append(context)
            self._handles[filename] = context.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        for context in reversed(self._contexts):
            context.__exit__(exc_type, exc, tb)
        self._contexts.clear()
        self._handles.clear()

    def tensor(self, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        try:
            filename = self.index[name]
        except KeyError as exc:
            raise KeyError(f"checkpoint tensor is missing: {name}") from exc
        tensor = self._handles[filename].get_tensor(name)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor


class NaiveRWKV7Oracle:
    """Layer-by-layer CPU implementation of the pinned FLA RWKV-7 formula."""

    def __init__(self, store: SafetensorStore, *, dtype: torch.dtype = torch.bfloat16):
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("reference dtype must be bfloat16 or float32")
        self.store = store
        self.dtype = dtype
        self.config = dict(store.config)
        r_k = store.tensor("model.layers.0.attn.r_k")
        if r_k.ndim != 2:
            raise ValueError("r_k must have [heads, head_dim] shape")
        self.num_heads, self.head_dim = map(int, r_k.shape)
        self.hidden_size = int(self.config["hidden_size"])
        self.num_layers = int(self.config["num_hidden_layers"])
        if self.num_heads * self.head_dim != self.hidden_size:
            raise ValueError("this oracle currently requires attention width == hidden width")
        self.config_repair = {
            "declared_num_heads": int(self.config.get("num_heads", self.num_heads)),
            "inferred_num_heads": self.num_heads,
            "inference_source": "model.layers.0.attn.r_k.shape",
            "changed": int(self.config.get("num_heads", self.num_heads)) != self.num_heads,
        }

    def _t(self, name: str) -> torch.Tensor:
        return self.store.tensor(name, self.dtype)

    @staticmethod
    def _linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return F.linear(x, weight, bias)

    def _layer_norm(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        return F.layer_norm(
            x,
            (self.hidden_size,),
            self._t(prefix + ".weight"),
            self._t(prefix + ".bias"),
            float(self.config.get("norm_eps", 1e-5)),
        )

    def _lora(
        self,
        x: torch.Tensor,
        prefix: str,
        *,
        inner_activation: str | None,
        output_sigmoid: bool = False,
    ) -> torch.Tensor:
        value = self._linear(x, self._t(prefix + ".lora.0.weight"))
        if inner_activation == "tanh":
            value = torch.tanh(value)
        elif inner_activation == "sigmoid":
            value = torch.sigmoid(value)
        elif inner_activation is not None:
            raise ValueError(inner_activation)
        bias_name = prefix + ".lora.2.bias"
        bias = self._t(bias_name) if bias_name in self.store.index else None
        value = self._linear(value, self._t(prefix + ".lora.2.weight"), bias)
        return torch.sigmoid(value) if output_sigmoid else value

    def init_state(self, batch_size: int) -> OracleState:
        b = int(batch_size)
        return OracleState(
            recurrent=[
                torch.zeros(b, self.num_heads, self.head_dim, self.head_dim, dtype=torch.float32)
                for _ in range(self.num_layers)
            ],
            attn_shift=[torch.zeros(b, self.hidden_size, dtype=self.dtype) for _ in range(self.num_layers)],
            ffn_shift=[torch.zeros(b, self.hidden_size, dtype=self.dtype) for _ in range(self.num_layers)],
            v_first=torch.zeros(b, self.hidden_size, dtype=self.dtype),
            valid_tokens=torch.zeros(b, dtype=torch.int64),
        )

    @staticmethod
    def _masked_shift(
        sequence: torch.Tensor,
        previous: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the last valid recurrent input for every token position."""
        shifted = []
        last = previous
        for token_idx in range(sequence.shape[1]):
            shifted.append(last)
            last = torch.where(mask[:, token_idx, None], sequence[:, token_idx], last)
        return torch.stack(shifted, dim=1)

    def _attention(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        state: OracleState,
        v_first_sequence: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p = f"model.layers.{layer_idx}.attn"
        b, t, d = hidden.shape
        previous = self._masked_shift(hidden, state.attn_shift[layer_idx], mask)
        delta = previous - hidden
        mixed = {
            role: torch.addcmul(hidden, delta, self._t(f"{p}.x_{role}"))
            for role in ("r", "w", "k", "v", "a", "g")
        }
        r = self._linear(mixed["r"], self._t(f"{p}.r_proj.weight"))
        w_raw = self._lora(mixed["w"], f"{p}.w_lora", inner_activation="tanh")
        k = self._linear(mixed["k"], self._t(f"{p}.k_proj.weight"))
        v = self._linear(mixed["v"], self._t(f"{p}.v_proj.weight"))
        a = self._lora(
            mixed["a"], f"{p}.a_lora", inner_activation=None, output_sigmoid=True
        )
        g = self._lora(mixed["g"], f"{p}.g_lora", inner_activation="sigmoid")
        kk = F.normalize(
            (k * self._t(f"{p}.k_k")).reshape(b, t, self.num_heads, self.head_dim),
            p=2.0,
            dim=-1,
        ).reshape(b, t, d)
        k = k * (1 + (a - 1) * self._t(f"{p}.k_a"))
        if layer_idx == 0:
            v_first_sequence = v
        else:
            assert v_first_sequence is not None
            mix = self._lora(
                mixed["v"], f"{p}.v_lora", inner_activation=None, output_sigmoid=True
            )
            v = torch.lerp(v, v_first_sequence, mix)
        decay = torch.exp(-EXP_HALF * torch.sigmoid(w_raw.float()))
        rh = r.reshape(b, t, self.num_heads, self.head_dim)
        kh = k.reshape(b, t, self.num_heads, self.head_dim)
        vh = v.reshape(b, t, self.num_heads, self.head_dim)
        kkh = kk.reshape(b, t, self.num_heads, self.head_dim)
        ah = a.reshape(b, t, self.num_heads, self.head_dim)
        recurrent = state.recurrent[layer_idx]
        outputs = []
        last_shift = state.attn_shift[layer_idx]
        for token_idx in range(t):
            vk = vh[:, token_idx, :, :, None] @ kh[:, token_idx, :, None, :]
            ab = (-kkh[:, token_idx, :, :, None]) @ (
                kkh[:, token_idx, :, None, :] * ah[:, token_idx, :, None, :]
            )
            candidate = (
                recurrent * decay[:, token_idx].reshape(b, self.num_heads, 1, self.head_dim)
                + recurrent @ ab.float()
                + vk.float()
            )
            active = mask[:, token_idx].reshape(b, 1, 1, 1)
            recurrent = torch.where(active, candidate, recurrent)
            # Masked-token output is thrown away at model boundary, so using
            # the candidate here preserves the upstream within-token dataflow.
            out_state = candidate.to(self.dtype)
            outputs.append((out_state @ rh[:, token_idx, :, :, None]).squeeze(-1))
            last_shift = torch.where(mask[:, token_idx, None], hidden[:, token_idx], last_shift)
        state.recurrent[layer_idx] = recurrent
        state.attn_shift[layer_idx] = last_shift
        out = torch.stack(outputs, dim=1).reshape(b, t, d)
        out = F.group_norm(
            out.reshape(b * t, d),
            num_groups=self.num_heads,
            weight=self._t(f"{p}.g_norm.weight"),
            bias=self._t(f"{p}.g_norm.bias"),
            eps=self.head_dim * float(self.config.get("norm_eps", 1e-5)),
        ).reshape(b, t, d)
        correction = (
            rh * kh * self._t(f"{p}.r_k").reshape(1, 1, self.num_heads, self.head_dim)
        ).sum(-1, keepdim=True) * vh
        out = (out + correction.reshape(b, t, d)) * g
        return self._linear(out, self._t(f"{p}.o_proj.weight")), v_first_sequence

    def _ffn(self, layer_idx: int, hidden: torch.Tensor, mask: torch.Tensor, state: OracleState) -> torch.Tensor:
        p = f"model.layers.{layer_idx}.ffn"
        previous = self._masked_shift(hidden, state.ffn_shift[layer_idx], mask)
        mixed = torch.addcmul(hidden, previous - hidden, self._t(f"{p}.x_k"))
        key = torch.relu(self._linear(mixed, self._t(f"{p}.key.weight"))).square()
        out = self._linear(key, self._t(f"{p}.value.weight"))
        last_shift = state.ffn_shift[layer_idx]
        for token_idx in range(hidden.shape[1]):
            last_shift = torch.where(mask[:, token_idx, None], hidden[:, token_idx], last_shift)
        state.ffn_shift[layer_idx] = last_shift
        return out

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        state: OracleState | None = None,
    ) -> OracleOutput:
        ids = torch.as_tensor(input_ids, dtype=torch.long, device="cpu")
        if ids.ndim == 1:
            ids = ids[None]
        if ids.ndim != 2 or ids.numel() == 0:
            raise ValueError("input_ids must be non-empty [batch, sequence]")
        b, t = map(int, ids.shape)
        if attention_mask is None:
            mask = torch.ones(b, t, dtype=torch.bool)
        else:
            mask = torch.as_tensor(attention_mask, dtype=torch.bool, device="cpu")
            if tuple(mask.shape) != (b, t):
                raise ValueError("attention_mask must match input_ids")
        if state is None:
            state = self.init_state(b)
        elif state.valid_tokens.shape != (b,):
            raise ValueError("oracle state batch size does not match input_ids")
        embedding = self._t("model.embeddings.weight")
        x = F.embedding(ids, embedding)
        v_first_sequence = None
        for layer_idx in range(self.num_layers):
            layer = f"model.layers.{layer_idx}"
            residual = self._layer_norm(x, layer + ".pre_norm") if layer_idx == 0 else x
            hidden = self._layer_norm(residual, layer + ".attn_norm")
            attention, v_first_sequence = self._attention(
                layer_idx, hidden, mask, state, v_first_sequence
            )
            x = residual + attention
            residual = x
            hidden = self._layer_norm(x, layer + ".ffn_norm")
            x = residual + self._ffn(layer_idx, hidden, mask, state)
        normalized = self._layer_norm(x, "model.norm")
        # Match the canonical adapter's recurrent padding contract: a masked
        # token repeats the previous normalized output and updates no state.
        prior = torch.zeros(b, self.hidden_size, dtype=self.dtype)
        visible = []
        for token_idx in range(t):
            prior = torch.where(mask[:, token_idx, None], normalized[:, token_idx], prior)
            visible.append(prior)
        normalized = torch.stack(visible, dim=1)
        logits = self._linear(normalized, self._t("lm_head.weight"))
        assert v_first_sequence is not None
        final_v = state.v_first
        for token_idx in range(t):
            final_v = torch.where(mask[:, token_idx, None], v_first_sequence[:, token_idx], final_v)
        state.v_first = final_v
        state.valid_tokens += mask.sum(dim=1, dtype=torch.int64)
        state.processed_width += t
        return OracleOutput(logits=logits, state=state)


def state_tensor_map(state: OracleState, prefix: str) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {
        f"{prefix}.v_first": state.v_first,
        f"{prefix}.valid_tokens": state.valid_tokens,
        f"{prefix}.processed_width": torch.tensor(state.processed_width, dtype=torch.int64),
    }
    for layer, value in enumerate(state.recurrent):
        tensors[f"{prefix}.recurrent.{layer:02d}"] = value
    for layer, value in enumerate(state.attn_shift):
        tensors[f"{prefix}.attn_shift.{layer:02d}"] = value
    for layer, value in enumerate(state.ffn_shift):
        tensors[f"{prefix}.ffn_shift.{layer:02d}"] = value
    return tensors


def _normalized_rmse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    ref = reference.float().reshape(-1)
    cand = candidate.float().reshape(-1)
    rmse = torch.mean((ref - cand).square()).sqrt()
    scale = torch.mean(ref.square()).sqrt().clamp_min(1e-12)
    return float((rmse / scale).item())


def _cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    ref = reference.float().reshape(-1)
    cand = candidate.float().reshape(-1)
    if not (torch.isfinite(ref).all() and torch.isfinite(cand).all()):
        return float("nan")
    if float(ref.norm()) == 0.0 and float(cand.norm()) == 0.0:
        return 1.0
    return float(F.cosine_similarity(ref, cand, dim=0).item())


def evaluate_capture(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    thresholds: Mapping[str, float | bool] = DEFAULT_ACCEPTANCE_THRESHOLDS,
) -> dict[str, object]:
    """Fail-closed logits/state/greedy comparison for an NPU candidate."""
    required = sorted(reference)
    missing = [name for name in required if name not in candidate]
    shape_or_dtype_mismatch = [
        name
        for name in required
        if name in candidate
        and (
            tuple(reference[name].shape) != tuple(candidate[name].shape)
            or reference[name].dtype != candidate[name].dtype
        )
    ]
    if missing or shape_or_dtype_mismatch:
        return {
            "status": "fail",
            "missing_tensors": missing,
            "shape_or_dtype_mismatch": shape_or_dtype_mismatch,
            "gates": {"complete_capture": False, "overall": False},
        }
    logits_names = [name for name in required if ".logits" in name]
    state_names = [name for name in required if ".state." in name and not name.endswith(("valid_tokens", "processed_width"))]
    logits_cos = min((_cosine(reference[n], candidate[n]) for n in logits_names), default=float("nan"))
    logits_nrmse = max((_normalized_rmse(reference[n], candidate[n]) for n in logits_names), default=float("inf"))
    state_cos = min((_cosine(reference[n], candidate[n]) for n in state_names), default=float("nan"))
    state_nrmse = max((_normalized_rmse(reference[n], candidate[n]) for n in state_names), default=float("inf"))
    token_names = [name for name in required if name.endswith("token_ids")]
    metadata_names = [
        name
        for name in required
        if name.endswith(("attention_mask", "valid_tokens", "processed_width"))
    ]
    greedy_exact = all(torch.equal(reference[name], candidate[name]) for name in token_names)
    metadata_exact = all(torch.equal(reference[name], candidate[name]) for name in metadata_names)
    finite = all(bool(torch.isfinite(candidate[name]).all()) for name in required if candidate[name].is_floating_point())
    gates = {
        "complete_capture": True,
        "finite": finite,
        "logits_cosine": logits_cos >= float(thresholds["logits_min_cosine"]),
        "logits_normalized_rmse": logits_nrmse <= float(thresholds["logits_max_normalized_rmse"]),
        "state_cosine": state_cos >= float(thresholds["state_min_cosine"]),
        "state_normalized_rmse": state_nrmse <= float(thresholds["state_max_normalized_rmse"]),
        "greedy_token_ids_exact": greedy_exact if thresholds["greedy_token_ids_exact"] else True,
        "input_and_state_metadata_exact": metadata_exact,
    }
    gates["overall"] = all(gates.values())
    return {
        "status": "pass" if gates["overall"] else "fail",
        "thresholds": dict(thresholds),
        "metrics": {
            "logits_min_cosine": logits_cos,
            "logits_max_normalized_rmse": logits_nrmse,
            "state_min_cosine": state_cos,
            "state_max_normalized_rmse": state_nrmse,
        },
        "gates": gates,
        "missing_tensors": [],
        "shape_or_dtype_mismatch": [],
    }


__all__ = [
    "DEFAULT_ACCEPTANCE_THRESHOLDS",
    "FLA_COMMIT",
    "FLA_REPOSITORY",
    "FLA_SOURCE_SHA256",
    "NaiveRWKV7Oracle",
    "OracleOutput",
    "OracleState",
    "REFERENCE_FORMAT_VERSION",
    "RWKV7_7P2_CHECKPOINT_SHA256",
    "RWKV7_7P2_TOKENIZER_SHA256",
    "SafetensorStore",
    "evaluate_capture",
    "sha256_file",
    "state_tensor_map",
    "tensor_map_sha256",
    "tensor_sha256",
    "verify_files",
    "verify_fla_checkout",
]
