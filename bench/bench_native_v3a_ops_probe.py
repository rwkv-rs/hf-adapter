#!/usr/bin/env python3
# coding=utf-8
"""Probe Albatross v3a exact-row linears inside NativeRWKV7 decode.

This is an external-reference A/B tool. It loads the Apache-2.0 v3a CUDA
extension from a separate checkout, installs narrow inference-only dispatch
hooks, then delegates measurement to ``bench_native_model_decode.py``. The
result identifies kernel boundaries worth porting; it is not a production
runtime dependency.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_TRANSPOSE_CACHE: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}


def _transpose_cached(weight: torch.Tensor) -> torch.Tensor:
    key = (int(weight.data_ptr()), tuple(int(value) for value in weight.shape))
    packed = _TRANSPOSE_CACHE.get(key)
    if packed is None:
        packed = weight.t().contiguous()
        _TRANSPOSE_CACHE[key] = packed
    return packed


def _linear_orig(x: torch.Tensor, weight: torch.Tensor, group: str) -> torch.Tensor:
    original_shape = tuple(x.shape[:-1])
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    rows = int(x2.shape[0])
    if rows == 1:
        if group == "att_c2c":
            use4 = int(weight.shape[1]) < 2048
        elif group == "ffn_key":
            use4 = int(weight.shape[1]) <= 1024
        else:
            use4 = True
        out = torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(
            x2,
            weight,
            128,
            2,
            use4,
        )
    else:
        out = torch.ops.rwkv7_v3a_ops.linear_f16_orig(x2, weight)
    return out.reshape(*original_shape, int(weight.shape[0]))


def _install_probe(mode: str) -> None:
    from rwkv7_hf import native_graph_runtime, native_jit

    original_dispatch = native_jit._native_graph_linear_dispatch
    original_up = native_jit._native_graph_ffn_up_relu2_dispatch
    original_down = native_jit._native_graph_ffn_down_add_dispatch
    original_ffn = native_jit._native_graph_ffn_dispatch
    linear_modes = {"hidden_head", "all", "all_native_ops"}
    ffn_linear_modes = {"ffn_head", "all"}
    lowrank_modes = {"lowrank", "cmix_lowrank", "all_native_ops"}
    cmix_modes = {"cmix", "cmix_lowrank", "all_native_ops"}

    def linear_dispatch(x, weight, *, role: str):
        if not isinstance(weight, torch.Tensor):
            return original_dispatch(x, weight, role=role)
        if role == "head" or (mode in linear_modes and role == "hidden"):
            group = "head" if role == "head" else "att_c2c"
            return _linear_orig(x, weight, group)
        return original_dispatch(x, weight, role=role)

    def ffn_up(x, weight):
        if mode not in ffn_linear_modes or not isinstance(weight, torch.Tensor):
            return original_up(x, weight)
        return torch.relu(_linear_orig(x, weight, "ffn_key")) ** 2

    def ffn_down(x, weight, residual):
        if mode not in ffn_linear_modes or not isinstance(weight, torch.Tensor):
            return original_down(x, weight, residual)
        return residual + _linear_orig(x, weight, "ffn_value")

    def ffn_dispatch(x, up_weight, down_weight, residual, *, sparse_out=None):
        if (
            mode not in cmix_modes
            or not isinstance(up_weight, torch.Tensor)
            or not isinstance(down_weight, torch.Tensor)
        ):
            return original_ffn(
                x,
                up_weight,
                down_weight,
                residual,
                sparse_out=sparse_out,
            )
        x2 = x.reshape(-1, x.shape[-1])
        rows = int(x2.shape[0])
        hidden = int(x2.shape[1])
        ffn = int(up_weight.shape[0])
        preact = _linear_orig(x2, up_weight, "ffn_key")
        packed_down = _transpose_cached(down_weight)
        if rows == 1:
            delta = torch.ops.rwkv7_fast_ops_fp16.cmix_sparse_down_relu_one(
                hidden,
                ffn,
                preact.reshape(-1).contiguous(),
                packed_down,
            )
        elif rows >= 8 and hidden % 512 == 0 and ffn % 512 == 0:
            delta = torch.ops.rwkv7_fast_ops_fp16.cmix_sparse_down_relu_rows_t512(
                rows,
                1,
                hidden,
                ffn,
                preact.reshape(rows, 1, ffn).contiguous(),
                packed_down,
            )
        else:
            delta = torch.ops.rwkv7_fast_ops_fp16.cmix_sparse_down_relu_rows(
                rows,
                1,
                hidden,
                ffn,
                preact.reshape(rows, 1, ffn).contiguous(),
                packed_down,
            )
        return residual + delta.reshape_as(residual)

    def wag_lora(
        xw,
        xa,
        xg,
        w_down,
        a_down,
        g_down,
        w_up,
        a_up,
        g_up,
        w_bias=None,
        a_bias=None,
        g_bias=None,
        **_kwargs,
    ):
        if int(xw.reshape(-1, xw.shape[-1]).shape[0]) > 4:
            w = F.linear(torch.tanh(F.linear(xw, w_down)), w_up, w_bias)
            a = F.linear(F.linear(xa, a_down), a_up, a_bias)
            g = F.linear(torch.sigmoid(F.linear(xg, g_down)), g_up, g_bias)
            return w, a, g
        w_mid, a_mid, g_mid = torch.ops.rwkv7_v3a_ops.linear_wag_rank_in_f16(
            xw.contiguous(),
            xa.contiguous(),
            xg.contiguous(),
            w_down.contiguous(),
            a_down.contiguous(),
            g_down.contiguous(),
        )
        w, a, g = torch.ops.rwkv7_v3a_ops.linear_wag_rank_out_f16(
            w_mid.contiguous(),
            a_mid.contiguous(),
            g_mid.contiguous(),
            w_up.contiguous(),
            a_up.contiguous(),
            g_up.contiguous(),
        )
        if w_bias is not None:
            w = w + w_bias
        if a_bias is not None:
            a = a + a_bias
        if g_bias is not None:
            g = g + g_bias
        return w, a, g

    native_jit._native_graph_linear_dispatch = linear_dispatch
    native_jit._native_graph_ffn_up_relu2_dispatch = ffn_up
    native_jit._native_graph_ffn_down_add_dispatch = ffn_down
    native_jit._native_graph_ffn_dispatch = ffn_dispatch
    native_graph_runtime._native_graph_linear_dispatch = linear_dispatch
    if mode in lowrank_modes:
        native_jit.fused_wag_lora = wag_lora
        os.environ["RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA"] = "1"
    os.environ["RWKV7_NATIVE_GRAPH_V3A_PROBE"] = mode


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v3a-dir", required=True)
    parser.add_argument(
        "--v3a-probe-mode",
        choices=[
            "head",
            "hidden_head",
            "ffn_head",
            "all",
            "lowrank",
            "cmix",
            "cmix_lowrank",
            "all_native_ops",
        ],
        required=True,
    )
    args, remaining = parser.parse_known_args()

    v3a_dir = str(Path(args.v3a_dir).resolve())
    if v3a_dir not in sys.path:
        sys.path.insert(0, v3a_dir)
    import rwkv7_fast_v3a as v3a

    v3a.load_extensions("fp16")
    _install_probe(args.v3a_probe_mode)

    import bench_native_model_decode

    sys.argv = [sys.argv[0], *remaining]
    return bench_native_model_decode.main()


if __name__ == "__main__":
    raise SystemExit(main())
