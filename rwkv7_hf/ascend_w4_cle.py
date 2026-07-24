# coding=utf-8
"""Explicit SmoothQuant/CLE candidate for RWKV-7 squared-ReLU FFNs.

For positive channel scale ``c`` the transformation
``key[j] /= sqrt(c[j])`` and ``value[:, j] *= c[j]`` is mathematically exact
before quantization because ``relu(key(x) / sqrt(c)) ** 2 = relu(key(x)) ** 2 / c``.
Nothing in this module is enabled automatically.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class W4CLECandidate:
    alpha: float | None
    scale: torch.Tensor
    mse: float
    normalized_mse: float
    cosine: float
    max_abs: float


def _groupwise_w4_dequant(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    if weight.dim()!=2 or weight.shape[1]%group_size:
        raise ValueError("weight input dimension must be divisible by group_size")
    n,k=weight.shape;g=weight.reshape(n,k//group_size,group_size).float()
    scale=(g.abs().amax(dim=2)/7.0).clamp_min(1e-8)
    q=torch.round(g/scale[:,:,None]).clamp(-8,7)
    return (q*scale[:,:,None]).reshape(n,k).to(weight.dtype)


def smooth_scale(
    activation_absmax: torch.Tensor,
    value_column_absmax: torch.Tensor,
    *, alpha: float,
    activation_rms: torch.Tensor | None=None,
    min_scale: float=1/16,
    max_scale: float=16,
) -> torch.Tensor:
    if not 0.0<=float(alpha)<=1.0:raise ValueError("alpha must be in [0,1]")
    act=activation_absmax.float().clamp_min(1e-8)
    if activation_rms is not None:
        act=torch.sqrt(act*activation_rms.float().clamp_min(1e-8))
    weight=value_column_absmax.float().clamp_min(1e-8)
    scale=act.pow(float(alpha))/weight.pow(1.0-float(alpha))
    # Remove an irrelevant global magnitude and constrain pathological tails.
    scale=scale/torch.exp(torch.mean(torch.log(scale)))
    return scale.clamp(float(min_scale),float(max_scale))


@torch.no_grad()
def apply_sqrelu_channel_equalization(
    key: nn.Linear, value: nn.Linear, scale: torch.Tensor
) -> None:
    if key.out_features!=value.in_features:raise ValueError("key/value intermediate dimensions differ")
    if scale.numel()!=key.out_features or bool((scale<=0).any()):raise ValueError("scale must be positive per intermediate channel")
    c=scale.to(device=key.weight.device,dtype=key.weight.dtype);root=torch.sqrt(c)
    key.weight.div_(root[:,None])
    if key.bias is not None:key.bias.div_(root)
    value.weight.mul_(c.to(device=value.weight.device,dtype=value.weight.dtype)[None,:])


@torch.no_grad()
def calibrate_sqrelu_value_w4(
    key: nn.Linear,
    value: nn.Linear,
    calibration_inputs: torch.Tensor,
    *,
    group_size: int=128,
    alphas: Iterable[float]=(0.0,0.25,0.5,0.75,1.0),
) -> W4CLECandidate:
    """Choose a per-channel exact CLE scale using a W4 projection oracle.

    The identity candidate is always included, so selection cannot be worse
    than naive groupwise W4 on this calibration tensor under the MSE objective.
    The function returns a scale but does not mutate the modules.
    """
    if key.bias is not None or value.bias is not None:
        # Exact application supports key bias, but RWKV production FFNs are
        # bias-free and the calibration contract stays intentionally narrow.
        raise ValueError("RWKV W4 CLE calibration expects bias-free key/value")
    x=calibration_inputs.to(device=key.weight.device,dtype=key.weight.dtype)
    h=torch.relu(F.linear(x,key.weight)).square()
    ref=F.linear(h,value.weight).float()
    reduce_dims=tuple(range(h.dim()-1))
    amax=h.float().abs().amax(dim=reduce_dims).clamp_min(1e-8)
    rms=h.float().square().mean(dim=reduce_dims).sqrt().clamp_min(1e-8)
    wmax=value.weight.float().abs().amax(dim=0).clamp_min(1e-8)
    scales=[(None,torch.ones_like(amax))]
    scales.extend((float(a),smooth_scale(amax,wmax,alpha=float(a),activation_rms=rms)) for a in alphas)
    candidates=[]
    denom=float(ref.square().mean().clamp_min(1e-12))
    for alpha,cpu_scale in scales:
        c=cpu_scale.to(device=h.device,dtype=h.dtype)
        equalized=value.weight*c.to(value.weight.dtype)[None,:]
        dequant=_groupwise_w4_dequant(equalized,group_size)
        out=F.linear(h/c,dequant).float()
        diff=out-ref;mse=float(diff.square().mean());cos=float(F.cosine_similarity(ref.flatten(),out.flatten(),dim=0))
        candidates.append(W4CLECandidate(alpha,cpu_scale.detach().cpu(),mse,mse/denom,cos,float(diff.abs().max())))
    return min(candidates,key=lambda item:item.mse)


__all__=["W4CLECandidate","apply_sqrelu_channel_equalization","calibrate_sqrelu_value_w4","smooth_scale"]
