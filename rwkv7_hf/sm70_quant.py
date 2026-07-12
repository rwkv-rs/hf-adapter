# coding=utf-8
"""Exact-sm70 W8/W4 projection kernels for graph-captured RWKV decode.

B1 uses a single weight-only warp kernel. B2/B4/B8 dynamically quantize each
activation row and use DP4A while loading every quantized weight row once and
accumulating all batch rows in registers. The extension is lazy, graph-safe,
and has CPU/non-sm70 fallbacks.
"""
from __future__ import annotations
import os, sys, threading
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


_CPP = r"""
#include <torch/extension.h>
torch::Tensor rwkv7_sm70_w8_cuda(torch::Tensor,torch::Tensor,torch::Tensor,int64_t);
torch::Tensor rwkv7_sm70_w8_out_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);
torch::Tensor rwkv7_sm70_w4_cuda(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);
torch::Tensor rwkv7_sm70_w4_out_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);
torch::Tensor rwkv7_sm70_w4_relu2_cuda(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("w8",&rwkv7_sm70_w8_cuda);m.def("w8_out",&rwkv7_sm70_w8_out_cuda);m.def("w4",&rwkv7_sm70_w4_cuda);m.def("w4_out",&rwkv7_sm70_w4_out_cuda);m.def("w4_relu2",&rwkv7_sm70_w4_relu2_cuda);}
"""
_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
namespace {
__device__ inline float warp_max(float v){for(int d=16;d;d>>=1)v=fmaxf(v,__shfl_down_sync(0xffffffff,v,d));return v;}
__global__ void quant_a8(int M,int K,const half* x,signed char* q,half* scales){
 int m=blockIdx.x,tid=threadIdx.x; const half* xr=x+(int64_t)m*K; signed char* qr=q+(int64_t)m*K; float mx=0;
 for(int k=tid;k<K;k+=blockDim.x)mx=fmaxf(mx,fabsf(__half2float(xr[k])));
 mx=warp_max(mx); __shared__ float sm[8]; if((tid&31)==0)sm[tid>>5]=mx; __syncthreads();
 if(tid<32){float v=tid<(blockDim.x>>5)?sm[tid]:0.f;v=warp_max(v);if(tid==0){sm[0]=fmaxf(v/127.f,1e-6f);scales[m]=__float2half_rn(sm[0]);}} __syncthreads();
 float inv=1.f/sm[0]; for(int k=tid;k<K;k+=blockDim.x){int v=__float2int_rn(__half2float(xr[k])*inv);qr[k]=(signed char)max(-127,min(127,v));}
}
__device__ inline float warp_sum(float v){for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);return v;}
__global__ void w8_a16_single(int K,int N,const half* x,const signed char* q,const half* ws,half* y){
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,warps=blockDim.x>>5,n=blockIdx.x*warps+warp;if(n>=N)return;
 const half2* xr=(const half2*)x;const char2* wr=(const char2*)(q+(int64_t)n*K);float acc=0.f;
 for(int k2=lane;k2<(K>>1);k2+=32){float2 xv=__half22float2(xr[k2]);char2 w=wr[k2];acc=fmaf(xv.x,float(w.x),acc);acc=fmaf(xv.y,float(w.y),acc);}
 acc=warp_sum(acc);if(lane==0)y[n]=__float2half_rn(acc*__half2float(ws[n]));
}
__global__ void w8_dp4a(int M,int K,int N,const signed char* x,const half* xs,const signed char* q,const half* ws,half* y){
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,warps=blockDim.x>>5,n=blockIdx.x*warps+warp;if(n>=N)return;
 const int* wr=(const int*)(q+(int64_t)n*K);int acc[8]={0,0,0,0,0,0,0,0};
 for(int k4=lane;k4<(K>>2);k4+=32){int w=wr[k4];
  #pragma unroll
  for(int m=0;m<8;m++)if(m<M)acc[m]=__dp4a(((const int*)(x+(int64_t)m*K))[k4],w,acc[m]);}
 #pragma unroll
 for(int m=0;m<8;m++)if(m<M){int v=acc[m];for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);if(lane==0)y[(int64_t)m*N+n]=__float2half_rn(float(v)*__half2float(xs[m])*__half2float(ws[n]));}
}
__device__ inline int pack4(int a,int b,int c,int d){return (a&255)|((b&255)<<8)|((c&255)<<16)|((d&255)<<24);}
__global__ void w4_a16_single(int K,int N,int KH,const half* x,const unsigned char* q,const half* ws,half* y,bool relu2){
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,warps=blockDim.x>>5,n=blockIdx.x*warps+warp;if(n>=N)return;
 const half2* xr=(const half2*)x;const unsigned char* wr=q+(int64_t)n*KH;float acc=0.f;
 for(int k2=lane;k2<KH;k2+=32){float2 xv=__half22float2(xr[k2]);unsigned char b=wr[k2];acc=fmaf(xv.x,float(int(b&15)-8),acc);acc=fmaf(xv.y,float(int(b>>4)-8),acc);}
 acc=warp_sum(acc);if(lane==0){half h=__float2half_rn(acc*__half2float(ws[n]));if(relu2){float z=fmaxf(__half2float(h),0.f);h=__float2half_rn(z*z);}y[n]=h;}
}
__global__ void w4_dp4a(int M,int K,int N,int KH,const signed char* x,const half* xs,const unsigned char* q,const half* ws,half* y,bool relu2){
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,warps=blockDim.x>>5,n=blockIdx.x*warps+warp;if(n>=N)return;
 const unsigned char* wr=q+(int64_t)n*KH;int acc[8]={0,0,0,0,0,0,0,0};
 for(int u=lane;u<(K>>3);u+=32){int j=u<<2;unsigned b0=wr[j],b1=wr[j+1],b2=wr[j+2],b3=wr[j+3];
  int p0=pack4(int(b0&15)-8,int(b0>>4)-8,int(b1&15)-8,int(b1>>4)-8);int p1=pack4(int(b2&15)-8,int(b2>>4)-8,int(b3&15)-8,int(b3>>4)-8);
  #pragma unroll
  for(int m=0;m<8;m++)if(m<M){const int* xr=(const int*)(x+(int64_t)m*K);acc[m]=__dp4a(xr[u*2],p0,acc[m]);acc[m]=__dp4a(xr[u*2+1],p1,acc[m]);}}
 #pragma unroll
 for(int m=0;m<8;m++)if(m<M){int v=acc[m];for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);if(lane==0){half h=__float2half_rn(float(v)*__half2float(xs[m])*__half2float(ws[n]));if(relu2){float z=fmaxf(__half2float(h),0.f);h=__float2half_rn(z*z);}y[(int64_t)m*N+n]=h;}}
}
void check(torch::Tensor x,torch::Tensor q,torch::Tensor s,torch::Tensor y,int N){TORCH_CHECK(x.is_cuda()&&q.is_cuda()&&s.is_cuda()&&y.is_cuda(),"CUDA required");TORCH_CHECK(x.scalar_type()==at::kHalf&&s.scalar_type()==at::kHalf&&y.scalar_type()==at::kHalf,"fp16 activation/scales/output required");TORCH_CHECK(x.dim()==2&&x.is_contiguous()&&q.is_contiguous()&&s.is_contiguous()&&y.is_contiguous(),"contiguous rank-2 activation required");TORCH_CHECK(x.size(1)%8==0&&x.size(0)<=8,"K multiple of 8 and M<=8 required");TORCH_CHECK(y.size(0)==x.size(0)&&y.size(1)==N,"output shape mismatch");}
torch::Tensor run8(torch::Tensor x,torch::Tensor q,torch::Tensor s,torch::Tensor y,int th){int M=x.size(0),K=x.size(1),N=q.size(0);check(x,q,s,y,N);TORCH_CHECK(q.scalar_type()==at::kChar&&q.size(1)==K,"int8 weight shape mismatch");auto st=at::cuda::getCurrentCUDAStream();if(M==1){w8_a16_single<<<dim3((N+(th/32)-1)/(th/32)),th,0,st>>>(K,N,(half*)x.data_ptr<at::Half>(),(signed char*)q.data_ptr<int8_t>(),(half*)s.data_ptr<at::Half>(),(half*)y.data_ptr<at::Half>());}else{auto qa=torch::empty({M,K},q.options());auto as=torch::empty({M},s.options());quant_a8<<<M,256,0,st>>>(M,K,(half*)x.data_ptr<at::Half>(),(signed char*)qa.data_ptr<int8_t>(),(half*)as.data_ptr<at::Half>());w8_dp4a<<<dim3((N+(th/32)-1)/(th/32)),th,0,st>>>(M,K,N,(signed char*)qa.data_ptr<int8_t>(),(half*)as.data_ptr<at::Half>(),(signed char*)q.data_ptr<int8_t>(),(half*)s.data_ptr<at::Half>(),(half*)y.data_ptr<at::Half>());}C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
torch::Tensor run4(torch::Tensor x,torch::Tensor q,torch::Tensor s,torch::Tensor y,int N,bool relu2){int M=x.size(0),K=x.size(1),KH=q.size(1);check(x,q,s,y,N);TORCH_CHECK(q.scalar_type()==at::kByte&&KH*2>=K,"uint8 weight shape mismatch");auto st=at::cuda::getCurrentCUDAStream();if(M==1){w4_a16_single<<<dim3((N+7)/8),256,0,st>>>(K,N,KH,(half*)x.data_ptr<at::Half>(),(unsigned char*)q.data_ptr<uint8_t>(),(half*)s.data_ptr<at::Half>(),(half*)y.data_ptr<at::Half>(),relu2);}else{auto qa=torch::empty({M,K},torch::TensorOptions().device(x.device()).dtype(torch::kInt8));auto as=torch::empty({M},s.options());quant_a8<<<M,256,0,st>>>(M,K,(half*)x.data_ptr<at::Half>(),(signed char*)qa.data_ptr<int8_t>(),(half*)as.data_ptr<at::Half>());w4_dp4a<<<dim3((N+7)/8),256,0,st>>>(M,K,N,KH,(signed char*)qa.data_ptr<int8_t>(),(half*)as.data_ptr<at::Half>(),(unsigned char*)q.data_ptr<uint8_t>(),(half*)s.data_ptr<at::Half>(),(half*)y.data_ptr<at::Half>(),relu2);}C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
}
torch::Tensor rwkv7_sm70_w8_cuda(torch::Tensor x,torch::Tensor q,torch::Tensor s,int64_t th){auto y=torch::empty({x.size(0),q.size(0)},x.options());return run8(x,q,s,y,th);}
torch::Tensor rwkv7_sm70_w8_out_cuda(torch::Tensor x,torch::Tensor q,torch::Tensor s,torch::Tensor y,int64_t th){return run8(x,q,s,y,th);}
torch::Tensor rwkv7_sm70_w4_cuda(torch::Tensor x,torch::Tensor q,torch::Tensor s,int64_t N,int64_t){auto y=torch::empty({x.size(0),N},x.options());return run4(x,q,s,y,N,false);}
torch::Tensor rwkv7_sm70_w4_out_cuda(torch::Tensor x,torch::Tensor q,torch::Tensor s,torch::Tensor y,int64_t N,int64_t){return run4(x,q,s,y,N,false);}
torch::Tensor rwkv7_sm70_w4_relu2_cuda(torch::Tensor x,torch::Tensor q,torch::Tensor s,int64_t N,int64_t){auto y=torch::empty({x.size(0),N},x.options());return run4(x,q,s,y,N,true);}
"""

_EXT = None
_ERR = None
_LOCK = threading.Lock()


def is_sm70(device=None):
    if torch is None or not torch.cuda.is_available():
        return False
    d = torch.device("cuda" if device is None else device)
    if d.type != "cuda":
        return False
    i = torch.cuda.current_device() if d.index is None else d.index
    return tuple(torch.cuda.get_device_capability(i)) == (7, 0)


def _load():
    global _EXT, _ERR
    if _EXT is not None:
        return _EXT
    if _ERR is not None or not is_sm70():
        return None
    with _LOCK:
        try:
            pb = str(Path(sys.executable).resolve().parent)
            os.environ["PATH"] = pb + os.pathsep + os.environ.get("PATH", "")
            nv = Path(pb) / "nvcc"
            if nv.exists():
                os.environ.setdefault("CUDA_HOME", str(nv.parent.parent))
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
            rt = (
                Path(sys.prefix)
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
                / "nvidia"
                / "cuda_runtime"
                / "lib"
            )
            ld = [f"-L{rt}", f"-Wl,-rpath,{rt}"] if rt.is_dir() else []
            from torch.utils.cpp_extension import load_inline

            _EXT = load_inline(
                name="rwkv7_sm70_quant_v9",
                cpp_sources=_CPP,
                cuda_sources=_CUDA,
                functions=None,
                extra_cflags=["-O3"],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "--extra-device-vectorization",
                ],
                extra_ldflags=ld,
                with_cuda=True,
                verbose=False,
            )
        except Exception as e:
            _ERR = f"{type(e).__name__}: {e}"
    return _EXT


def build_error():
    return _ERR


def quantize_w8_row(weight):
    w = weight.detach().float()
    s = (w.abs().amax(1) / 127).clamp_min(1e-6)
    q = (w / s[:, None]).round().clamp(-127, 127).to(torch.int8)
    return q.contiguous(), s.to(weight.dtype)


def quantize_w4_row(weight):
    w = weight.detach().float()
    s = (w.abs().amax(1) / 7).clamp_min(1e-6)
    q = (w / s[:, None]).round().clamp(-7, 7).to(torch.int16) + 8
    if q.shape[1] & 1:
        q = F.pad(q, (0, 1), value=8)
    packed = ((q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8)).contiguous()
    return packed, s.to(weight.dtype), weight.shape[1]


def w8_linear(x, q, s, out=None):
    scalar = x.dim() == 1
    x2 = x.reshape(-1, x.shape[-1])
    e = _load() if x2.is_cuda and is_sm70(x2.device) else None
    if e is None:
        result = F.linear(x, q.to(x.dtype) * s[:, None])
        if out is not None:
            out.copy_(result)
            return out
        return result
    raw_threads = os.environ.get("RWKV7_SM70_W8_THREADS")
    o = (
        int(raw_threads)
        if raw_threads is not None
        else (256 if int(x2.shape[0]) == 1 else 128)
    )
    y = (
        e.w8(x2.contiguous(), q, s, o)
        if out is None
        else e.w8_out(x2.contiguous(), q, s, out.reshape(x2.shape[0], q.shape[0]), o)
    )
    return y.reshape(q.shape[0]) if scalar else y.reshape(*x.shape[:-1], q.shape[0])


def w4_linear(x, q, s, out_features, in_features, out=None):
    scalar = x.dim() == 1
    x2 = x.reshape(-1, x.shape[-1])
    e = _load() if x2.is_cuda and is_sm70(x2.device) else None
    if e is None:
        lo = (q & 15).to(x.dtype) - 8
        hi = (q >> 4).to(x.dtype) - 8
        w = torch.empty(out_features, q.shape[1] * 2, device=q.device, dtype=x.dtype)
        w[:, 0::2] = lo
        w[:, 1::2] = hi
        result = F.linear(x, w[:, :in_features] * s[:, None])
        if out is not None:
            out.copy_(result)
            return out
        return result
    o = int(os.environ.get("RWKV7_SM70_QUANT_OUT_TILE", "4"))
    y = (
        e.w4(x2.contiguous(), q, s, int(out_features), o)
        if out is None
        else e.w4_out(
            x2.contiguous(),
            q,
            s,
            out.reshape(x2.shape[0], out_features),
            int(out_features),
            o,
        )
    )
    return y.reshape(out_features) if scalar else y.reshape(*x.shape[:-1], out_features)


def w4_linear_relu2(x, q, s, out_features, in_features):
    """Apply exact-sm70 W4 linear with an in-kernel ReLU-square epilogue."""

    scalar = x.dim() == 1
    x2 = x.reshape(-1, x.shape[-1])
    e = _load() if x2.is_cuda and is_sm70(x2.device) else None
    if e is None:
        return torch.relu(w4_linear(x, q, s, out_features, in_features)) ** 2
    tile = int(os.environ.get("RWKV7_SM70_QUANT_OUT_TILE", "4"))
    y = e.w4_relu2(x2.contiguous(), q, s, int(out_features), tile)
    return y.reshape(out_features) if scalar else y.reshape(*x.shape[:-1], out_features)
