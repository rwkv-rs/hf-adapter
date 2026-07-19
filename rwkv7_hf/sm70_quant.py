# coding=utf-8
"""Exact-sm70 W8/W4 projection kernels for graph-captured RWKV decode.

B1 uses a single weight-only warp kernel. B2/B4/B8 dynamically quantize each
activation row and use DP4A while loading every quantized weight row once and
accumulating all batch rows in registers. The extension is lazy, graph-safe,
and has CPU/non-sm70 fallbacks.
"""
from __future__ import annotations
import os
import sys
import threading
from pathlib import Path

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
torch::Tensor rwkv7_sm70_w8_dequant_out_cuda(torch::Tensor,torch::Tensor,torch::Tensor,int64_t);
torch::Tensor rwkv7_sm70_w4_cuda(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);
torch::Tensor rwkv7_sm70_w4_out_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);
torch::Tensor rwkv7_sm70_w4_relu2_cuda(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);
torch::Tensor rwkv7_sm70_w4_add_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);
torch::Tensor rwkv7_sm70_w4_group_cuda(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);
torch::Tensor rwkv7_sm70_w4_group_out_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);
torch::Tensor rwkv7_sm70_w4_dequant_cuda(torch::Tensor,torch::Tensor,int64_t,int64_t);
torch::Tensor rwkv7_sm70_w4_dequant_out_cuda(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("w8",&rwkv7_sm70_w8_cuda);m.def("w8_out",&rwkv7_sm70_w8_out_cuda);m.def("w8_dequant_out",&rwkv7_sm70_w8_dequant_out_cuda);m.def("w4",&rwkv7_sm70_w4_cuda);m.def("w4_out",&rwkv7_sm70_w4_out_cuda);m.def("w4_relu2",&rwkv7_sm70_w4_relu2_cuda);m.def("w4_add",&rwkv7_sm70_w4_add_cuda);m.def("w4_group",&rwkv7_sm70_w4_group_cuda);m.def("w4_group_out",&rwkv7_sm70_w4_group_out_cuda);m.def("w4_dequant",&rwkv7_sm70_w4_dequant_cuda);m.def("w4_dequant_out",&rwkv7_sm70_w4_dequant_out_cuda);}
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
__device__ inline float half_warp_sum(float v){unsigned mask=__activemask();for(int d=8;d;d>>=1)v+=__shfl_down_sync(mask,v,d,16);return v;}
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
__global__ void w8_dequant_kernel(int N,int K,const signed char* q,const half* s,half* w){
 int K2=K>>1,total=N*K2;const char2* q2=(const char2*)q;half2* w2=(half2*)w;for(int idx=blockIdx.x*blockDim.x+threadIdx.x;idx<total;idx+=blockDim.x*gridDim.x){int n=idx/K2;char2 v=q2[idx];float scale=__half2float(s[n]);w2[idx]=__halves2half2(__float2half_rn(float(v.x)*scale),__float2half_rn(float(v.y)*scale));}
}
__device__ inline int pack4(int a,int b,int c,int d){return (a&255)|((b&255)<<8)|((c&255)<<16)|((d&255)<<24);}
template<int BN,int TN,int MODE>
__global__ void w4_a16_bn_tn(int K,int N,int KH,const half* x,const unsigned char* q,const half* ws,const half* residual,half* y){
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,n0=blockIdx.x*BN+warp*TN;
 const half2* xr=(const half2*)x;float acc[TN]={0.f};
 for(int k2=lane;k2<KH;k2+=32){
  float2 xv=__half22float2(xr[k2]);
  #pragma unroll
  for(int t=0;t<TN;t++){int n=n0+t;if(n<N){unsigned char b=q[(int64_t)n*KH+k2];acc[t]=fmaf(xv.x,float(int(b&15)-8),acc[t]);acc[t]=fmaf(xv.y,float(int(b>>4)-8),acc[t]);}}
 }
 #pragma unroll
 for(int t=0;t<TN;t++){int n=n0+t;if(n<N){float v=warp_sum(acc[t]);if(lane==0){half h=__float2half_rn(v*__half2float(ws[n]));if(MODE==1){float z=fmaxf(__half2float(h),0.f);h=__float2half_rn(z*z);}else if(MODE==2){h=__hadd(h,residual[n]);}y[n]=h;}}}
}
template<int BN,int TN,int MODE>
__global__ void w4_dp4a_bn_tn(int M,int K,int N,int KH,const signed char* x,const half* xs,const unsigned char* q,const half* ws,const half* residual,half* y){
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,n0=blockIdx.x*BN+warp*TN,mb=blockIdx.y*8;int acc[TN][8]={};
 for(int u=lane;u<(K>>3);u+=32){
  #pragma unroll
  for(int t=0;t<TN;t++){int n=n0+t;if(n<N){const unsigned char* wr=q+(int64_t)n*KH;int j=u<<2;unsigned b0=wr[j],b1=wr[j+1],b2=wr[j+2],b3=wr[j+3];
   int p0=pack4(int(b0&15)-8,int(b0>>4)-8,int(b1&15)-8,int(b1>>4)-8);int p1=pack4(int(b2&15)-8,int(b2>>4)-8,int(b3&15)-8,int(b3>>4)-8);
   #pragma unroll
   for(int m=0;m<8;m++)if(mb+m<M){const int* xr=(const int*)(x+(int64_t)(mb+m)*K);acc[t][m]=__dp4a(xr[u*2],p0,acc[t][m]);acc[t][m]=__dp4a(xr[u*2+1],p1,acc[t][m]);}
  }}
 }
 #pragma unroll
 for(int t=0;t<TN;t++){int n=n0+t;if(n<N){
  #pragma unroll
  for(int m=0;m<8;m++)if(mb+m<M){int v=acc[t][m];for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);if(lane==0){int64_t idx=(int64_t)(mb+m)*N+n;half h=__float2half_rn(float(v)*__half2float(xs[mb+m])*__half2float(ws[n]));if(MODE==1){float z=fmaxf(__half2float(h),0.f);h=__float2half_rn(z*z);}else if(MODE==2){h=__hadd(h,residual[idx]);}y[idx]=h;}}
 }}
}
template<int BN,int TN,int GS>
__global__ void w4g_a16_bn_tn(int K,int N,int KH,int G,const half* x,const unsigned char* q,const half* ws,half* y){
 int lane=threadIdx.x&15,subwarp=threadIdx.x>>4,n0=blockIdx.x*BN+subwarp*TN;const half2* xr=(const half2*)x;float acc[TN]={0.f};
 #pragma unroll
 for(int g=0;g<G;g++){
  float scale[TN];
  #pragma unroll
  for(int t=0;t<TN;t++){int n=n0+t;scale[t]=n<N?__half2float(ws[(int64_t)n*G+g]):0.f;}
  int begin=g*(GS>>1),end=min(begin+(GS>>1),KH);
  for(int k2=begin+lane;k2<end;k2+=16){float2 xv=__half22float2(xr[k2]);
   #pragma unroll
   for(int t=0;t<TN;t++){int n=n0+t;if(n<N){unsigned char b=q[(int64_t)n*KH+k2];float s=scale[t];acc[t]=fmaf(xv.x,float(int(b&15)-8)*s,acc[t]);acc[t]=fmaf(xv.y,float(int(b>>4)-8)*s,acc[t]);}}
  }
 }
 #pragma unroll
 for(int t=0;t<TN;t++){int n=n0+t;if(n<N){float v=half_warp_sum(acc[t]);if(lane==0)y[n]=__float2half_rn(v);}}
}
template<int BN,int TN,int GS>
__global__ void w4g_dp4a_bn_tn(int M,int K,int N,int KH,int G,const signed char* x,const half* xs,const unsigned char* q,const half* ws,half* y){
 int lane=threadIdx.x&15,subwarp=threadIdx.x>>4,n0=blockIdx.x*BN+subwarp*TN,mb=blockIdx.y*8;float acc[TN][8]={};
 #pragma unroll
 for(int g=0;g<G;g++){
  float scale[TN];
  #pragma unroll
  for(int t=0;t<TN;t++){int n=n0+t;scale[t]=n<N?__half2float(ws[(int64_t)n*G+g]):0.f;}
  int begin=g*(GS>>3),end=min(begin+(GS>>3),K>>3);
  for(int u=begin+lane;u<end;u+=16){
   #pragma unroll
   for(int t=0;t<TN;t++){int n=n0+t;if(n<N){const unsigned char* wr=q+(int64_t)n*KH;int j=u<<2;unsigned b0=wr[j],b1=wr[j+1],b2=wr[j+2],b3=wr[j+3];
    int p0=pack4(int(b0&15)-8,int(b0>>4)-8,int(b1&15)-8,int(b1>>4)-8);int p1=pack4(int(b2&15)-8,int(b2>>4)-8,int(b3&15)-8,int(b3>>4)-8);
    #pragma unroll
    for(int m=0;m<8;m++)if(mb+m<M){const int* xr=(const int*)(x+(int64_t)(mb+m)*K);int v=__dp4a(xr[u*2],p0,0);v=__dp4a(xr[u*2+1],p1,v);acc[t][m]+=float(v)*scale[t];}
   }}
  }
 }
 #pragma unroll
 for(int t=0;t<TN;t++){int n=n0+t;if(n<N){
  #pragma unroll
  for(int m=0;m<8;m++)if(mb+m<M){float v=half_warp_sum(acc[t][m]);if(lane==0)y[(int64_t)(mb+m)*N+n]=__float2half_rn(v*__half2float(xs[mb+m]));}
 }}
}
template<int GS>
__global__ void w4_dequant_kernel(int N,int KH,int G,const unsigned char* q,const half* s,half* w){
 int total=N*KH;for(int idx=blockIdx.x*blockDim.x+threadIdx.x;idx<total;idx+=blockDim.x*gridDim.x){int n=idx/KH,k2=idx-n*KH;unsigned char b=q[idx];float scale;if constexpr(GS==0)scale=__half2float(s[n]);else scale=__half2float(s[(int64_t)n*G+(k2*2)/GS]);((half2*)w)[idx]=__halves2half2(__float2half_rn(float(int(b&15)-8)*scale),__float2half_rn(float(int(b>>4)-8)*scale));}
}
void check(torch::Tensor x,torch::Tensor q,torch::Tensor s,torch::Tensor y,int N){TORCH_CHECK(x.is_cuda()&&q.is_cuda()&&s.is_cuda()&&y.is_cuda(),"CUDA required");TORCH_CHECK(x.scalar_type()==at::kHalf&&s.scalar_type()==at::kHalf&&y.scalar_type()==at::kHalf,"fp16 activation/scales/output required");TORCH_CHECK(x.dim()==2&&x.is_contiguous()&&q.is_contiguous()&&s.is_contiguous()&&y.is_contiguous(),"contiguous rank-2 activation required");TORCH_CHECK(x.size(1)%8==0&&x.size(0)>0,"K multiple of 8 and non-empty M required");TORCH_CHECK(y.size(0)==x.size(0)&&y.size(1)==N,"output shape mismatch");}
torch::Tensor run8(torch::Tensor x,torch::Tensor q,torch::Tensor s,torch::Tensor y,int th){int M=x.size(0),K=x.size(1),N=q.size(0);check(x,q,s,y,N);TORCH_CHECK(M<=8,"sm70 W8 decode kernel requires M<=8");TORCH_CHECK(q.scalar_type()==at::kChar&&q.size(1)==K,"int8 weight shape mismatch");auto st=at::cuda::getCurrentCUDAStream();if(M==1){w8_a16_single<<<dim3((N+(th/32)-1)/(th/32)),th,0,st>>>(K,N,(half*)x.data_ptr<at::Half>(),(signed char*)q.data_ptr<int8_t>(),(half*)s.data_ptr<at::Half>(),(half*)y.data_ptr<at::Half>());}else{auto qa=torch::empty({M,K},q.options());auto as=torch::empty({M},s.options());quant_a8<<<M,256,0,st>>>(M,K,(half*)x.data_ptr<at::Half>(),(signed char*)qa.data_ptr<int8_t>(),(half*)as.data_ptr<at::Half>());w8_dp4a<<<dim3((N+(th/32)-1)/(th/32)),th,0,st>>>(M,K,N,(signed char*)qa.data_ptr<int8_t>(),(half*)as.data_ptr<at::Half>(),(signed char*)q.data_ptr<int8_t>(),(half*)s.data_ptr<at::Half>(),(half*)y.data_ptr<at::Half>());}C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
template<int MODE>
torch::Tensor run4(torch::Tensor x,torch::Tensor q,torch::Tensor s,torch::Tensor residual,torch::Tensor y,int N,int bn,int tn){int M=x.size(0),K=x.size(1),KH=q.size(1);check(x,q,s,y,N);TORCH_CHECK(q.scalar_type()==at::kByte&&KH*2>=K,"uint8 weight shape mismatch");if constexpr(MODE==2){TORCH_CHECK(residual.is_cuda()&&residual.scalar_type()==at::kHalf&&residual.is_contiguous()&&residual.sizes()==y.sizes(),"fp16 contiguous residual shape mismatch");}const half* rp=MODE==2?(half*)residual.data_ptr<at::Half>():nullptr;auto st=at::cuda::getCurrentCUDAStream();bool launched=false;auto qa=M==1?torch::Tensor():torch::empty({M,K},torch::TensorOptions().device(x.device()).dtype(torch::kInt8));auto as=M==1?torch::Tensor():torch::empty({M},s.options());if(M>1)quant_a8<<<M,256,0,st>>>(M,K,(half*)x.data_ptr<at::Half>(),(signed char*)qa.data_ptr<int8_t>(),(half*)as.data_ptr<at::Half>());
#define LAUNCH4(BN_,TN_) if(bn==BN_&&tn==TN_){constexpr int TH=(BN_/TN_)*32;if(M==1)w4_a16_bn_tn<BN_,TN_,MODE><<<dim3((N+BN_-1)/BN_),TH,0,st>>>(K,N,KH,(half*)x.data_ptr<at::Half>(),(unsigned char*)q.data_ptr<uint8_t>(),(half*)s.data_ptr<at::Half>(),rp,(half*)y.data_ptr<at::Half>());else w4_dp4a_bn_tn<BN_,TN_,MODE><<<dim3((N+BN_-1)/BN_,(M+7)/8),TH,0,st>>>(M,K,N,KH,(signed char*)qa.data_ptr<int8_t>(),(half*)as.data_ptr<at::Half>(),(unsigned char*)q.data_ptr<uint8_t>(),(half*)s.data_ptr<at::Half>(),rp,(half*)y.data_ptr<at::Half>());launched=true;}
 if(!launched){LAUNCH4(1,1) LAUNCH4(2,1) LAUNCH4(4,1) LAUNCH4(4,2) LAUNCH4(4,4) LAUNCH4(8,1) LAUNCH4(8,2) LAUNCH4(8,4) LAUNCH4(16,1) LAUNCH4(16,2) LAUNCH4(16,4) LAUNCH4(32,1) LAUNCH4(32,2)}
#undef LAUNCH4
 TORCH_CHECK(launched,"unsupported sm70 W4 BN/TN pair");C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
torch::Tensor run4g(torch::Tensor x,torch::Tensor q,torch::Tensor s,torch::Tensor y,int N,int gs,int bn,int tn){int M=x.size(0),K=x.size(1),KH=q.size(1);check(x,q,s,y,N);TORCH_CHECK((gs==128||gs==256)&&K%gs==0,"sm70 groupwise W4 requires group_size=128/256 and divisible K");int G=K/gs;TORCH_CHECK(q.scalar_type()==at::kByte&&KH*2==K,"uint8 groupwise weight shape mismatch");TORCH_CHECK(s.dim()==2&&s.size(0)==N&&s.size(1)==G,"groupwise scale shape mismatch");auto st=at::cuda::getCurrentCUDAStream();bool launched=false;auto qa=M==1?torch::Tensor():torch::empty({M,K},torch::TensorOptions().device(x.device()).dtype(torch::kInt8));auto as=M==1?torch::Tensor():torch::empty({M},s.options());if(M>1)quant_a8<<<M,256,0,st>>>(M,K,(half*)x.data_ptr<at::Half>(),(signed char*)qa.data_ptr<int8_t>(),(half*)as.data_ptr<at::Half>());
#define LAUNCH4G(BN_,TN_,GS_) if(!launched&&gs==GS_&&bn==BN_&&tn==TN_){constexpr int TH=(BN_/TN_)*16;if(M==1)w4g_a16_bn_tn<BN_,TN_,GS_><<<dim3((N+BN_-1)/BN_),TH,0,st>>>(K,N,KH,G,(half*)x.data_ptr<at::Half>(),(unsigned char*)q.data_ptr<uint8_t>(),(half*)s.data_ptr<at::Half>(),(half*)y.data_ptr<at::Half>());else w4g_dp4a_bn_tn<BN_,TN_,GS_><<<dim3((N+BN_-1)/BN_,(M+7)/8),TH,0,st>>>(M,K,N,KH,G,(signed char*)qa.data_ptr<int8_t>(),(half*)as.data_ptr<at::Half>(),(unsigned char*)q.data_ptr<uint8_t>(),(half*)s.data_ptr<at::Half>(),(half*)y.data_ptr<at::Half>());launched=true;}
#define LAUNCH4G_SET(GS_) LAUNCH4G(1,1,GS_) LAUNCH4G(2,1,GS_) LAUNCH4G(4,1,GS_) LAUNCH4G(4,2,GS_) LAUNCH4G(4,4,GS_) LAUNCH4G(8,1,GS_) LAUNCH4G(8,2,GS_) LAUNCH4G(8,4,GS_) LAUNCH4G(16,1,GS_) LAUNCH4G(16,2,GS_) LAUNCH4G(16,4,GS_) LAUNCH4G(32,1,GS_) LAUNCH4G(32,2,GS_)
  LAUNCH4G_SET(128) LAUNCH4G_SET(256)
#undef LAUNCH4G_SET
#undef LAUNCH4G
 TORCH_CHECK(launched,"unsupported sm70 groupwise W4 BN/TN pair");C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
torch::Tensor dequant4_out(torch::Tensor q,torch::Tensor s,torch::Tensor w,int gs,int block_cap){TORCH_CHECK(q.is_cuda()&&s.is_cuda()&&w.is_cuda()&&q.scalar_type()==at::kByte&&s.scalar_type()==at::kHalf&&w.scalar_type()==at::kHalf&&q.dim()==2&&q.is_contiguous()&&s.is_contiguous()&&w.is_contiguous(),"sm70 W4 dequant requires contiguous CUDA uint8 weights and fp16 scales/output");int N=q.size(0),KH=q.size(1),K=KH*2,G=gs?K/gs:1;TORCH_CHECK(w.dim()==2&&w.size(0)==N&&w.size(1)==K,"sm70 W4 dequant output shape mismatch");if(gs==0){TORCH_CHECK(s.dim()==1&&s.size(0)==N,"rowwise scale shape mismatch");}else{TORCH_CHECK((gs==128||gs==256)&&K%gs==0&&s.dim()==2&&s.size(0)==N&&s.size(1)==G,"groupwise scale shape mismatch");}auto st=at::cuda::getCurrentCUDAStream();int total=N*KH,th=256,bl=std::min(std::max(1,block_cap),(total+th-1)/th);if(gs==0)w4_dequant_kernel<0><<<bl,th,0,st>>>(N,KH,G,(unsigned char*)q.data_ptr<uint8_t>(),(half*)s.data_ptr<at::Half>(),(half*)w.data_ptr<at::Half>());else if(gs==128)w4_dequant_kernel<128><<<bl,th,0,st>>>(N,KH,G,(unsigned char*)q.data_ptr<uint8_t>(),(half*)s.data_ptr<at::Half>(),(half*)w.data_ptr<at::Half>());else w4_dequant_kernel<256><<<bl,th,0,st>>>(N,KH,G,(unsigned char*)q.data_ptr<uint8_t>(),(half*)s.data_ptr<at::Half>(),(half*)w.data_ptr<at::Half>());C10_CUDA_KERNEL_LAUNCH_CHECK();return w;}
torch::Tensor dequant4(torch::Tensor q,torch::Tensor s,int gs,int block_cap){auto w=torch::empty({q.size(0),q.size(1)*2},s.options());return dequant4_out(q,s,w,gs,block_cap);}
}
torch::Tensor rwkv7_sm70_w8_cuda(torch::Tensor x,torch::Tensor q,torch::Tensor s,int64_t th){auto y=torch::empty({x.size(0),q.size(0)},x.options());return run8(x,q,s,y,th);}
torch::Tensor rwkv7_sm70_w8_out_cuda(torch::Tensor x,torch::Tensor q,torch::Tensor s,torch::Tensor y,int64_t th){return run8(x,q,s,y,th);}
torch::Tensor rwkv7_sm70_w8_dequant_out_cuda(torch::Tensor q,torch::Tensor s,torch::Tensor w,int64_t block_cap){TORCH_CHECK(q.is_cuda()&&s.is_cuda()&&w.is_cuda()&&q.scalar_type()==at::kChar&&s.scalar_type()==at::kHalf&&w.scalar_type()==at::kHalf&&q.dim()==2&&q.is_contiguous()&&s.is_contiguous()&&w.is_contiguous(),"sm70 W8 dequant requires contiguous CUDA int8 weights and fp16 scales/output");int N=q.size(0),K=q.size(1);TORCH_CHECK(K%2==0&&s.dim()==1&&s.size(0)==N&&w.dim()==2&&w.size(0)==N&&w.size(1)==K,"sm70 W8 dequant shape mismatch");auto st=at::cuda::getCurrentCUDAStream();int total=N*(K/2),th=256,bl=std::min(std::max(1,(int)block_cap),(total+th-1)/th);w8_dequant_kernel<<<bl,th,0,st>>>(N,K,(signed char*)q.data_ptr<int8_t>(),(half*)s.data_ptr<at::Half>(),(half*)w.data_ptr<at::Half>());C10_CUDA_KERNEL_LAUNCH_CHECK();return w;}
torch::Tensor rwkv7_sm70_w4_cuda(torch::Tensor x,torch::Tensor q,torch::Tensor s,int64_t N,int64_t bn,int64_t tn){auto y=torch::empty({x.size(0),N},x.options());return run4<0>(x,q,s,torch::Tensor(),y,N,bn,tn);}
torch::Tensor rwkv7_sm70_w4_out_cuda(torch::Tensor x,torch::Tensor q,torch::Tensor s,torch::Tensor y,int64_t N,int64_t bn,int64_t tn){return run4<0>(x,q,s,torch::Tensor(),y,N,bn,tn);}
torch::Tensor rwkv7_sm70_w4_relu2_cuda(torch::Tensor x,torch::Tensor q,torch::Tensor s,int64_t N,int64_t bn,int64_t tn){auto y=torch::empty({x.size(0),N},x.options());return run4<1>(x,q,s,torch::Tensor(),y,N,bn,tn);}
torch::Tensor rwkv7_sm70_w4_add_cuda(torch::Tensor x,torch::Tensor q,torch::Tensor s,torch::Tensor residual,int64_t N,int64_t bn,int64_t tn){auto y=torch::empty_like(residual);return run4<2>(x,q,s,residual,y,N,bn,tn);}
torch::Tensor rwkv7_sm70_w4_group_cuda(torch::Tensor x,torch::Tensor q,torch::Tensor s,int64_t N,int64_t gs,int64_t bn,int64_t tn){auto y=torch::empty({x.size(0),N},x.options());return run4g(x,q,s,y,N,gs,bn,tn);}
torch::Tensor rwkv7_sm70_w4_group_out_cuda(torch::Tensor x,torch::Tensor q,torch::Tensor s,torch::Tensor y,int64_t N,int64_t gs,int64_t bn,int64_t tn){return run4g(x,q,s,y,N,gs,bn,tn);}
torch::Tensor rwkv7_sm70_w4_dequant_cuda(torch::Tensor q,torch::Tensor s,int64_t gs,int64_t block_cap){return dequant4(q,s,gs,block_cap);}
torch::Tensor rwkv7_sm70_w4_dequant_out_cuda(torch::Tensor q,torch::Tensor s,torch::Tensor w,int64_t gs,int64_t block_cap){return dequant4_out(q,s,w,gs,block_cap);}
"""

_EXT = None
_ERR = None
_LOCK = threading.Lock()


def _cccl_include_paths() -> list[str]:
    """Find the CUDA C++ core-library headers needed by CUDA 12.4+.

    Some minimal CUDA installations keep ``cuda_fp16.h`` under ``CUDA_HOME``
    but install ``nv/target`` through the Python/Conda CCCL package.  NVCC then
    fails while lazily building the sm_70 extension and inference silently
    falls back to the much slower dequantized implementation.  Pass every
    detected CCCL root explicitly so the production kernel does not depend on
    a shell-specific ``CPATH``.
    """

    candidates: list[Path] = []
    explicit = os.environ.get("CUDA_CCCL_INCLUDE_PATH", "")
    candidates.extend(Path(item).expanduser() for item in explicit.split(os.pathsep) if item)
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidates.append(Path(cuda_home) / "include")
    candidates.extend(
        [
            Path(sys.prefix) / "include",
            Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages" / "nvidia" / "cuda_cccl" / "include",
            Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages" / "nvidia" / "cuda_runtime" / "include",
        ]
    )
    found: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = str(candidate.resolve())
        except OSError:
            resolved = str(candidate)
        if resolved in seen or not (candidate / "nv" / "target").is_file():
            continue
        seen.add(resolved)
        found.append(resolved)
    return found


SM70_W4_BN_TN_CHOICES = (
    (1, 1),
    (2, 1),
    (4, 1),
    (4, 2),
    (4, 4),
    (8, 1),
    (8, 2),
    (8, 4),
    (16, 1),
    (16, 2),
    (16, 4),
    (32, 1),
    (32, 2),
)
SM70_W4_AUTO_BN_TN = {
    (1, 10240, 2560): (16, 1),
    (1, 16384, 4096): (4, 1),
    (1, 2048, 2048): (8, 2),
    (1, 2048, 65536): (8, 2),
    (1, 2048, 8192): (4, 2),
    (1, 2560, 10240): (4, 1),
    (1, 2560, 2560): (4, 1),
    (1, 2560, 65536): (4, 1),
    (1, 4096, 16384): (16, 2),
    (1, 4096, 4096): (8, 2),
    (1, 4096, 65536): (16, 2),
    (1, 8192, 2048): (4, 1),
    (2, 10240, 2560): (16, 1),
    (2, 16384, 4096): (16, 2),
    (2, 2048, 2048): (16, 1),
    (2, 2048, 65536): (4, 1),
    (2, 2048, 8192): (4, 1),
    (2, 2560, 10240): (4, 1),
    (2, 2560, 2560): (4, 4),
    (2, 2560, 65536): (4, 1),
    (2, 4096, 16384): (4, 1),
    (2, 4096, 4096): (16, 2),
    (2, 4096, 65536): (4, 1),
    (2, 8192, 2048): (8, 1),
    (4, 10240, 2560): (16, 1),
    (4, 16384, 4096): (16, 2),
    (4, 2048, 2048): (4, 2),
    (4, 2048, 65536): (4, 1),
    (4, 2048, 8192): (4, 1),
    (4, 2560, 10240): (4, 1),
    (4, 2560, 2560): (8, 4),
    (4, 2560, 65536): (4, 1),
    (4, 4096, 16384): (4, 1),
    (4, 4096, 4096): (16, 2),
    (4, 4096, 65536): (4, 1),
    (4, 8192, 2048): (8, 1),
    (8, 10240, 2560): (8, 1),
    (8, 16384, 4096): (8, 2),
    (8, 2048, 2048): (16, 1),
    (8, 2048, 65536): (4, 1),
    (8, 2048, 8192): (4, 1),
    (8, 2560, 10240): (4, 1),
    (8, 2560, 2560): (8, 1),
    (8, 2560, 65536): (4, 1),
    (8, 4096, 16384): (16, 1),
    (8, 4096, 4096): (8, 2),
    (8, 4096, 65536): (4, 1),
    (8, 8192, 2048): (16, 1),
}
SM70_W4_GROUP_AUTO_BN_TN = {
    (1, 2048, 65536): (8, 1),
    (1, 2560, 65536): (16, 1),
    (1, 4096, 65536): (16, 1),
    (2, 2048, 65536): (8, 1),
    (2, 2560, 65536): (8, 1),
    (2, 4096, 65536): (4, 1),
    (4, 2048, 65536): (8, 1),
    (4, 2560, 65536): (8, 1),
    (4, 4096, 65536): (8, 1),
    (8, 2048, 65536): (8, 1),
    (8, 2560, 65536): (8, 1),
    (8, 4096, 65536): (8, 1),
}
SM70_W4_GROUP256_AUTO_BN_TN = {
    (1, 2560, 65536): (32, 1),
    (2, 2560, 65536): (8, 1),
    (4, 2560, 65536): (8, 1),
    (8, 2560, 65536): (32, 1),
}


def is_sm70(device=None):
    if torch is None or not torch.cuda.is_available():
        return False
    d = torch.device("cuda" if device is None else device)
    if d.type != "cuda":
        return False
    i = torch.cuda.current_device() if d.index is None else d.index
    return tuple(torch.cuda.get_device_capability(i)) == (7, 0)


def sm70_w4_bn_tn_config(
    rows: int | None = None,
    in_features: int | None = None,
    out_features: int | None = None,
) -> tuple[int, int]:
    """Resolve the exact-sm70 W4 output tile."""

    explicit = "RWKV7_SM70_W4_BN" in os.environ or "RWKV7_SM70_W4_TN" in os.environ
    if explicit:
        bn = int(os.environ.get("RWKV7_SM70_W4_BN", "8"))
        tn = int(os.environ.get("RWKV7_SM70_W4_TN", "1"))
    else:
        key = (int(rows or 0), int(in_features or 0), int(out_features or 0))
        bn, tn = SM70_W4_AUTO_BN_TN.get(key, (8, 1))
    if (bn, tn) not in SM70_W4_BN_TN_CHOICES:
        raise ValueError(
            f"unsupported sm70 W4 BN/TN pair {(bn, tn)}; "
            f"expected one of {SM70_W4_BN_TN_CHOICES}"
        )
    return bn, tn


def sm70_w4_group_bn_tn_config(
    rows: int | None = None,
    in_features: int | None = None,
    out_features: int | None = None,
    group_size: int = 128,
) -> tuple[int, int]:
    """Resolve the independent exact-sm70 groupwise output tile."""

    explicit = (
        "RWKV7_SM70_W4_GROUP_BN" in os.environ
        or "RWKV7_SM70_W4_GROUP_TN" in os.environ
    )
    if explicit:
        bn = int(os.environ.get("RWKV7_SM70_W4_GROUP_BN", "8"))
        tn = int(os.environ.get("RWKV7_SM70_W4_GROUP_TN", "1"))
    else:
        key = (int(rows or 0), int(in_features or 0), int(out_features or 0))
        table = (
            SM70_W4_GROUP256_AUTO_BN_TN
            if int(group_size) == 256
            else SM70_W4_GROUP_AUTO_BN_TN
        )
        bn, tn = table.get(key, (8, 1))
    if (bn, tn) not in SM70_W4_BN_TN_CHOICES:
        raise ValueError(
            f"unsupported sm70 groupwise W4 BN/TN pair {(bn, tn)}; "
            f"expected one of {SM70_W4_BN_TN_CHOICES}"
        )
    return bn, tn


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
                name="rwkv7_sm70_quant_v25",
                cpp_sources=_CPP,
                cuda_sources=_CUDA,
                functions=None,
                extra_cflags=["-O3"],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "--extra-device-vectorization",
                ],
                extra_include_paths=_cccl_include_paths(),
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


def quantize_w4_groupwise(weight, group_size=128):
    """Symmetric row-major W4 with one fp16 scale per output/group."""

    group_size = int(group_size)
    if group_size not in {128, 256}:
        raise ValueError("sm70 groupwise W4 requires group_size=128 or 256")
    if weight.dim() != 2 or weight.shape[1] % group_size:
        raise ValueError(
            "groupwise W4 requires rank-2 weight with K divisible by group_size"
        )
    w = weight.detach().float()
    out_features, in_features = w.shape
    grouped = w.reshape(out_features, in_features // group_size, group_size)
    scales = (grouped.abs().amax(dim=2) / 7).clamp_min(1e-6)
    q = (grouped / scales[:, :, None]).round().clamp(-7, 7).to(torch.int16) + 8
    q = q.reshape(out_features, in_features)
    packed = ((q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8)).contiguous()
    return packed, scales.to(weight.dtype).contiguous(), in_features


def _w4_prefill_dense_min_rows() -> int:
    """Threshold for transient fused-dequant + cuBLAS prefill.

    The persistent model stays packed. Only the active projection is expanded
    into a caching-allocator workspace, consumed by tensor-core GEMM, and then
    released. This avoids the decode kernel's eight-row weight-reuse ceiling.
    """

    try:
        return max(0, int(os.environ.get("RWKV7_SM70_W4_PREFILL_DENSE_MIN_ROWS", "16")))
    except ValueError:
        return 16


def _w4_use_transient_dense(x2) -> bool:
    threshold = _w4_prefill_dense_min_rows()
    return bool(
        threshold > 0
        and int(x2.shape[0]) >= threshold
        and x2.is_cuda
        and x2.dtype == torch.float16
        and is_sm70(x2.device)
    )


def _w4_transient_dense_linear(x, q, scales, out_features, group_size=0):
    extension = _load()
    if extension is None:
        return None
    weight = extension.w4_dequant(
        q, scales, int(group_size), _w4_dequant_block_cap()
    )
    return F.linear(x, weight)


def _w4_dequant_block_cap() -> int:
    try:
        # Eight blocks per V100 SM saturated both FFN matrix orientations in
        # the exact-sm70 sweep; larger grids only added scheduling overhead.
        return max(1, int(os.environ.get("RWKV7_SM70_W4_DEQUANT_BLOCKS", "640")))
    except ValueError:
        return 640


def w4_prefill_dequant_weight(q, scales, rows, group_size=0, out=None):
    """Return a transient fp16 weight when the exact-sm70 prefill gate is on."""

    threshold = _w4_prefill_dense_min_rows()
    if (
        threshold <= 0
        or int(rows) < threshold
        or not q.is_cuda
        or scales.dtype != torch.float16
        or not is_sm70(q.device)
    ):
        return None
    extension = _load()
    if extension is None:
        return None
    if out is not None:
        return extension.w4_dequant_out(
            q, scales, out, int(group_size), _w4_dequant_block_cap()
        )
    return extension.w4_dequant(
        q, scales, int(group_size), _w4_dequant_block_cap()
    )


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


def w8_prefill_dequant_weight(q, scales, rows, out=None):
    """Expand one symmetric W8 operand for a tensor-core prefill GEMM.

    The packed weight remains the persistent model representation.  Native
    prefill owns a bounded reusable fp16 workspace and overlaps this single
    conversion kernel with the attention half of the same RWKV layer.
    """

    try:
        threshold = max(
            0,
            int(os.environ.get("RWKV7_SM70_W8_PREFILL_DENSE_MIN_ROWS", "16")),
        )
    except ValueError:
        threshold = 16
    if (
        threshold <= 0
        or int(rows) < threshold
        or not q.is_cuda
        or q.dtype != torch.int8
        or scales.dtype != torch.float16
        or not is_sm70(q.device)
    ):
        return None
    extension = _load()
    if extension is None:
        return None
    expected = (int(q.shape[0]), int(q.shape[1]))
    if out is None:
        out = torch.empty(expected, device=q.device, dtype=scales.dtype)
    elif tuple(out.shape) != expected:
        raise ValueError(
            f"sm70 W8 prefill workspace must have shape {expected}, got {tuple(out.shape)}"
        )
    return extension.w8_dequant_out(q, scales, out, _w4_dequant_block_cap())


def w4_linear(x, q, s, out_features, in_features, out=None):
    scalar = x.dim() == 1
    x2 = x.reshape(-1, x.shape[-1])
    if int(q.shape[1]) * 2 == int(in_features) and _w4_use_transient_dense(x2):
        result = _w4_transient_dense_linear(x, q, s, out_features, 0)
        if result is not None:
            if out is not None:
                out.copy_(result)
                return out
            return result
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
    bn, tn = sm70_w4_bn_tn_config(
        int(x2.shape[0]), int(in_features), int(out_features)
    )
    y = (
        e.w4(x2.contiguous(), q, s, int(out_features), bn, tn)
        if out is None
        else e.w4_out(
            x2.contiguous(),
            q,
            s,
            out.reshape(x2.shape[0], out_features),
            int(out_features),
            bn,
            tn,
        )
    )
    return y.reshape(out_features) if scalar else y.reshape(*x.shape[:-1], out_features)


def w4_groupwise_linear(
    x,
    q,
    scales,
    out_features,
    in_features,
    *,
    group_size=128,
    out=None,
):
    """Apply exact-sm70 groupwise W4, with a deterministic torch fallback."""

    group_size = int(group_size)
    if group_size not in {128, 256} or int(in_features) % group_size:
        raise ValueError(
            "sm70 groupwise W4 requires group_size=128/256 and divisible K"
        )
    scalar = x.dim() == 1
    x2 = x.reshape(-1, x.shape[-1])
    if _w4_use_transient_dense(x2):
        result = _w4_transient_dense_linear(
            x, q, scales, out_features, group_size
        )
        if result is not None:
            if out is not None:
                out.copy_(result)
                return out
            return result
    e = _load() if x2.is_cuda and is_sm70(x2.device) else None
    if e is None:
        lo = (q & 15).to(x.dtype) - 8
        hi = (q >> 4).to(x.dtype) - 8
        w = torch.empty(out_features, int(in_features), device=q.device, dtype=x.dtype)
        w[:, 0::2] = lo
        w[:, 1::2] = hi
        expanded_scales = scales.to(x.dtype).repeat_interleave(group_size, dim=1)
        result = F.linear(x, w * expanded_scales)
        if out is not None:
            out.copy_(result)
            return out
        return result
    bn, tn = sm70_w4_group_bn_tn_config(
        int(x2.shape[0]),
        int(in_features),
        int(out_features),
        group_size=group_size,
    )
    y = (
        e.w4_group(
            x2.contiguous(),
            q,
            scales,
            int(out_features),
            group_size,
            bn,
            tn,
        )
        if out is None
        else e.w4_group_out(
            x2.contiguous(),
            q,
            scales,
            out.reshape(x2.shape[0], out_features),
            int(out_features),
            group_size,
            bn,
            tn,
        )
    )
    return y.reshape(out_features) if scalar else y.reshape(*x.shape[:-1], out_features)


def w4_linear_relu2(x, q, s, out_features, in_features):
    """Apply rowwise W4 and the FFN ReLU-squared epilogue on exact sm70."""

    scalar = x.dim() == 1
    x2 = x.reshape(-1, x.shape[-1])
    e = _load() if x2.is_cuda and is_sm70(x2.device) else None
    if e is None:
        return torch.relu(w4_linear(x, q, s, out_features, in_features)) ** 2
    bn, tn = sm70_w4_bn_tn_config(
        int(x2.shape[0]), int(in_features), int(out_features)
    )
    y = e.w4_relu2(x2.contiguous(), q, s, int(out_features), bn, tn)
    return y.reshape(out_features) if scalar else y.reshape(*x.shape[:-1], out_features)


def w4_linear_add(x, q, s, residual, out_features, in_features):
    """Apply rowwise W4 and the residual-add epilogue on exact sm70."""

    scalar = x.dim() == 1
    x2 = x.reshape(-1, x.shape[-1])
    residual2 = residual.reshape(-1, int(out_features)).contiguous()
    e = _load() if x2.is_cuda and is_sm70(x2.device) else None
    if e is None:
        return w4_linear(x, q, s, out_features, in_features) + residual
    bn, tn = sm70_w4_bn_tn_config(
        int(x2.shape[0]), int(in_features), int(out_features)
    )
    y = e.w4_add(
        x2.contiguous(), q, s, residual2, int(out_features), bn, tn
    )
    return y.reshape(out_features) if scalar else y.reshape(*residual.shape)
