import json,time,torch,torch_npu
torch.manual_seed(3); DEV='npu:0'
def sync(): torch.npu.synchronize()
def bench(fn,n):
 for _ in range(30): fn()
 sync(); t=time.perf_counter()
 for _ in range(n): fn()
 sync(); return (time.perf_counter()-t)*1000/n
for m,k,n in [(1,768,768),(8,768,768),(1,2048,2048),(8,2048,2048),(1,4096,4096),(8,4096,4096),(1,4096,16384),(8,4096,16384)]:
 for g in (32,64,128):
  try:
   x=torch.randn(m,k,device=DEV,dtype=torch.float16); w=torch.randn(n,k,device=DEV,dtype=torch.float16)/k**.5
   pad=(-k)%g; wp=torch.nn.functional.pad(w,(0,pad)); kg=wp.shape[1]//g
   wg=wp.reshape(n,kg,g); s=(wg.float().abs().amax(2)/7).clamp_min(1e-8).to(torch.float16)
   q=torch.round(wg.float()/s[:,:,None]).clamp(-8,7).to(torch.int32).reshape(n,-1)[:,:k]
   packed=torch_npu.npu_convert_weight_to_int4pack(q.t().contiguous())
   scale=s.t().contiguous(); off=torch.zeros_like(scale)
   def fp(): return x@w.t()
   def qf(): return torch_npu.npu_weight_quant_batchmatmul(x,packed,scale,off,None,None,None,g,1)
   ref=fp(); out=qf(); sync(); it=300 if n<=4096 else 100
   a=bench(fp,it); b=bench(qf,it)
   cos=torch.nn.functional.cosine_similarity(ref.float().flatten(),out.float().flatten(),dim=0).item();speedup=a/b
   print(json.dumps(dict(axis='ascend_w4_raw_operator',status='measured',evidence_scope='raw_operator',M=m,K=k,N=n,G=g,cos=cos,fp16_ms=a,w4_ms=b,speedup=speedup,operator_quality_gate_pass=cos>=0.999,operator_speed_gate_pass=speedup>=1.0,production_gate_pass=False)),flush=True)
  except Exception as e: print(json.dumps(dict(axis='ascend_w4_raw_operator',status='measurement_error',evidence_scope='raw_operator',M=m,K=k,N=n,G=g,error=repr(e),production_gate_pass=False)),flush=True)
