import json,time,torch,torch_npu
torch.manual_seed(5);D='npu:0'
def sy():torch.npu.synchronize()
def b(fn,n=300):
 for _ in range(40):fn()
 sy();t=time.perf_counter()
 for _ in range(n):fn()
 sy();return (time.perf_counter()-t)*1000/n
for m,k,n in [(1,768,3072),(8,768,3072),(1,3072,768),(8,3072,768),(1,2048,8192),(8,2048,8192),(1,8192,2048),(8,8192,2048),(1,4096,16384),(8,4096,16384),(64,4096,16384),(1,16384,4096),(8,16384,4096),(64,16384,4096)]:
 try:
  x=torch.randn(m,k,device=D,dtype=torch.float16);w=torch.randn(n,k,device=D,dtype=torch.float16)/k**.5;s=(w.float().abs().amax(1)/127).clamp_min(1e-8).half();q=torch.round(w.float()/s[:,None]).clamp(-127,127).to(torch.int8)
  # transposed-weight mode [K,N], scale [N]
  wt=q.t();
  def fp():return x@w.t()
  def qf():return torch_npu.npu_weight_quant_batchmatmul(x,wt,s)
  ref=fp();out=qf();sy();a=b(fp);c=b(qf)
  speedup=a/c;cos=torch.nn.functional.cosine_similarity(ref.float().flatten(),out.float().flatten(),dim=0).item()
  print(json.dumps(dict(axis='ascend_w8_raw_operator',status='measured',evidence_scope='raw_operator',M=m,K=k,N=n,fp=a,w8=c,speed=speedup,cos=cos,operator_quality_gate_pass=cos>=0.999,operator_speed_gate_pass=speedup>=1.0,production_gate_pass=False)),flush=True)
 except Exception as e:print(json.dumps(dict(axis='ascend_w8_raw_operator',status='measurement_error',evidence_scope='raw_operator',M=m,K=k,N=n,error=repr(e),production_gate_pass=False)),flush=True)
