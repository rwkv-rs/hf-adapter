#!/usr/bin/env python3
import argparse,gc,json,statistics,time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from rwkv7_hf import enable_ascend,quantize_ascend_w4a16_candidate
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();P=args.model;O=Path(args.output)
enable_ascend('npu:0',backend='eager');torch.manual_seed(13);torch.npu.reset_peak_memory_stats();ids=torch.tensor([[1,2,3,4]],device='npu:0')
def load():return AutoModelForCausalLM.from_pretrained(P,trust_remote_code=True,dtype=torch.float16,low_cpu_mem_usage=True).eval().to('npu:0')
def footprint(m):return sum(x.numel()*x.element_size() for x in list(m.parameters())+list(m.buffers()))
def trace(m,steps=8):
 with torch.inference_mode():
  out=m(ids,use_cache=True,logits_to_keep=1);past=out.past_key_values;logits=[];tokens=[]
  for _ in range(steps):
   logits.append(out.logits[:,-1].float().cpu());tok=torch.argmax(out.logits[:,-1:],dim=-1);tokens.append(int(tok.item()));out=m(tok,past_key_values=past,use_cache=True,logits_to_keep=1);past=out.past_key_values
 return logits,tokens
fixed=[torch.tensor([[100+i]],device='npu:0') for i in range(32)]
def once(m):
 with torch.inference_mode():
  out=m(ids,use_cache=True,logits_to_keep=1);past=out.past_key_values;torch.npu.synchronize();t=time.perf_counter()
  for tok in fixed:out=m(tok,past_key_values=past,use_cache=True,logits_to_keep=1);past=out.past_key_values
  torch.npu.synchronize();return time.perf_counter()-t

dense=load();torch.npu.synchronize();dense_bytes=footprint(dense);dense_single_alloc=torch.npu.memory_allocated();candidate=load();torch.npu.synchronize();two_dense_alloc=torch.npu.memory_allocated();t=time.perf_counter();rep=quantize_ascend_w4a16_candidate(candidate,group_size=128,roles=('ffn.value',),require_explicit_candidate=False);torch.npu.synchronize();quantize_s=time.perf_counter()-t;quant_bytes=footprint(candidate);dense_logits,dense_tokens=trace(dense);quant_logits,quant_tokens=trace(candidate)
# warm fastpaths then collect alternating paired groups
once(dense);once(candidate);pairs=[]
for i in range(5):
 if i%2==0:d=once(dense);q=once(candidate)
 else:q=once(candidate);d=once(dense)
 pairs.append({'group':i,'order':'dense-quant' if i%2==0 else 'quant-dense','dense_s':d,'quant_s':q,'speedup':d/q})
dmed=statistics.median(x['dense_s'] for x in pairs);qmed=statistics.median(x['quant_s'] for x in pairs)
cos=[];kl=[];top=[]
for a,b in zip(dense_logits,quant_logits):
 cos.append(F.cosine_similarity(a.flatten(),b.flatten(),dim=0).item());pa=F.softmax(a,dim=-1);logpa=F.log_softmax(a,dim=-1);logpb=F.log_softmax(b,dim=-1);kl.append((pa*(logpa-logpb)).sum().item());ia=set(torch.topk(a,20,dim=-1).indices.flatten().tolist());ib=set(torch.topk(b,20,dim=-1).indices.flatten().tolist());top.append(len(ia&ib)/20)
del dense;gc.collect();torch.npu.empty_cache();torch.npu.synchronize();quant_single_alloc=torch.npu.memory_allocated()
row={'axis':'huawei_ascend_hf_real_7p2b_w4fp16_value_only','status':'measured','model_source':'fla-hub/rwkv7-7.2B-g0a','device':torch.npu.get_device_name(0),'dtype':'float16','backend':'eager_cached_fastpath','policy':'candidate_ffn_value_g128','replaced_count':len(rep),'quantize_s':quantize_s,'dense_model_bytes':dense_bytes,'quant_model_bytes':quant_bytes,'model_payload_ratio':quant_bytes/dense_bytes,'dense_single_npu_allocated_bytes':dense_single_alloc,'quant_single_npu_allocated_bytes':quant_single_alloc,'allocated_ratio':quant_single_alloc/dense_single_alloc,'two_dense_allocated_bytes':two_dense_alloc,'paired':pairs,'dense_decode_median_s':dmed,'quant_decode_median_s':qmed,'decode_speedup':dmed/qmed,'trace_steps':len(cos),'min_logit_cosine':min(cos),'max_kl_divergence':max(kl),'mean_kl_divergence':sum(kl)/len(kl),'min_top20_overlap':min(top),'mean_top20_overlap':sum(top)/len(top),'dense_tokens':dense_tokens,'quant_tokens':quant_tokens,'greedy_equal':dense_tokens==quant_tokens,'npu_peak_allocated_bytes':torch.npu.max_memory_allocated()}
row['quality_gate_pass']=bool(min(cos)>=0.98 and max(kl)<=0.1 and min(top)>=0.8);row['speed_gate_pass']=bool(row['decode_speedup']>=1.0);row['memory_gate_pass']=bool(quant_bytes<dense_bytes and quant_single_alloc<dense_single_alloc);row['status']='pass' if row['quality_gate_pass'] and row['speed_gate_pass'] and row['memory_gate_pass'] else 'fail_gate';assert len(rep)==32;print(json.dumps(row));O.write_text(json.dumps(row,indent=2)+'\n')
