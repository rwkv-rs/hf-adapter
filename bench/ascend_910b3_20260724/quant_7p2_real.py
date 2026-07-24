#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from rwkv7_hf import enable_ascend, quantize_ascend_w8a16

ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
P=args.model;O=Path(args.output)
enable_ascend('npu:0',backend='eager');torch.manual_seed(11);torch.npu.reset_peak_memory_stats()
model=AutoModelForCausalLM.from_pretrained(P,trust_remote_code=True,dtype=torch.bfloat16,low_cpu_mem_usage=True).eval().to('npu:0');torch.npu.synchronize()
ids=torch.tensor([[1,2,3,4]],device='npu:0')

def footprint(m):
 return sum(x.numel()*x.element_size() for x in list(m.parameters())+list(m.buffers()))
def trace(steps=8):
 with torch.inference_mode():
  out=model(ids,use_cache=True,logits_to_keep=1);past=out.past_key_values; logits=[];tokens=[]
  for _ in range(steps):
   logits.append(out.logits[:,-1].float().cpu())
   tok=torch.argmax(out.logits[:,-1:],dim=-1);tokens.append(int(tok.item()))
   out=model(tok,past_key_values=past,use_cache=True,logits_to_keep=1);past=out.past_key_values
 return logits,tokens
def decode_bench(steps=32,repeats=3):
 times=[]
 fixed=[torch.tensor([[100+i]],device='npu:0') for i in range(steps)]
 for _ in range(repeats):
  with torch.inference_mode():
   out=model(ids,use_cache=True,logits_to_keep=1);past=out.past_key_values;torch.npu.synchronize();t=time.perf_counter()
   for tok in fixed: out=model(tok,past_key_values=past,use_cache=True,logits_to_keep=1);past=out.past_key_values
   torch.npu.synchronize();times.append(time.perf_counter()-t)
 return sorted(times)[len(times)//2],times

dense_bytes=footprint(model);dense_alloc=torch.npu.memory_allocated();dense_logits,dense_tokens=trace();dense_s,dense_repeats=decode_bench()
t=time.perf_counter();replaced=quantize_ascend_w8a16(model,policy='candidate',strict=True,chunk_rows=512);torch.npu.synchronize();quantize_s=time.perf_counter()-t
torch.npu.empty_cache();quant_bytes=footprint(model);quant_alloc=torch.npu.memory_allocated();quant_logits,quant_tokens=trace();quant_s,quant_repeats=decode_bench()
cos=[torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0).item() for a,b in zip(dense_logits,quant_logits)]
row={
 'axis':'huawei_ascend_hf_real_7p2b_w8bf16','status':'measured','model_source':'fla-hub/rwkv7-7.2B-g0a',
 'device':torch.npu.get_device_name(0),'dtype':'bfloat16','backend':'eager','policy':'speed_ffn_value_only',
 'replaced_count':len(replaced),'replaced':replaced,'quantize_s':quantize_s,
 'dense_model_bytes':dense_bytes,'quant_model_bytes':quant_bytes,'model_payload_ratio':quant_bytes/dense_bytes,
 'dense_npu_allocated_bytes':dense_alloc,'quant_npu_allocated_bytes':quant_alloc,'allocated_ratio':quant_alloc/dense_alloc,
 'dense_decode_s':dense_s,'quant_decode_s':quant_s,'decode_speedup':dense_s/quant_s,
 'dense_decode_repeats_s':dense_repeats,'quant_decode_repeats_s':quant_repeats,
 'trace_steps':len(cos),'min_logit_cosine':min(cos),'dense_tokens':dense_tokens,'quant_tokens':quant_tokens,
 'greedy_equal':dense_tokens==quant_tokens,'npu_peak_allocated_bytes':torch.npu.max_memory_allocated(),
}
assert len(replaced)==32 and quant_bytes<dense_bytes and quant_alloc<dense_alloc
row['quality_gate_pass']=bool(min(cos)>0.999 and row['greedy_equal'])
row['speed_gate_pass']=bool(row['decode_speedup']>=1.0)
row['status']='pass' if row['quality_gate_pass'] and row['speed_gate_pass'] else 'fail_gate'
print(json.dumps(row));O.write_text(json.dumps(row,indent=2)+'\n')
