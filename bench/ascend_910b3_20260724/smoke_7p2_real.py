#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from rwkv7_hf import enable_ascend

ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
model_dir=args.model;out_path=Path(args.output)
info=enable_ascend('npu:0',backend='eager')
torch.npu.reset_peak_memory_stats()
t0=time.perf_counter();cfg=AutoConfig.from_pretrained(model_dir,trust_remote_code=True)
tok=AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(model_dir,trust_remote_code=True,dtype=torch.bfloat16,low_cpu_mem_usage=True).eval()
load_cpu_s=time.perf_counter()-t0
t1=time.perf_counter();model.to('npu:0');torch.npu.synchronize();to_npu_s=time.perf_counter()-t1
ids=torch.tensor([[1,2,3,4]],device='npu:0',dtype=torch.long)
with torch.inference_mode():
 t2=time.perf_counter();full=model(ids,use_cache=True,logits_to_keep=1);torch.npu.synchronize();forward_s=time.perf_counter()-t2
 chunk=model.rwkv7_prefill_chunks(ids,chunk_size=2,logits_to_keep=1)
 cos=torch.nn.functional.cosine_similarity(full.logits.float().flatten(),chunk.logits.float().flatten(),dim=0).item()
 assert torch.isfinite(full.logits).all().detach().cpu().item() and cos>0.9999
 cache=chunk.past_key_values
 step=model(torch.argmax(chunk.logits[:,-1:],dim=-1),past_key_values=cache,use_cache=True)
 assert step.past_key_values.seen_tokens==5 and torch.isfinite(step.logits).all().detach().cpu().item()
 t3=time.perf_counter();generated=model.generate(ids,max_new_tokens=2,do_sample=False,use_cache=True,pad_token_id=0,eos_token_id=None);torch.npu.synchronize();generate_s=time.perf_counter()-t3
gates={
 'finite_forward':bool(torch.isfinite(full.logits).all().detach().cpu().item()),
 'chunked_prefill_parity':cos>0.9999,
 'finite_decode':bool(torch.isfinite(step.logits).all().detach().cpu().item()),
 'cache_advanced':step.past_key_values.seen_tokens==5,
}
row={
 'axis':'huawei_ascend_hf_real_7p2b','status':'pass' if all(gates.values()) else 'fail_gate',
 'evidence_scope':'hf_compatibility_smoke','gates':gates,'model_source':'fla-hub/rwkv7-7.2B-g0a',
 'model_parameters':7199141888,'device':info.device_name,'cann':info.cann_version,
 'dtype':'bfloat16','backend':'eager','auto_config':type(cfg).__name__,'auto_tokenizer':type(tok).__name__,
 'auto_model':type(model).__name__,'load_cpu_s':load_cpu_s,'to_npu_s':to_npu_s,'forward_s':forward_s,
 'prompt_tokens':4,'generated_tokens':2,'generate_s':generate_s,'chunked_prefill_cosine':cos,
 'cache_seen_tokens':step.past_key_values.seen_tokens,'greedy_ids':generated.detach().cpu().tolist(),
 'npu_allocated_bytes':torch.npu.memory_allocated(),'npu_peak_allocated_bytes':torch.npu.max_memory_allocated(),
}
print(json.dumps(row));out_path.write_text(json.dumps(row,indent=2)+'\n')
