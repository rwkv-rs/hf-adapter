#!/usr/bin/env python3
"""Capture B2 and ragged canonical HF/Ascend cases in one model process."""
from __future__ import annotations
import argparse,json,platform,sys,time
from pathlib import Path
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM,__version__ as transformers_version
from rwkv7_hf import enable_ascend
from rwkv7_hf.ascend_reference_oracle import sha256_file,tensor_map_sha256,tensor_sha256

def args():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference-json',type=Path,required=True);p.add_argument('--output-json',type=Path,required=True);p.add_argument('--output-tensors',type=Path,required=True);return p.parse_args()
def cachemap(cache,prefix,valid):
 recurrent,xpa,xpf,vf=tuple(cache);d={f'{prefix}.v_first':vf.cpu(),f'{prefix}.valid_tokens':valid.cpu().long(),f'{prefix}.processed_width':torch.tensor(int(cache.seen_tokens),dtype=torch.long)}
 for i,v in enumerate(recurrent):d[f'{prefix}.recurrent.{i:02d}']=v.cpu()
 for i,v in enumerate(xpa):d[f'{prefix}.attn_shift.{i:02d}']=v.cpu()
 for i,v in enumerate(xpf):d[f'{prefix}.ffn_shift.{i:02d}']=v.cpu()
 return d
def run(model,prefix,case,tensors,times):
 ids=torch.tensor(case['input_token_ids'],device='npu:0',dtype=torch.long);mask=torch.tensor(case['attention_mask'],device='npu:0',dtype=torch.long);valid=mask.sum(1).cpu().long()
 tensors[f'{prefix}.input.token_ids']=ids.cpu();tensors[f'{prefix}.input.attention_mask']=mask.cpu();start=time.perf_counter();out=model(input_ids=ids,attention_mask=mask,use_cache=True);torch.npu.synchronize();times[f'{prefix}_prefill_s']=time.perf_counter()-start
 tensors[f'{prefix}.prefill.logits']=out.logits.cpu();tensors.update(cachemap(out.past_key_values,f'{prefix}.prefill.state',valid));state=out.past_key_values;logits=out.logits[:,-1];gen=[]
 for i in range(case['decode_steps']):
  token=logits.argmax(-1).long();gen.append(token.cpu());start=time.perf_counter();out=model(input_ids=token[:,None],past_key_values=state,use_cache=True);torch.npu.synchronize();times[f'{prefix}_decode_{i:02d}_s']=time.perf_counter()-start;state=out.past_key_values;logits=out.logits[:,-1];valid+=1
  tensors[f'{prefix}.decode.{i:02d}.input_token_ids']=token[:,None].cpu();tensors[f'{prefix}.decode.{i:02d}.logits']=logits.cpu()
 tensors[f'{prefix}.greedy.token_ids']=torch.stack(gen,1);tensors.update(cachemap(state,f'{prefix}.final.state',valid));return tensors[f'{prefix}.greedy.token_ids'].tolist()
def main():
 a=args();ref=json.loads(a.reference_json.read_text());scenario=ref['scenario'];info=enable_ascend('npu:0',backend='eager');s=time.perf_counter();model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,dtype=torch.bfloat16,low_cpu_mem_usage=True).eval();load=time.perf_counter()-s;s=time.perf_counter();model.to('npu:0');torch.npu.synchronize();move=time.perf_counter()-s;torch.npu.reset_peak_memory_stats();tensors={};times={}
 with torch.inference_mode():
  b2=run(model,'b2',scenario['cases']['b2'],tensors,times);rag=run(model,'ragged',scenario['cases']['ragged'],tensors,times)
 ordered={k:tensors[k].cpu().contiguous() for k in sorted(tensors)};a.output_tensors.parent.mkdir(parents=True,exist_ok=True);save_file(ordered,a.output_tensors,metadata={'format':'rwkv7-ascend-hf-candidate-matrix-v1','dtype':'bf16'})
 report={'format_version':ref['format_version'],'axis':'huawei_ascend_hf_candidate_matrix','status':'candidate_captured','fla_source':ref['fla_source'],'checkpoint_files_sha256':ref['checkpoint_files_sha256'],'tokenizer_files_sha256':ref['tokenizer_files_sha256'],'config_repair':ref['config_repair'],'scenario':scenario,'measurement':{'observed_greedy_token_ids':{'b2':b2,'ragged':rag},'oracle_greedy_exact':b2==scenario['cases']['b2']['oracle_greedy_token_ids'] and rag==scenario['cases']['ragged']['oracle_greedy_token_ids'],'timings':times,'memory':{'allocated_bytes':int(torch.npu.memory_allocated()),'reserved_bytes':int(torch.npu.memory_reserved()),'peak_allocated_bytes':int(torch.npu.max_memory_allocated()),'peak_reserved_bytes':int(torch.npu.max_memory_reserved())}},'runtime':{**info.to_dict(),'python_version':platform.python_version(),'torch_version':torch.__version__,'transformers_version':transformers_version,'dtype':'bfloat16','backend':'eager','load_cpu_s':load,'move_npu_s':move},'capture':{'tensor_file':a.output_tensors.name,'tensor_file_sha256':sha256_file(a.output_tensors),'capture_sha256':tensor_map_sha256(ordered),'tensor_count':len(ordered),'tensor_sha256':{k:tensor_sha256(v) for k,v in ordered.items()}},'command':'scripts/capture_ascend_hf_candidate_matrix.py ...'}
 a.output_json.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':report['status'],'b2':b2,'ragged':rag,'runtime':report['runtime'],'memory':report['measurement']['memory']}));return 0
if __name__=='__main__':sys.exit(main())
