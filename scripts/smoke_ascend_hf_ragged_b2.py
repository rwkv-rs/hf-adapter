#!/usr/bin/env python3
"""Real 7.2B B2 ragged/cache/chunk parity gate on canonical HF/Ascend."""
from __future__ import annotations
import argparse,hashlib,json,platform,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,__version__ as transformers_version
from rwkv7_hf import enable_ascend
from rwkv7_hf.ascend_reference_oracle import DEFAULT_ACCEPTANCE_THRESHOLDS

def args():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference-json',type=Path,required=True);p.add_argument('--output',type=Path,required=True);return p.parse_args()
def cosine(a,b):
 a=a.detach().float().reshape(-1);b=b.detach().float().reshape(-1)
 if float(a.norm())==0 and float(b.norm())==0:return 1.0
 return float(F.cosine_similarity(a,b,dim=0).cpu())
def nrmse(a,b):
 a=a.detach().float();b=b.detach().float();return float(((a-b).square().mean().sqrt()/a.square().mean().sqrt().clamp_min(1e-12)).cpu())
def cache_metrics(batch_cache,row,compact_cache):
 bs,bxpa,bxpf,bvf=tuple(batch_cache);cs,cxpa,cxpf,cvf=tuple(compact_cache);pairs=[]
 for i,(a,b) in enumerate(zip(bs,cs)):pairs.append((f'recurrent.{i:02d}',a[row:row+1],b))
 for i,(a,b) in enumerate(zip(bxpa,cxpa)):pairs.append((f'attn_shift.{i:02d}',a[row:row+1],b))
 for i,(a,b) in enumerate(zip(bxpf,cxpf)):pairs.append((f'ffn_shift.{i:02d}',a[row:row+1],b))
 pairs.append(('v_first',bvf[row:row+1],cvf))
 vals=[(name,cosine(ref,cand),nrmse(ref,cand)) for name,cand,ref in pairs]
 worst_cos=min(vals,key=lambda x:x[1]);worst_nrmse=max(vals,key=lambda x:x[2])
 return {'min_cosine':worst_cos[1],'min_cosine_tensor':worst_cos[0],'max_normalized_rmse':worst_nrmse[2],'max_normalized_rmse_tensor':worst_nrmse[0],'tensor_count':len(vals)}
def logits_metrics(batch_logits,row,compact_logits):
 return {'cosine':cosine(compact_logits,batch_logits[row:row+1]),'normalized_rmse':nrmse(compact_logits,batch_logits[row:row+1]),'argmax_exact':bool(torch.equal(batch_logits[row:row+1].argmax(-1).cpu(),compact_logits.argmax(-1).cpu()))}
def main():
 a=args();ref=json.loads(a.reference_json.read_text());thr=DEFAULT_ACCEPTANCE_THRESHOLDS;info=enable_ascend('npu:0',backend='eager');s=time.perf_counter();model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,dtype=torch.bfloat16,low_cpu_mem_usage=True).eval();load=time.perf_counter()-s;s=time.perf_counter();model.to('npu:0');torch.npu.synchronize();move=time.perf_counter()-s;torch.npu.reset_peak_memory_stats()
 ids=torch.tensor([[33155,45,308],[0,0,33155]],device='npu:0');mask=torch.tensor([[1,1,1],[0,0,1]],device='npu:0');compact_ids=[ids[0:1],ids[1:2,-1:]];times={}
 with torch.inference_mode():
  s=time.perf_counter();full=model(input_ids=ids,attention_mask=mask,use_cache=True);torch.npu.synchronize();times['ragged_b2_prefill_s']=time.perf_counter()-s
  full_cache=full.past_key_values.clone();compact=[]
  for row,value in enumerate(compact_ids):
   s=time.perf_counter();out=model(input_ids=value,use_cache=True);torch.npu.synchronize();times[f'compact_b1_row{row}_prefill_s']=time.perf_counter()-s;compact.append(out)
  prefill_logits=[logits_metrics(full.logits[:,-1],row,compact[row].logits[:,-1]) for row in range(2)]
  prefill_state=[cache_metrics(full_cache,row,compact[row].past_key_values) for row in range(2)]
  next_ids=full.logits[:,-1].argmax(-1);s=time.perf_counter();continued=model(input_ids=next_ids[:,None],past_key_values=full_cache.clone(),use_cache=True);torch.npu.synchronize();times['ragged_b2_continuation_s']=time.perf_counter()-s
  compact_cont=[]
  for row in range(2):compact_cont.append(model(input_ids=next_ids[row:row+1,None],past_key_values=compact[row].past_key_values.clone(),use_cache=True))
  torch.npu.synchronize();continuation_logits=[logits_metrics(continued.logits[:,-1],row,compact_cont[row].logits[:,-1]) for row in range(2)];continuation_state=[cache_metrics(continued.past_key_values,row,compact_cont[row].past_key_values) for row in range(2)]
  s=time.perf_counter();chunked=model.rwkv7_prefill_chunks(ids,attention_mask=mask,chunk_size=1,logits_to_keep=1);torch.npu.synchronize();times['ragged_b2_chunk1_s']=time.perf_counter()-s
  chunk_logits={'cosine':cosine(full.logits[:,-1:],chunked.logits),'normalized_rmse':nrmse(full.logits[:,-1:],chunked.logits),'argmax_exact':bool(torch.equal(full.logits[:,-1:].argmax(-1).cpu(),chunked.logits.argmax(-1).cpu()))};chunk_state=[cache_metrics(chunked.past_key_values,row,compact[row].past_key_values) for row in range(2)]
 logrows=prefill_logits+continuation_logits+[chunk_logits];states=prefill_state+continuation_state+chunk_state;gates={'finite':all(torch.isfinite(x).all().item() for x in (full.logits,continued.logits,chunked.logits)),'prefill_compact_logits':all(x['cosine']>=thr['logits_min_cosine'] and x['normalized_rmse']<=thr['logits_max_normalized_rmse'] and x['argmax_exact'] for x in prefill_logits),'prefill_compact_state':all(x['min_cosine']>=thr['state_min_cosine'] and x['max_normalized_rmse']<=thr['state_max_normalized_rmse'] for x in prefill_state),'cache_continuation_logits':all(x['cosine']>=thr['logits_min_cosine'] and x['normalized_rmse']<=thr['logits_max_normalized_rmse'] and x['argmax_exact'] for x in continuation_logits),'cache_continuation_state':all(x['min_cosine']>=thr['state_min_cosine'] and x['max_normalized_rmse']<=thr['state_max_normalized_rmse'] for x in continuation_state),'chunk_split_logits':chunk_logits['cosine']>=thr['logits_min_cosine'] and chunk_logits['normalized_rmse']<=thr['logits_max_normalized_rmse'] and chunk_logits['argmax_exact'],'chunk_split_state':all(x['min_cosine']>=thr['state_min_cosine'] and x['max_normalized_rmse']<=thr['state_max_normalized_rmse'] for x in chunk_state)};gates['overall']=all(gates.values())
 report={'axis':'huawei_ascend_hf_real_7p2b_b2_ragged_gate','status':'pass' if gates['overall'] else 'fail','input_token_ids':ids.cpu().tolist(),'attention_mask':mask.cpu().tolist(),'compact_input_token_ids':[x.cpu().tolist() for x in compact_ids],'continuation_input_token_ids':next_ids.cpu().tolist(),'continuation_output_argmax':continued.logits[:,-1].argmax(-1).cpu().tolist(),'thresholds':thr,'gates':gates,'metrics':{'prefill_logits':prefill_logits,'prefill_state':prefill_state,'continuation_logits':continuation_logits,'continuation_state':continuation_state,'chunk_logits':chunk_logits,'chunk_state':chunk_state,'global_logits_min_cosine':min(x['cosine'] for x in logrows),'global_logits_max_normalized_rmse':max(x['normalized_rmse'] for x in logrows),'global_state_min_cosine':min(x['min_cosine'] for x in states),'global_state_max_normalized_rmse':max(x['max_normalized_rmse'] for x in states)},'runtime':{**info.to_dict(),'python_version':platform.python_version(),'torch_version':torch.__version__,'transformers_version':transformers_version,'dtype':'bfloat16','backend':'eager','load_cpu_s':load,'move_npu_s':move},'timings':times,'memory':{'allocated_bytes':int(torch.npu.memory_allocated()),'reserved_bytes':int(torch.npu.memory_reserved()),'peak_allocated_bytes':int(torch.npu.max_memory_allocated()),'peak_reserved_bytes':int(torch.npu.max_memory_reserved())},'checkpoint_files_sha256':ref['checkpoint_files_sha256'],'fla_source':ref['fla_source'],'command':'scripts/smoke_ascend_hf_ragged_b2.py --model <native-view> --reference-json <reference.json> --output <result.json>','return_code':0 if gates['overall'] else 1}
 canonical=json.dumps(report,sort_keys=True,separators=(',',':')).encode();report['evidence_sha256']=hashlib.sha256(canonical).hexdigest();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':report['status'],'gates':gates,'metrics':report['metrics'],'evidence_sha256':report['evidence_sha256']}));return 0 if gates['overall'] else 1
if __name__=='__main__':sys.exit(main())
