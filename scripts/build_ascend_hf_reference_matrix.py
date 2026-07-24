#!/usr/bin/env python3
"""Build real 7.2B B2 and ragged CPU-oracle captures."""
from __future__ import annotations
import argparse,json,platform,sys
from pathlib import Path
import torch
from safetensors.torch import save_file
from rwkv7_hf.ascend_reference_oracle import (
 DEFAULT_ACCEPTANCE_THRESHOLDS,NaiveRWKV7Oracle,REFERENCE_FORMAT_VERSION,
 RWKV7_7P2_CHECKPOINT_SHA256,RWKV7_7P2_TOKENIZER_SHA256,SafetensorStore,
 sha256_file,state_tensor_map,tensor_map_sha256,tensor_sha256,verify_files,verify_fla_checkout,
)

def args():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--fla-checkout',type=Path,required=True);p.add_argument('--output-json',type=Path,required=True);p.add_argument('--output-tensors',type=Path,required=True);return p.parse_args()

def run(oracle,prefix,ids,mask,steps,tensors):
 ids=torch.tensor(ids,dtype=torch.long);mask=torch.tensor(mask,dtype=torch.long)
 tensors[f'{prefix}.input.token_ids']=ids;tensors[f'{prefix}.input.attention_mask']=mask
 out=oracle.forward(ids,attention_mask=mask);tensors[f'{prefix}.prefill.logits']=out.logits
 tensors.update(state_tensor_map(out.state.clone(),f'{prefix}.prefill.state'))
 state=out.state;logits=out.logits[:,-1];generated=[]
 for i in range(steps):
  token=logits.argmax(-1).long();generated.append(token);out=oracle.forward(token[:,None],state=state);state=out.state;logits=out.logits[:,-1]
  tensors[f'{prefix}.decode.{i:02d}.input_token_ids']=token[:,None];tensors[f'{prefix}.decode.{i:02d}.logits']=logits
 tensors[f'{prefix}.greedy.token_ids']=torch.stack(generated,1)
 tensors.update(state_tensor_map(state.clone(),f'{prefix}.final.state'))
 return tensors[f'{prefix}.greedy.token_ids'].tolist()

def main():
 a=args();fla=verify_fla_checkout(a.fla_checkout);ck=verify_files(a.model,RWKV7_7P2_CHECKPOINT_SHA256);tok=verify_files(a.model,RWKV7_7P2_TOKENIZER_SHA256);tensors={}
 with SafetensorStore(a.model) as store:
  oracle=NaiveRWKV7Oracle(store,dtype=torch.bfloat16)
  b2=run(oracle,'b2',[[33155],[33155]],[[1],[1]],2,tensors)
  rag=run(oracle,'ragged',[[33155,45,308],[0,0,33155]],[[1,1,1],[0,0,1]],1,tensors)
  repair=oracle.config_repair
 ordered={k:tensors[k].cpu().contiguous() for k in sorted(tensors)};a.output_tensors.parent.mkdir(parents=True,exist_ok=True);save_file(ordered,a.output_tensors,metadata={'format':'rwkv7-ascend-independent-reference-matrix-v1','dtype':'bf16'})
 report={'format_version':REFERENCE_FORMAT_VERSION,'axis':'huawei_ascend_hf_independent_cpu_oracle_matrix','status':'reference_generated','reference_backend':'pinned_fla_formula_naive_pytorch_cpu','candidate_adapter_forward_called':False,'fla_source':fla,'checkpoint_files_sha256':ck,'tokenizer_files_sha256':tok,'config_repair':repair,
 'scenario':{'name':'b2_and_ragged_matrix','cases':{'b2':{'input_token_ids':[[33155],[33155]],'attention_mask':[[1],[1]],'decode_steps':2,'oracle_greedy_token_ids':b2},'ragged':{'input_token_ids':[[33155,45,308],[0,0,33155]],'attention_mask':[[1,1,1],[0,0,1]],'decode_steps':1,'oracle_greedy_token_ids':rag}}},'runtime':{'device':'cpu','dtype':'bf16','torch_version':torch.__version__,'python_version':platform.python_version(),'machine':platform.machine()},'acceptance_thresholds':DEFAULT_ACCEPTANCE_THRESHOLDS,
 'capture':{'tensor_file':a.output_tensors.name,'tensor_file_sha256':sha256_file(a.output_tensors),'capture_sha256':tensor_map_sha256(ordered),'tensor_count':len(ordered),'tensor_sha256':{k:tensor_sha256(v) for k,v in ordered.items()}},'pending_npu_gates':['hf_npu_b2_prefill_multi_decode','hf_npu_b2_ragged_prefill_decode']}
 a.output_json.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':report['status'],'b2':b2,'ragged':rag,'capture_sha256':report['capture']['capture_sha256']}));return 0
if __name__=='__main__':sys.exit(main())
