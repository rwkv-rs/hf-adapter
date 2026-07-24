#!/usr/bin/env python3
import json, time
from pathlib import Path
import torch
from rwkv7_hf import enable_ascend
from rwkv7_hf.ascend_quant import AscendW8A16Linear, ascend_w8a16_decision

enable_ascend('npu:0', backend='eager')
torch.manual_seed(20260724)
out_path=Path(__file__).with_name('w8_layer_integration.jsonl')
out_path.unlink(missing_ok=True)

def sync(): torch.npu.synchronize()
def timed(fn, iters=100):
    for _ in range(30): fn()
    sync(); start=time.perf_counter()
    for _ in range(iters): fn()
    sync(); return (time.perf_counter()-start)*1000/iters

for role,k,n in [('ffn.key',4096,16384),('ffn.value',16384,4096)]:
    dense=torch.nn.Linear(k,n,bias=False,device='npu:0',dtype=torch.float16).eval()
    with torch.no_grad(): dense.weight.normal_(std=k**-0.5)
    quant=AscendW8A16Linear.from_float(dense,chunk_rows=512).eval()
    assert quant.packed_bytes < quant.dense_fp16_bytes
    for rows in (1,8,64):
        x=torch.randn(rows,k,device='npu:0',dtype=torch.float16)
        with torch.inference_mode():
            ref=dense(x); out=quant(x); sync()
            fp_ms=timed(lambda:dense(x)); w8_ms=timed(lambda:quant(x))
        decision=ascend_w8a16_decision('model.layers.0.'+role,k,n,device_name=torch.npu.get_device_name(0),rows=rows)
        cosine=torch.nn.functional.cosine_similarity(ref.float().flatten(),out.float().flatten(),dim=0).item()
        speedup=fp_ms/w8_ms
        payload_ratio=quant.packed_bytes/quant.dense_fp16_bytes
        row={
          'axis':'hf_ascend_w8a16_linear','status':'measured','evidence_scope':'isolated_linear',
          'device':torch.npu.get_device_name(0),
          'role':role,'M':rows,'K':k,'N':n,'fp16_ms':fp_ms,'w8a16_ms':w8_ms,
          'speedup':speedup,'cosine':cosine,
          'packed_bytes':quant.packed_bytes,'dense_fp16_bytes':quant.dense_fp16_bytes,
          'payload_ratio':payload_ratio,
          'policy_enabled':decision.enabled,'policy_speed_validated':decision.speed_validated,
          'operator_quality_gate_pass':cosine>=0.999,
          'operator_speed_gate_pass':speedup>=1.0,
          'operator_memory_gate_pass':payload_ratio<1.0,
          'production_gate_pass':False,
        }
        print(json.dumps(row)); out_path.open('a').write(json.dumps(row)+'\n')
    del quant,dense
    torch.npu.empty_cache()
