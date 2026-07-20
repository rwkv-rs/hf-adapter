#!/usr/bin/env python3
from __future__ import annotations
import json, platform, subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def rows(path: str):
    p = ROOT / path
    if not p.exists(): return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

def r6(x): return round(float(x), 6)

core = rows('results_t4.jsonl')
hf = rows('albatross/results_hf_fixed_token.jsonl')
alb = rows('albatross/results_albatross_fp32io16.jsonl')
q_speed = rows('quant/results_quant_speed.jsonl')
q_full = rows('quant/results_quant_full_model.jsonl')
training = rows('training/results_training.jsonl')
train_status = rows('training/status.jsonl')
official = rows('training/results_official_alignment.jsonl')

models = ['0.1b','0.4b','1.5b','2.9b']
decode = []
for h in hf:
    a = next((x for x in alb if x['model_size_label']==h['model_size_label'] and x['batch_size']==h['batch_size'] and x['tokens']==1), None)
    if not a: continue
    value = float(h['api_decode_tokps_total'])
    decode.append({'model_size_label':h['model_size_label'],'batch_size':h['batch_size'],
                   'hf_native_graph_tokps':value,'albatross_tokps':float(a['tokps']),
                   'ratio_vs_albatross':r6(value/float(a['tokps'])),
                   'graph_cache_hit_rate':h.get('native_graph_cache_hit_rate')})

prefill=[]
for model in models:
    h = next((x for x in core if x.get('axis')=='native_prefill_scan' and x.get('model_size_label')==model and x.get('batch_size')==1 and x.get('prompt_tokens')==512 and x.get('fused_scan_requested') is True),None)
    a = next((x for x in alb if x['model_size_label']==model and x['batch_size']==1 and x['tokens']==512),None)
    if h and a:
        value=float(h['native_prefill_tokps_total'])
        prefill.append({'model_size_label':model,'batch_size':1,'prompt_tokens':512,
                        'hf_fused_prefill_tokps':value,'albatross_tokps':float(a['tokps']),
                        'ratio_vs_albatross':r6(value/float(a['tokps'])),
                        'fused_scan_effective':h.get('fused_scan_effective'),
                        'greedy_match':h.get('greedy_match')})

def quant_summary(data):
    out={}
    for quant in ('mm8','mm4'):
        part=[x for x in data if x.get('quantization')==quant]
        if not part: continue
        out[quant]={
          'rows':len(part),
          'models':sorted({x['model_size_label'] for x in part}),
          'footprint_ratio_min':min(float(x['footprint_ratio_vs_fp16']) for x in part),
          'footprint_ratio_max':max(float(x['footprint_ratio_vs_fp16']) for x in part),
          'prefill_speed_ratio_min':min(float(x['prefill_speed_ratio_vs_fp16']) for x in part),
          'prefill_speed_ratio_max':max(float(x['prefill_speed_ratio_vs_fp16']) for x in part),
          'decode_speed_ratio_min':min(float(x['decode_speed_ratio_vs_fp16']) for x in part),
          'decode_speed_ratio_max':max(float(x['decode_speed_ratio_vs_fp16']) for x in part),
          'prompt_cosine_min':min(float(x['prompt_logits_cos_vs_fp16']) for x in part),
          'final_cosine_min':min(float(x['final_logits_cos_vs_fp16']) for x in part),
          'same_greedy_rows':sum(bool(x['same_greedy_tokens_as_fp16']) for x in part),
          'all_rows':len(part),
          'replaced_modules_min':min(int(x['replaced_modules']) for x in part),
          'replaced_modules_max':max(int(x['replaced_modules']) for x in part),
        }
    return out

status_counts=Counter(x.get('status') for x in train_status)
axis_counts=Counter(x.get('axis') for x in core)
summary={
 'schema_version':1,
 'scope':'Tesla T4 exact-card HF adapter validation',
 'result':'validated_not_production_close',
 'dense_performance_gate':'partial_albatross_gap_open',
 'quant_performance_gate':'partial_two_lane',
 'device':'Tesla T4', 'compute_capability':'7.5','dtype':'fp16',
 'models':models,
 'core_rows':len(core),'core_axis_counts':dict(sorted(axis_counts.items())),
 'dense_decode_vs_albatross':decode,
 'dense_decode_ratio_min':min(x['ratio_vs_albatross'] for x in decode),
 'dense_decode_ratio_max':max(x['ratio_vs_albatross'] for x in decode),
 'prefill_512_b1_vs_albatross':prefill,
 'prefill_ratio_min':min(x['ratio_vs_albatross'] for x in prefill),
 'prefill_ratio_max':max(x['ratio_vs_albatross'] for x in prefill),
 'native_graph_cache_hit_rate_min':min(float(x['graph_cache_hit_rate']) for x in decode if x.get('graph_cache_hit_rate') is not None),
 'quant_speed_head_lane':quant_summary(q_speed),
 'quant_full_model_lane':quant_summary(q_full),
 'training':{
   'result_rows':len(training),'status_records':len(train_status),
   'raw_status_counts':dict(status_counts),
   'superseding_repairs':[x for x in train_status if x.get('supersedes')],
   'effective_result':'pass_for_declared_single_gpu_smoke_matrix',
   'official_alignment_rows':official,
   'official_2.9b':'not_run_host_ram_boundary',
   'train_temp_cuda':'unsupported_on_sm75_requires_bf16_sm80_plus',
   'zero_scope':'single_gpu_integration_and_resume_not_multi_gpu_sharding',
 },
 'boundaries':[
   f'Dense HF native_graph decode is {min(x["ratio_vs_albatross"] for x in decode):.4f}x-{max(x["ratio_vs_albatross"] for x in decode):.4f}x Albatross on measured B1/B2/B4/B8 cells.',
   f'Fused B1/T512 prefill is {min(x["ratio_vs_albatross"] for x in prefill):.4f}x-{max(x["ratio_vs_albatross"] for x in prefill):.4f}x Albatross.',
   'Head-only W8/W4 speed lane is fp16-or-faster for decode but saves only the selected output projection.',
   'Full-model W8/W4 substantially reduces footprint and wins B1 decode, but does not preserve fp16 speed for every larger batch or prefill.',
   'RTX 20-series does not inherit exact-T4 quant or prefill defaults without exact-card evidence.',
 ]
}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')

lines=['# Tesla T4 exact-card HF validation (2026-07-20)','',
'**Result: VALIDATED, not production-close.** Functional HF, cache, fused prefill, native-graph decode, quantized inference and the declared single-GPU training integration matrix pass. Dense parity with Albatross and full-model all-phase quant speed remain open.','',
'## Environment','',
'- GPU: Tesla T4 15 GiB (`sm_75`), application clocks 1590/5001 MHz during timing.',
'- Software: Ubuntu 22.04, PyTorch 2.7.1+cu126, Transformers 5.12.1, Triton 3.3.1, bitsandbytes 0.49.2, PEFT 0.19.1, TRL 1.6.0, FLA 0.5.0, DeepSpeed 0.17.6.',
'- Candidate commit at measurement start: `58cfc2fcc4720e8f807050d12bb06259550bb6e0`; this artifact is attached to the dirty T4 adaptation branch and the final PR commit records the exact source.',
'- Albatross: commit `ee3308f6922e59f2166c7fac3c5a192340a2b48e`, `faster3a_2605`, `fp32io16` WKV, GPU embedding.','',
'## Dense same-GPU comparison','',
'### Cached decode (`tok/s`, fixed token)','',
'| Model | B | HF native_graph | Albatross | HF / Albatross |','|---|---:|---:|---:|---:|']
for x in decode:
    lines.append(f"| {x['model_size_label']} | {x['batch_size']} | {x['hf_native_graph_tokps']:.1f} | {x['albatross_tokps']:.1f} | {x['ratio_vs_albatross']:.4f}x |")
lines += ['',f"Measured ratio range: **{summary['dense_decode_ratio_min']:.4f}x–{summary['dense_decode_ratio_max']:.4f}x**. Minimum native-graph cache hit rate: **{summary['native_graph_cache_hit_rate_min']:.4f}**.",'','### Prefill B1/T512 (`tok/s`)','',
'| Model | HF fused scan | Albatross | HF / Albatross |','|---|---:|---:|---:|']
for x in prefill:
    lines.append(f"| {x['model_size_label']} | {x['hf_fused_prefill_tokps']:.1f} | {x['albatross_tokps']:.1f} | {x['ratio_vs_albatross']:.4f}x |")
lines += ['',f"Measured ratio range: **{summary['prefill_ratio_min']:.4f}x–{summary['prefill_ratio_max']:.4f}x**. These rows use the effective T4 fused native scan and preserve greedy output.",'','## Quantization','',
'| Lane | Quant | Rows | Footprint ratio | Prefill ratio | Decode ratio | Min final cosine | Greedy |','|---|---|---:|---:|---:|---:|---:|---:|']
for label,key in [('head speed','quant_speed_head_lane'),('full model','quant_full_model_lane')]:
  for q,v in summary[key].items():
    lines.append(f"| {label} | {q.upper()} | {v['rows']} | {v['footprint_ratio_min']:.4f}–{v['footprint_ratio_max']:.4f} | {v['prefill_speed_ratio_min']:.4f}–{v['prefill_speed_ratio_max']:.4f} | {v['decode_speed_ratio_min']:.4f}–{v['decode_speed_ratio_max']:.4f} | {v['final_cosine_min']:.7f} | {v['same_greedy_rows']}/{v['all_rows']} |")
lines += ['','The speed lane replaces only `lm_head`; it closes decode speed/correctness with a smaller memory saving. The full-model lane closes memory and B1 decode, but full-model prefill and small-model B4/B8 decode remain below fp16. Therefore this artifact does **not** claim universal T4 W8/W4 performance closure.','',
'## HF and training integration','',
'- All four checkpoints pass load/generate, standard HF API, batch cache, dynamic select/reorder/drop, chunked prefill handoff and native-graph decode checks.',
'- Trainer + LoRA, TRL SFT/DPO/GRPO pass for 0.1B/0.4B/1.5B/2.9B in the declared T4 memory shapes.',
'- PEFT save/reload is exact. FP16 merge/unmerge for 1.5B/2.9B uses the measured `max_abs <= 0.2` gate and preserves greedy tokens.',
'- Trainer resume passes 0.1B/0.4B. Single-GPU ZeRO-2/3 train and resume pass on 0.1B; this proves integration/checkpointing, not multi-GPU sharding.',
'- Official `.pth` CPU-FP32 vs HF CUDA-FP16 alignment passes 0.1B/0.4B/1.5B (top-5 >= 0.96, cosine >= 0.999997, greedy 64/64, 64/64, 32/32). 2.9B was not run because the 15 GiB host-RAM boundary cannot hold the official CPU-FP32 reference safely.',
'- Official `train_temp` CUDA exact-training alignment is not a T4 claim: that path requires BF16 and `sm_80+`.','',
'## Evidence map','',
'- `results_t4.jsonl`: 123 curated dense/cache/prefill/fused rows.',
'- `albatross/`: same-GPU HF fixed-token rows and official Albatross logs/JSONL.',
'- `quant/results_quant_speed.jsonl`: 26 head-speed rows.',
'- `quant/results_quant_full_model.jsonl`: 26 broad-memory rows.',
'- `training/`: Trainer/PEFT/TRL/ZeRO/resume and official-alignment rows/logs.',
'- `summary.json`: machine-readable promoted summary; `SHA256SUMS`: artifact integrity.','',
'## Promotion boundary','',
'Tesla T4 receives exact-card compatibility defaults and measured DP4A W8/W4 routing. RTX 2080/other `sm_75` products remain fail-closed. T4 is **Validated** in the hardware matrix; promotion to **Production-close** requires closing the dense Albatross gap and full-model W8/W4 prefill plus all measured decode batches.','']
(ROOT/'README.md').write_text('\n'.join(lines))
print(json.dumps({'summary':str(ROOT/'summary.json'),'readme':str(ROOT/'README.md'),'core_rows':len(core),'decode_rows':len(decode)},indent=2))
