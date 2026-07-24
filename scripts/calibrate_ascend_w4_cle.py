#!/usr/bin/env python3
"""Calibrate one explicit RWKV FFN W4 CLE candidate from saved tensors."""
import argparse,json
from pathlib import Path
import torch
import torch.nn as nn
from rwkv7_hf.ascend_w4_cle import calibrate_sqrelu_value_w4

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--key-weight',required=True);ap.add_argument('--value-weight',required=True);ap.add_argument('--calibration-inputs',required=True);ap.add_argument('--output',required=True);ap.add_argument('--group-size',type=int,default=128);ap.add_argument('--alphas',default='0,0.25,0.5,0.75,1');a=ap.parse_args()
 kw=torch.load(a.key_weight,map_location='cpu',weights_only=True);vw=torch.load(a.value_weight,map_location='cpu',weights_only=True);x=torch.load(a.calibration_inputs,map_location='cpu',weights_only=True)
 if isinstance(kw,dict):kw=kw['weight'];
 if isinstance(vw,dict):vw=vw['weight'];
 key=nn.Linear(kw.shape[1],kw.shape[0],bias=False,dtype=kw.dtype);value=nn.Linear(vw.shape[1],vw.shape[0],bias=False,dtype=vw.dtype);key.weight.data.copy_(kw);value.weight.data.copy_(vw)
 result=calibrate_sqrelu_value_w4(key,value,x,group_size=a.group_size,alphas=[float(v) for v in a.alphas.split(',') if v])
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);torch.save({'scale':result.scale,'alpha':result.alpha},out);meta={k:v for k,v in result.__dict__.items() if k!='scale'};out.with_suffix(out.suffix+'.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta));return 0
if __name__=='__main__':raise SystemExit(main())
