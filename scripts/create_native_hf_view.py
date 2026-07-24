#!/usr/bin/env python3
"""Create a code-refreshed native HF view over an existing safetensors model."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
from safetensors import safe_open
try:
    from scripts.sync_hf_adapter_code import sync_one
except ModuleNotFoundError:
    from sync_hf_adapter_code import sync_one


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('output')
    ap.add_argument('--copy-weights',action='store_true',help='copy instead of symlink safetensors shards')
    args=ap.parse_args();src=Path(args.source).resolve();dst=Path(args.output).resolve()
    if not (src/'config.json').is_file():raise FileNotFoundError(src/'config.json')
    dst.mkdir(parents=True,exist_ok=True)
    for path in src.iterdir():
        if path.name in {'configuration_rwkv7.py','modeling_rwkv7.py','native_model.py'} or path.is_dir():continue
        target=dst/path.name
        if path.suffix=='.safetensors':
            if target.exists() or target.is_symlink():target.unlink()
            shutil.copy2(path,target) if args.copy_weights else target.symlink_to(path)
        else:shutil.copy2(path,target)
    cfg=json.loads((dst/'config.json').read_text())
    index_path=dst/'model.safetensors.index.json'
    if index_path.exists():
        index=json.loads(index_path.read_text());shard=index['weight_map']['model.layers.0.attn.r_k'];weight_file=src/shard
    else:weight_file=next(src.glob('*.safetensors'))
    with safe_open(weight_file,framework='pt',device='cpu') as f:r_k_shape=tuple(f.get_slice('model.layers.0.attn.r_k').get_shape())
    inferred_heads=int(r_k_shape[-2]);source_heads=int(cfg.get('num_heads',inferred_heads))
    if source_heads!=inferred_heads:
        cfg['num_heads']=inferred_heads
        cfg['source_config_repair']={'num_heads':{'source':source_heads,'inferred_from':'model.layers.0.attn.r_k','value':inferred_heads}}
        (dst/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    result=sync_one(dst)
    result.update({'source':str(src),'num_heads':inferred_heads,'source_num_heads':source_heads,'weight_mode':'copy' if args.copy_weights else 'symlink'})
    (dst/'native_view_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    return 0
if __name__=='__main__':raise SystemExit(main())
