
### 2026-07-01 — fused kernel 在 5070 全部跑通(最新 main 16dedd6)
最新 main 的 fused kernel prototype 在 RTX 5070(sm_120)结果:

| kernel | 5070 speedup | cos | 备注 |
|---|---|---|---|
| **fused RKV WAG projection** | **1.68×** ✅ | 1.0 | R/K/V + W/A/G LoRA 全融合,**超 baseline** |
| **fused FFN** | **1.30×** ✅ | 0.9999999 | CMix 融合,**超 baseline** |
| int8 fused RKV quant | 0.77× fp16 | 0.99995 | 旧版 0.38×→0.77×(dequant+RKV 深融) |
| int4 fused RKV quant | 0.62× fp16 | 0.986 | 旧版 0.36×→0.62× |

**关键**:projection + FFN **在 Blackwell 上 >1.0×**(V100 上单 projection 才 0.84×,但 5070 上融合 RKV+WAG 后 1.68×——更深融合 + Blackwell 对 Triton 友好)。fused kernel 策略在 sm_120 **验证成功**。

(注:main 新增了 fused_recurrent_update.py / fused_attention_projection.py / fused_ffn.py / fused_lora.py / native_fused.py / native_quant.py 等,convert 脚本只拷 3 个文件——HF 目录需手动拷全部 rwkv7_hf/*.py 否则 trust_remote_code 找不到模块。)
