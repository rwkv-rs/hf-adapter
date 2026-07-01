
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

### 2026-07-01 — 完整 fused kernel 5070 数据(全部 >1.0×!)
全部 fused kernel prototype 在 RTX 5070(sm_120)的 speedup vs current baseline:

| kernel | V100(sm_70) | **5070(sm_120)** | cos | 状态 |
|---|---|---|---|---|
| fused recurrent | 2.79× | **5.71×** | 1.0 | ✅✅ Blackwell 大幅领先 |
| fused WAG LoRA | — | **1.79×** | 1.0 | ✅ |
| fused RKV WAG projection | 0.84× | **1.68×** | 1.0 | ✅ V100 不过、5070 过 |
| fused W/A LoRA | — | **1.33×** | 1.0 | ✅ |
| fused FFN | — | **1.30×** | 1.0 | ✅ |
| fused shift-mix | 0.77× | **1.02×** | 1.0 | ✅ V100 不过、5070 刚过 |
| int8 fused RKV quant | 0.38× | 0.77× | 0.99995 | ⚠️ 还没 ≥1.0× fp16 |
| int4 fused RKV quant | 0.36× | 0.62× | 0.986 | ⚠️ 还没 ≥1.0× fp16 |

**重大发现**:Blackwell sm_120 对 Triton fused kernel **比 V100 sm_70 友好得多**。V100 上 projection(0.84×)和 shift-mix(0.77×)都没过 1.0×,但 5070 上**全部 >1.0×**。recurrent 从 2.79× 飙到 **5.71×**。说明 Blackwell 新架构(tensor core / 寄存器 / Triton codegen)更适合 fused kernel 策略。

**量化(int8/int4)大幅改善**(0.38→0.77, 0.36→0.62)但还没 ≥1.0× fp16,需继续 tensor-core 优化。
