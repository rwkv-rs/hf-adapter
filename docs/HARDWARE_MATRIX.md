# Hardware matrix

Last audited: **2026-08-17**.

| Platform | Current declared scope | Evidence |
|---|---|---|
| Tesla V100 32GB | Native dense/Albatross, paired P/D, FP16 state, MM4 Decode, W4 Prefill, 2-GPU TP | [index](../bench/INDEX.md) |
| Tesla T4 | Native HF/cache/quant/training exact-card validation | [artifact](../bench/t4_production_close_20260720/README.md) |
| RTX 3090 | Native paired P/D and quantization | [paired](../bench/3090_qwen35_paired_pd_v2_20260816/README.md), [quant](../bench/3090_native_quant_20260713/README.md) |
| RTX 4080 | Native paired P/D, B8 projection, 7.2B FP16 state | [paired](../bench/4080_qwen35_paired_pd_v1_20260814/README.md) |
| RTX 4090 | Native paired P/D, exact route tuning, small/large B8 quantization | [paired](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| RTX 5070 Laptop | Native exact-card performance and quant loading | [Native](../bench/5070_max_perf_20260811/README.md) |
| RTX 5090 | Native paired Decode, W4, Native/official inference, training | [paired](../bench/5090_qwen35_paired_decode_v1_20260813/README.md) |
| AMD gfx1100 | Native exact-card and quantization | [close](../bench/amd_gfx1100_full_close_20260730/README.md) |
| Apple M5 | MLX production-close and current B1/B8 paired lines | [B1](../bench/apple_bsz1_active_m5_20260715/README.md), [B8](../bench/apple_bsz8_active_m5_20260714/README.md) |
| Moore Threads S70 | Native compatibility and opt-in shift-mix | [MUSA](hardware/MUSA.md) |
| Huawei Ascend 910B3 | Native eager/JIT, cache, chunked prefill and bounded graph/W8 routes | [Ascend](hardware/HUAWEI_ASCEND.md) |
| Biren BR106M | BF16 Native eager and bounded training compatibility | [Biren](hardware/BIREN_BR106M.md) |
| MetaX C500 | Native eager and bounded Trainer/PEFT scope | [MetaX](hardware/METAX_C500.md) |

Policy coverage is not the same as exact-card validation. Unlisted cards use
the conservative compatibility path until their own evidence is promoted.

FLA remains available for explicit reference/compatibility testing; it is not
the current RWKV performance line.
