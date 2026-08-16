# RTX 3090 strict RWKV/Qwen Prefill+Decode v2

Status: **PASS**

- raw_prefill_ratio: min `1.574925x`, median `2.182651x`, pass `48/48`
- adjusted_prefill_ratio: min `1.208324x`, median `1.535161x`, pass `48/48`
- raw_decode_ratio: min `1.253926x`, median `1.740275x`, pass `48/48`
- adjusted_decode_ratio: min `1.017763x`, median `1.207730x`, pass `48/48`

All gates use unrounded raw throughput and require strict `> 1.0`.
