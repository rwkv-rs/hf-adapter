# RTX 4090 strict RWKV/Qwen Prefill+Decode v2

Status: **PASS**

- raw_prefill_ratio: min `1.415206x`, median `2.449410x`, pass `48/48`
- adjusted_prefill_ratio: min `1.148668x`, median `1.695334x`, pass `48/48`
- raw_decode_ratio: min `1.276285x`, median `1.770640x`, pass `48/48`
- adjusted_decode_ratio: min `1.026173x`, median `1.323737x`, pass `48/48`

All gates use unrounded raw throughput and require strict `> 1.0`.
