# Frozen Qwen3.5 vs RWKV paired Decode v1

Status: **PASS**; adjusted Decode: **48/48** cells strictly above 1.0x.

Adjusted ratio = raw RWKV/Qwen Decode ratio * RWKV/Qwen active-parameter ratio.
The gate uses unrounded `decode_tokps_total_raw`; rendered values are display-only.
B8 Decode is aggregate throughput across eight sequences. This is not a continuous E2E route.

| RWKV / Qwen | GPU | B | P | D | RWKV tok/s | Qwen tok/s | Raw | Param | Adjusted | Required RWKV tok/s | Margin tok/s | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.4b / 0.8b | NVIDIA GeForce RTX 5090 | 1 | 128 | 128 | 1,125 | 654 | 1.719153x | 0.599112x | 1.029966x | 1,092 | 32.7 | PASS |
| 0.4b / 0.8b | NVIDIA GeForce RTX 5090 | 1 | 128 | 512 | 1,129 | 586 | 1.926711x | 0.599112x | 1.154316x | 978 | 151 | PASS |
| 0.4b / 0.8b | NVIDIA GeForce RTX 5090 | 1 | 512 | 128 | 1,125 | 584 | 1.924286x | 0.599112x | 1.152863x | 975 | 149 | PASS |
| 0.4b / 0.8b | NVIDIA GeForce RTX 5090 | 1 | 512 | 512 | 1,129 | 534 | 2.115358x | 0.599112x | 1.267337x | 891 | 238 | PASS |
| 0.4b / 0.8b | NVIDIA GeForce RTX 5090 | 1 | 2048 | 128 | 1,123 | 424 | 2.648163x | 0.599112x | 1.586547x | 708 | 415 | PASS |
| 0.4b / 0.8b | NVIDIA GeForce RTX 5090 | 1 | 2048 | 512 | 1,127 | 396 | 2.844787x | 0.599112x | 1.704346x | 661 | 466 | PASS |
| 0.4b / 0.8b | NVIDIA GeForce RTX 5090 | 8 | 128 | 128 | 6,466 | 3,722 | 1.737121x | 0.599112x | 1.040730x | 6,213 | 253 | PASS |
| 0.4b / 0.8b | NVIDIA GeForce RTX 5090 | 8 | 128 | 512 | 6,485 | 3,379 | 1.918986x | 0.599112x | 1.149688x | 5,641 | 844 | PASS |
| 0.4b / 0.8b | NVIDIA GeForce RTX 5090 | 8 | 512 | 128 | 6,462 | 3,371 | 1.916835x | 0.599112x | 1.148399x | 5,627 | 835 | PASS |
| 0.4b / 0.8b | NVIDIA GeForce RTX 5090 | 8 | 512 | 512 | 6,485 | 2,988 | 2.170417x | 0.599112x | 1.300323x | 4,988 | 1,498 | PASS |
| 0.4b / 0.8b | NVIDIA GeForce RTX 5090 | 8 | 2048 | 128 | 6,442 | 2,322 | 2.774894x | 0.599112x | 1.662473x | 3,875 | 2,567 | PASS |
| 0.4b / 0.8b | NVIDIA GeForce RTX 5090 | 8 | 2048 | 512 | 6,465 | 2,142 | 3.017760x | 0.599112x | 1.807977x | 3,576 | 2,889 | PASS |
| 1.5b / 2b | NVIDIA GeForce RTX 5090 | 1 | 128 | 128 | 547 | 352 | 1.554462x | 0.811661x | 1.261697x | 434 | 113 | PASS |
| 1.5b / 2b | NVIDIA GeForce RTX 5090 | 1 | 128 | 512 | 548 | 333 | 1.647438x | 0.811661x | 1.337161x | 410 | 138 | PASS |
| 1.5b / 2b | NVIDIA GeForce RTX 5090 | 1 | 512 | 128 | 547 | 334 | 1.638311x | 0.811661x | 1.329753x | 411 | 136 | PASS |
| 1.5b / 2b | NVIDIA GeForce RTX 5090 | 1 | 512 | 512 | 548 | 317 | 1.727442x | 0.811661x | 1.402098x | 391 | 157 | PASS |
| 1.5b / 2b | NVIDIA GeForce RTX 5090 | 1 | 2048 | 128 | 547 | 272 | 2.011151x | 0.811661x | 1.632373x | 335 | 212 | PASS |
| 1.5b / 2b | NVIDIA GeForce RTX 5090 | 1 | 2048 | 512 | 548 | 261 | 2.102720x | 0.811661x | 1.706697x | 321 | 227 | PASS |
| 1.5b / 2b | NVIDIA GeForce RTX 5090 | 8 | 128 | 128 | 3,106 | 2,261 | 1.373660x | 0.811661x | 1.114947x | 2,786 | 320 | PASS |
| 1.5b / 2b | NVIDIA GeForce RTX 5090 | 8 | 128 | 512 | 3,111 | 2,122 | 1.466037x | 0.811661x | 1.189925x | 2,614 | 497 | PASS |
| 1.5b / 2b | NVIDIA GeForce RTX 5090 | 8 | 512 | 128 | 3,105 | 2,114 | 1.469245x | 0.811661x | 1.192529x | 2,604 | 501 | PASS |
| 1.5b / 2b | NVIDIA GeForce RTX 5090 | 8 | 512 | 512 | 3,111 | 2,003 | 1.552723x | 0.811661x | 1.260285x | 2,468 | 642 | PASS |
| 1.5b / 2b | NVIDIA GeForce RTX 5090 | 8 | 2048 | 128 | 3,102 | 1,650 | 1.879573x | 0.811661x | 1.525577x | 2,033 | 1,069 | PASS |
| 1.5b / 2b | NVIDIA GeForce RTX 5090 | 8 | 2048 | 512 | 3,107 | 1,561 | 1.991094x | 0.811661x | 1.616094x | 1,923 | 1,185 | PASS |
| 2.9b / 4b | NVIDIA GeForce RTX 5090 | 1 | 128 | 128 | 309 | 127 | 2.437146x | 0.700882x | 1.708151x | 181 | 128 | PASS |
| 2.9b / 4b | NVIDIA GeForce RTX 5090 | 1 | 128 | 512 | 309 | 122 | 2.525531x | 0.700882x | 1.770099x | 175 | 135 | PASS |
| 2.9b / 4b | NVIDIA GeForce RTX 5090 | 1 | 512 | 128 | 309 | 123 | 2.520204x | 0.700882x | 1.766366x | 175 | 134 | PASS |
| 2.9b / 4b | NVIDIA GeForce RTX 5090 | 1 | 512 | 512 | 309 | 118 | 2.615947x | 0.700882x | 1.833470x | 169 | 141 | PASS |
| 2.9b / 4b | NVIDIA GeForce RTX 5090 | 1 | 2048 | 128 | 309 | 108 | 2.862294x | 0.700882x | 2.006130x | 154 | 155 | PASS |
| 2.9b / 4b | NVIDIA GeForce RTX 5090 | 1 | 2048 | 512 | 309 | 105 | 2.944646x | 0.700882x | 2.063849x | 150 | 159 | PASS |
| 2.9b / 4b | NVIDIA GeForce RTX 5090 | 8 | 128 | 128 | 1,248 | 796 | 1.568412x | 0.700882x | 1.099272x | 1,136 | 113 | PASS |
| 2.9b / 4b | NVIDIA GeForce RTX 5090 | 8 | 128 | 512 | 1,249 | 752 | 1.659868x | 0.700882x | 1.163372x | 1,074 | 175 | PASS |
| 2.9b / 4b | NVIDIA GeForce RTX 5090 | 8 | 512 | 128 | 1,247 | 751 | 1.659928x | 0.700882x | 1.163413x | 1,072 | 175 | PASS |
| 2.9b / 4b | NVIDIA GeForce RTX 5090 | 8 | 512 | 512 | 1,248 | 711 | 1.755583x | 0.700882x | 1.230457x | 1,015 | 234 | PASS |
| 2.9b / 4b | NVIDIA GeForce RTX 5090 | 8 | 2048 | 128 | 1,248 | 618 | 2.020969x | 0.700882x | 1.416461x | 881 | 367 | PASS |
| 2.9b / 4b | NVIDIA GeForce RTX 5090 | 8 | 2048 | 512 | 1,249 | 591 | 2.111777x | 0.700882x | 1.480107x | 844 | 405 | PASS |
| 7.2b / 9b | NVIDIA GeForce RTX 5090 | 1 | 128 | 128 | 146 | 82.0 | 1.778081x | 0.804032x | 1.429633x | 102 | 43.8 | PASS |
| 7.2b / 9b | NVIDIA GeForce RTX 5090 | 1 | 128 | 512 | 146 | 80.3 | 1.815336x | 0.804032x | 1.459588x | 99.9 | 45.9 | PASS |
| 7.2b / 9b | NVIDIA GeForce RTX 5090 | 1 | 512 | 128 | 146 | 80.1 | 1.818026x | 0.804032x | 1.461751x | 99.7 | 46.0 | PASS |
| 7.2b / 9b | NVIDIA GeForce RTX 5090 | 1 | 512 | 512 | 146 | 78.2 | 1.864888x | 0.804032x | 1.499429x | 97.2 | 48.6 | PASS |
| 7.2b / 9b | NVIDIA GeForce RTX 5090 | 1 | 2048 | 128 | 146 | 73.7 | 1.977385x | 0.804032x | 1.589880x | 91.6 | 54.1 | PASS |
| 7.2b / 9b | NVIDIA GeForce RTX 5090 | 1 | 2048 | 512 | 146 | 72.3 | 2.017878x | 0.804032x | 1.622438x | 89.9 | 55.9 | PASS |
| 7.2b / 9b | NVIDIA GeForce RTX 5090 | 8 | 128 | 128 | 867 | 550 | 1.574995x | 0.804032x | 1.266346x | 684 | 182 | PASS |
| 7.2b / 9b | NVIDIA GeForce RTX 5090 | 8 | 128 | 512 | 867 | 529 | 1.638816x | 0.804032x | 1.317660x | 658 | 209 | PASS |
| 7.2b / 9b | NVIDIA GeForce RTX 5090 | 8 | 512 | 128 | 866 | 529 | 1.637404x | 0.804032x | 1.316525x | 658 | 208 | PASS |
| 7.2b / 9b | NVIDIA GeForce RTX 5090 | 8 | 512 | 512 | 867 | 508 | 1.706545x | 0.804032x | 1.372116x | 632 | 235 | PASS |
| 7.2b / 9b | NVIDIA GeForce RTX 5090 | 8 | 2048 | 128 | 867 | 459 | 1.890928x | 0.804032x | 1.520366x | 570 | 297 | PASS |
| 7.2b / 9b | NVIDIA GeForce RTX 5090 | 8 | 2048 | 512 | 868 | 444 | 1.955493x | 0.804032x | 1.572278x | 552 | 316 | PASS |
