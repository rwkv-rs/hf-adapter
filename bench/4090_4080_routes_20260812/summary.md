# RTX 4090 reproduction of RTX 4080 routes

Status: **pass**

## B8 grouped W/A/V BMM

| Model | Median speedup | Range | VRAM delta | Greedy |
|---|---:|---:|---:|---:|
| 0.4b | **1.2002x** | 1.1992x-1.2002x | +27.0 MiB | 3072/3072 |
| 1.5b | **1.1426x** | 1.1420x-1.1426x | +280.8 MiB | 3072/3072 |
| 2.9b | **1.1259x** | 1.1258x-1.1266x | +709.5 MiB | 3072/3072 |

## Block-scoped FP16 Prefill accumulation

All 18 exact shapes pass. Minimum per-order speedup: **1.0057x**; largest shape median: **1.3007x**.

## Default policy

All 18 exact Prefill rows: **PASS**.
