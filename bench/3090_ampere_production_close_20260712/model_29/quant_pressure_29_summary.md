| Model | Prompt | Decode | Bsz | Quant | Tok/s total | Ratio | Footprint MB | Footprint ratio | Peak MB | Prompt cos | Final cos | Same next | Status |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2.9b | 128 | 128 | 8 | none | 617.0 | 1.0000 | 5622.4 | 1.0000 | 6165.3 | 1.000000 | 1.000000 | True | pass |
| 2.9b | 128 | 128 | 8 | mm8 | 617.7 | 0.9998 | 5462.6 | 0.9716 | 6780.6 | 0.999966 | 0.999960 | True | pass |
| 2.9b | 128 | 128 | 8 | mm4 | 401.2 | 0.6494 | 5382.6 | 0.9573 | 7028.7 | 0.999819 | 0.999826 | True | pass |
| 2.9b | 2048 | 128 | 8 | none | 617.6 | 1.0000 | 5622.4 | 1.0000 | 9015.4 | 1.000000 | 1.000000 | True | pass |
| 2.9b | 2048 | 128 | 8 | mm8 | 615.3 | 0.9968 | 5462.6 | 0.9716 | 9175.7 | 0.999958 | 0.999960 | True | pass |
| 2.9b | 2048 | 128 | 8 | mm4 | 400.5 | 0.6496 | 5382.6 | 0.9573 | 8783.8 | 0.999834 | 0.999840 | True | pass |

## Ratio summary
| Model | Quant | Rows | Min speed ratio | Median speed ratio | Min footprint ratio | Same-next pass |
|---|---|---:|---:|---:|---:|---:|
| 2.9b | mm4 | 2 | 0.6494 | 0.6495 | 0.9573 | 2/2 |
| 2.9b | mm8 | 2 | 0.9968 | 0.9983 | 0.9716 | 2/2 |

## Acceptance gate
FAIL: speed_fail=2/4 threshold=0.98
