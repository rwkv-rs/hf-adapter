| Model | Prompt | Decode | Bsz | Quant | Tok/s total | Ratio | Footprint MB | Footprint ratio | Peak MB | Prompt cos | Final cos | Same next | Status |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 13.3b | 128 | 128 | 8 | none | 225.1 | 1.0000 | 25309.1 | 1.0000 | 26583.7 | 1.000000 | 1.000000 | True | pass |
| 13.3b | 128 | 128 | 8 | mm8 | 224.8 | 1.0013 | 25053.3 | 0.9899 | 26327.9 | 0.999976 | 0.999971 | True | pass |
| 13.3b | 128 | 128 | 8 | mm4 | 222.9 | 0.9845 | 24925.3 | 0.9848 | 26199.9 | 0.999851 | 0.999854 | True | pass |

## Ratio summary
| Model | Quant | Rows | Min speed ratio | Median speed ratio | Min footprint ratio | Same-next pass |
|---|---|---:|---:|---:|---:|---:|
| 13.3b | mm4 | 1 | 0.9845 | 0.9845 | 0.9848 | 1/1 |
| 13.3b | mm8 | 1 | 1.0013 | 1.0013 | 0.9899 | 1/1 |

## Acceptance gate
PASS
