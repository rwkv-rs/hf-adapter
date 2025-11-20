| Model | Prompt | Decode | Bsz | Quant | Tok/s total | Ratio | Footprint MB | Footprint ratio | Peak MB | Prompt cos | Final cos | Same next | Status |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2.9b | 128 | 128 | 8 | none | 619.6 | 1.0000 | 5622.4 | 1.0000 | 6165.3 | 1.000000 | 1.000000 | True | pass |
| 2.9b | 128 | 128 | 8 | mm4 | 631.9 | 1.0208 | 5382.6 | 0.9573 | 5933.7 | 0.999819 | 0.999825 | True | pass |
| 2.9b | 2048 | 128 | 8 | none | 616.9 | 1.0000 | 5622.4 | 1.0000 | 9015.4 | 1.000000 | 1.000000 | True | pass |
| 2.9b | 2048 | 128 | 8 | mm4 | 629.8 | 1.0216 | 5382.6 | 0.9573 | 8783.8 | 0.999834 | 0.999838 | True | pass |

## Ratio summary
| Model | Quant | Rows | Min speed ratio | Median speed ratio | Min footprint ratio | Same-next pass |
|---|---|---:|---:|---:|---:|---:|
| 2.9b | mm4 | 2 | 1.0208 | 1.0212 | 0.9573 | 2/2 |

## Acceptance gate
PASS
