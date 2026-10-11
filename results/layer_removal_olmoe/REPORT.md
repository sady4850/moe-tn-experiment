# Stage 1: one-shot layer removal (no healing)

Pre-registered in PRUNING_PLAN.md. Lower score = removed first.

## Layer scores and single-layer removal damage (dev ΔNLL)

| Layer | TN density | TN-whitened | BI | Dev ΔNLL if removed alone |
|---:|---:|---:|---:|---:|
| 0 | 0.9858 | 0.9333 | 0.2589 | +2.1931 |
| 1 | 0.9872 | 0.9569 | 0.1873 | +0.2834 |
| 2 | 0.9873 | 0.9627 | 0.1886 | +0.0116 |
| 3 | 0.9879 | 0.9704 | 0.1655 | +0.0741 |
| 4 | 0.9873 | 0.9680 | 0.1680 | +0.1079 |
| 5 | 0.9863 | 0.9658 | 0.1616 | +0.1674 |
| 6 | 0.9863 | 0.9667 | 0.1395 | +0.0855 |
| 7 | 0.9868 | 0.9669 | 0.1259 | +0.1329 |
| 8 | 0.9867 | 0.9673 | 0.1184 | +0.1192 |
| 9 | 0.9868 | 0.9665 | 0.1312 | +0.1094 |
| 10 | 0.9861 | 0.9651 | 0.1398 | +0.1713 |
| 11 | 0.9862 | 0.9633 | 0.1258 | +0.1382 |
| 12 | 0.9858 | 0.9640 | 0.1232 | +0.2771 |
| 13 | 0.9859 | 0.9648 | 0.1118 | +0.2299 |
| 14 | 0.9873 | 0.9645 | 0.1177 | +0.2616 |
| 15 | 0.9892 | 0.9482 | 0.2088 | +0.7425 |

Spearman of score with single-removal damage (positive = useful ordering): tn -0.362, tn_whitened -0.694, bi +0.050.

## k = 1

Dev reference: 16 (all) sets, median ΔNLL +0.1528, best +0.0116 (oracle [2]).

| Criterion | Set | Dev ΔNLL | Dev percentile | wikitext_test ΔNLL | c4 ΔNLL | Downstream mean (Δ pts) |
|---|---|---:|---:|---:|---:|---:|
| tn | [0] | +2.1931 | 100% | +2.2243 | +2.1876 | 40.19 (-30.31) |
| tn_whitened | [0] | +2.1931 | 100% | +2.2243 | +2.1876 | 40.19 (-30.31) |
| bi | [13] | +0.2299 | 69% | +0.2704 | +0.1851 | 67.33 (-3.17) |
| oracle | [2] | +0.0116 | 6% | +0.0879 | +0.2009 | 66.36 (-4.13) |

- **tn vs BI: lose**. NLL difference (test): wikitext_test +1.9539 [+1.8972, +2.0119]; c4 +2.0025 [+1.9608, +2.0469].
- **tn_whitened vs BI: lose**. NLL difference (test): wikitext_test +1.9539 [+1.8972, +2.0119]; c4 +2.0025 [+1.9608, +2.0469].

## k = 2 (primary)

Dev reference: 120 (all) sets, median ΔNLL +0.3901, best +0.1285 (oracle [2, 6]).

| Criterion | Set | Dev ΔNLL | Dev percentile | wikitext_test ΔNLL | c4 ΔNLL | Downstream mean (Δ pts) |
|---|---|---:|---:|---:|---:|---:|
| tn | [0, 12] | +2.3706 | 89% | +2.4043 | +2.3959 | 39.81 (-30.69) |
| tn_whitened | [0, 15] | +3.3453 | 100% | +3.5175 | +3.5642 | 39.52 (-30.98) |
| bi | [13, 14] | +0.7813 | 75% | +0.8713 | +0.6026 | 62.87 (-7.63) |
| oracle | [2, 6] | +0.1285 | 1% | +0.2115 | +0.3398 | 64.07 (-6.43) |

- **tn vs BI: lose**. NLL difference (test): wikitext_test +1.5330 [+1.4693, +1.5959]; c4 +1.7933 [+1.7480, +1.8401].
- **tn_whitened vs BI: lose**. NLL difference (test): wikitext_test +2.6463 [+2.5570, +2.7273]; c4 +2.9616 [+2.9098, +3.0159].

## k = 4

Dev reference: 102 random + criterion sets, median ΔNLL +1.3403, best +0.6307 (oracle [6, 7, 8, 13]).

| Criterion | Set | Dev ΔNLL | Dev percentile | wikitext_test ΔNLL | c4 ΔNLL | Downstream mean (Δ pts) |
|---|---|---:|---:|---:|---:|---:|
| tn | [0, 10, 12, 13] | +3.3782 | 85% | +3.4284 | +3.4831 | 38.25 (-32.25) |
| tn_whitened | [0, 1, 2, 15] | +4.8897 | 100% | +4.7988 | +4.6860 | 38.43 (-32.06) |
| bi | [8, 12, 13, 14] | +2.0679 | 69% | +2.1437 | +1.8562 | 53.19 (-17.31) |
| oracle | [6, 7, 8, 13] | +0.6307 | 1% | +0.6663 | +0.5139 | 58.17 (-12.33) |

- **tn vs BI: lose**. NLL difference (test): wikitext_test +1.2846 [+1.2078, +1.3595]; c4 +1.6269 [+1.5609, +1.6950].
- **tn_whitened vs BI: lose**. NLL difference (test): wikitext_test +2.6551 [+2.5095, +2.8045]; c4 +2.8297 [+2.7331, +2.9245].

## Pre-registered decision

Primary (k = 2) TN vs BI: **lose**. TN-whitened vs BI: **lose**.
Layer-level TN arm for stage 3: **none (killed)**.

Margin: lower NLL on every test corpus with the paired 95% CI below zero, and mean downstream accuracy not lower by more than 0.5 points. A loss also follows when the TN set is worse than the dev median. TN-whitened can replace TN only with a win.
