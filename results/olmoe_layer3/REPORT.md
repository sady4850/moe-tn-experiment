# MoE TN-proxy experiment report

Pairs evaluated: 48
Primary reconstruction error: 10.0%

| Signal | Spearman vs merge damage | 95% bootstrap CI | top-k mean ΔNLL |
|---|---:|---:|---:|
| tn_incremental_rank | 0.0646 | [-0.2256, 0.3561] | 0.00308652 |
| activation_cosine | -0.0828 | [-0.3763, 0.2208] | 0.00105293 |
| weight_cosine | -0.1546 | [-0.4320, 0.1429] | 0.00318626 |
| router_usage_sum | 0.4167 | [0.1466, 0.6238] | 0.000635466 |
| router_probability_sum | 0.3234 | [0.0556, 0.5627] | 0.00204743 |
| router_co_routing | -0.1663 | [-0.4319, 0.1111] | 0.0024446 |

Random-pair mean ΔNLL: 0.00138669

Predictors are oriented so lower means 'safer to merge'; positive Spearman therefore means the signal correctly tracks greater merge damage.

This tests an effective-rank TN proxy for expert-pair selection, not tensorization in general.

## TN error-budget sensitivity

| Relative reconstruction error | Spearman vs merge damage | 95% bootstrap CI |
|---:|---:|---:|
| 1.0% | 0.0378 | [-0.2728, 0.3518] |
| 5.0% | 0.1093 | [-0.1882, 0.3811] |
| 10.0% | 0.0646 | [-0.2256, 0.3561] |
| 20.0% | 0.0422 | [-0.2807, 0.3334] |
