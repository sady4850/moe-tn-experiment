# MoE TN-proxy experiment report

Pairs evaluated: 48
Primary reconstruction error: 10.0%

| Signal | Spearman vs merge damage | 95% bootstrap CI | top-k mean ΔNLL |
|---|---:|---:|---:|
| tn_incremental_rank | -0.0111 | [-0.3022, 0.2898] | 0.000501825 |
| activation_cosine | 0.2731 | [-0.0066, 0.5250] | -0.000135145 |
| weight_cosine | 0.1249 | [-0.1778, 0.4101] | -0.00147711 |
| router_usage_sum | 0.2284 | [-0.0721, 0.5223] | 0.000666603 |
| router_probability_sum | 0.1296 | [-0.1833, 0.4497] | 0.0011152 |
| router_co_routing | 0.0026 | [-0.2817, 0.2878] | 0.00121587 |

Random-pair mean ΔNLL: 0.00139767

Predictors are oriented so lower means 'safer to merge'; positive Spearman therefore means the signal correctly tracks greater merge damage.

This tests an effective-rank TN proxy for expert-pair selection, not tensorization in general.

## TN error-budget sensitivity

| Relative reconstruction error | Spearman vs merge damage | 95% bootstrap CI |
|---:|---:|---:|
| 1.0% | -0.0414 | [-0.3052, 0.2412] |
| 5.0% | -0.0034 | [-0.2623, 0.2799] |
| 10.0% | -0.0111 | [-0.3022, 0.2898] |
| 20.0% | 0.0295 | [-0.2810, 0.3502] |
