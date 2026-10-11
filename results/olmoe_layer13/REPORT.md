# MoE TN-proxy experiment report

Pairs evaluated: 48
Primary reconstruction error: 10.0%

| Signal | Spearman vs merge damage | 95% bootstrap CI | top-k mean ΔNLL |
|---|---:|---:|---:|
| tn_incremental_rank | -0.0777 | [-0.4003, 0.2394] | 0.00825122 |
| activation_cosine | -0.2000 | [-0.5092, 0.1229] | 0.00631633 |
| weight_cosine | -0.1164 | [-0.3883, 0.1672] | 0.00462523 |
| router_usage_sum | 0.1618 | [-0.1529, 0.4605] | 0.00512654 |
| router_probability_sum | 0.1975 | [-0.1258, 0.4871] | 0.00255296 |
| router_co_routing | -0.1759 | [-0.4720, 0.1371] | 0.00538338 |

Random-pair mean ΔNLL: 0.00325449

Predictors are oriented so lower means 'safer to merge'; positive Spearman therefore means the signal correctly tracks greater merge damage.

This tests an effective-rank TN proxy for expert-pair selection, not tensorization in general.

## TN error-budget sensitivity

| Relative reconstruction error | Spearman vs merge damage | 95% bootstrap CI |
|---:|---:|---:|
| 1.0% | -0.2738 | [-0.5301, 0.0298] |
| 5.0% | -0.1629 | [-0.4531, 0.1568] |
| 10.0% | -0.0777 | [-0.4003, 0.2394] |
| 20.0% | 0.0069 | [-0.3188, 0.3166] |
