# Cross-layer MoE merge-signal report

Layers: layer3, layer8, layer13
Pairs per layer: 48

| Signal | Mean within-layer Spearman | Cluster-bootstrap 95% CI | Mean top-5 ΔNLL vs random |
|---|---:|---:|---:|
| tn_incremental_rank | -0.0081 | [-0.2060, 0.1891] | +0.00193357 |
| activation_cosine | -0.0032 | [-0.1560, 0.1473] | +0.000398421 |
| weight_cosine | -0.0487 | [-0.2195, 0.1213] | +9.85083e-05 |
| router_usage_sum | 0.2690 | [0.0966, 0.4229] | +0.000129921 |
| router_probability_sum | 0.2168 | [0.0607, 0.3690] | -0.000107755 |
| router_co_routing | -0.1132 | [-0.2684, 0.0492] | +0.00100167 |

## Direct comparison with primary TN signal

Positive values favor the baseline over TN.

| Baseline | Mean Spearman difference | Cluster-bootstrap 95% CI |
|---|---:|---:|
| activation_cosine | 0.0048 | [-0.2349, 0.2415] |
| weight_cosine | -0.0406 | [-0.2817, 0.2102] |
| router_usage_sum | 0.2770 | [0.0409, 0.4957] |
| router_probability_sum | 0.2249 | [0.0021, 0.4466] |
| router_co_routing | -0.1051 | [-0.3504, 0.1493] |

## Per-layer Spearman correlations

| Signal | layer3 | layer8 | layer13 |
|---|---:|---:|---:|
| tn_incremental_rank | 0.0646 | -0.0111 | -0.0777 |
| activation_cosine | -0.0828 | 0.2731 | -0.2000 |
| weight_cosine | -0.1546 | 0.1249 | -0.1164 |
| router_usage_sum | 0.4167 | 0.2284 | 0.1618 |
| router_probability_sum | 0.3234 | 0.1296 | 0.1975 |
| router_co_routing | -0.1663 | 0.0026 | -0.1759 |

## TN reconstruction-error sensitivity

| Relative reconstruction error | Mean within-layer Spearman | Cluster-bootstrap 95% CI |
|---:|---:|---:|
| 1.0% | -0.0925 | [-0.2641, 0.0789] |
| 5.0% | -0.0190 | [-0.1992, 0.1574] |
| 10.0% | -0.0081 | [-0.2060, 0.1891] |
| 20.0% | 0.0262 | [-0.1711, 0.2264] |

All predictors are oriented so lower means safer to merge. Positive Spearman therefore means the signal tracks greater merge damage. The primary aggregate is the mean of within-layer correlations; it does not pool incomparable damage scales across layers.

The cluster bootstrap resamples the same expert-pair identities in every layer. Its interval quantifies pair-sampling uncertainty for these fixed layers, not uncertainty over all model layers.

This is a falsification test of one effective-rank TN proxy for selecting expert pairs. It is not a test of every tensorization method or of post-merge recovery training.
