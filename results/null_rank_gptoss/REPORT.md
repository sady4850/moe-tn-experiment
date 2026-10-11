# Random-subspace null for pairwise incremental rank: GPT-OSS-120B

Model `openai/gpt-oss-120b` @ `b5c939de8f75`. MXFP4 expert weights dequantized exactly to BF16, then FP32 CPU SVD. Biases ignored. Same score and null as `results/null_rank` (OLMoE): Haar rotation of the second expert in the shared hidden mode, 3 draws per pair and projection.

Score: normalized incremental rank, mean over gate/up/down; 0 means identical subspaces. Experts here are square (2880 x 2880), so a stacked pair can never exceed rank 2880 and the random-null score is capped well below OLMoE's. Compare observed with null, not across models.

## Observed vs null, primary error 10%

| Layer | Pairs | Mean effective rank / 2880 (gate, up, down) | Observed | Null | Observed − null | min per-pair diff | Below null min |
|---|---|---|---:|---:|---:|---:|---:|
| layer7 | random (32) | 0.71, 0.73, 0.67 | 0.2155 | 0.2376 | -0.0221 | -0.0333 | 32/32 |
| layer7 | top_overlap (5) | 0.71, 0.73, 0.67 | 0.1639 | 0.1956 | -0.0316 | -0.0416 | 5/5 |
| layer18 | random (32) | 0.68, 0.72, 0.68 | 0.2165 | 0.2405 | -0.0240 | -0.0338 | 32/32 |
| layer18 | top_overlap (5) | 0.68, 0.72, 0.68 | 0.2441 | 0.2833 | -0.0392 | -0.0418 | 5/5 |
| layer29 | random (32) | 0.73, 0.74, 0.72 | 0.2063 | 0.2240 | -0.0177 | -0.0256 | 32/32 |
| layer29 | top_overlap (5) | 0.73, 0.74, 0.72 | 0.2074 | 0.2389 | -0.0315 | -0.0325 | 5/5 |

Relative shift (observed − null) / null for random pairs, i.e. the fraction of the way from random toward identical subspaces: layer7 -9.3%, layer18 -10.0%, layer29 -7.9%.

## Error-budget sensitivity, random pairs (mean observed − null)

| Layer | 1% | 5% | 10% | 20% |
|---|---:|---:|---:|---:|
| layer7 | -0.0038 | -0.0131 | -0.0221 | -0.0367 |
| layer18 | -0.0081 | -0.0180 | -0.0240 | -0.0329 |
| layer29 | -0.0029 | -0.0118 | -0.0177 | -0.0247 |

## Principal-subspace overlap, all pairs

Ratio of ||P_i P_j||_F^2 / min(r_i, r_j) to the random expectation max(r_i, r_j)/n at 10% error. With near-full-rank square experts this ratio is bounded by n / max(r_i, r_j).

| Layer | Projection | Mean | 99th pct | Max |
|---|---|---:|---:|---:|
| layer7 | gate_proj | 1.023 | 1.049 | 1.071 |
| layer7 | up_proj | 1.022 | 1.048 | 1.070 |
| layer7 | down_proj | 1.065 | 1.118 | 1.300 |
| layer7 | top pairs (mean of projections) | (56, 84) 1.120, (71, 84) 1.112, (30, 84) 1.090, (37, 84) 1.078, (44, 84) 1.077 | | |
| layer18 | gate_proj | 1.016 | 1.043 | 1.110 |
| layer18 | up_proj | 1.013 | 1.029 | 1.052 |
| layer18 | down_proj | 1.078 | 1.101 | 1.121 |
| layer18 | top pairs (mean of projections) | (21, 25) 1.083, (60, 99) 1.076, (21, 82) 1.073, (25, 99) 1.069, (13, 108) 1.069 | | |
| layer29 | gate_proj | 1.015 | 1.031 | 1.040 |
| layer29 | up_proj | 1.014 | 1.027 | 1.036 |
| layer29 | down_proj | 1.041 | 1.061 | 1.071 |
| layer29 | top pairs (mean of projections) | (34, 53) 1.048, (53, 127) 1.047, (25, 53) 1.045, (53, 69) 1.045, (18, 53) 1.045 | | |

No merge-damage ground truth is measured for GPT-OSS. Raw weights only, no whitening or activation weighting, no neuron alignment.
