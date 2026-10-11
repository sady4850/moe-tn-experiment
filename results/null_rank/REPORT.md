# Random-subspace null for pairwise incremental rank

Model `allenai/OLMoE-1B-7B-0924` @ `6d84c48581ec`, raw BF16 expert weights cast to FP32, CPU SVD.
Pairs: the saved Experiment 0 pairs. Null: second expert rotated by a Haar-random orthogonal matrix in the shared mode, 6 draws per pair and projection (pool of 16). Spectra of both experts are preserved exactly.

Score: normalized incremental rank averaged over gate/up/down, as in Experiment 0. Lower means more shared subspace.

## Observed vs null, primary error 10%

| Layer | Observed mean | Null mean | Observed − null (mean) | min / max per-pair difference | Pairs below their null min |
|---|---:|---:|---:|---:|---:|
| layer3 | 0.6447 | 0.6508 | -0.0062 | -0.0142 / +0.0035 | 46/48 |
| layer8 | 0.6347 | 0.6426 | -0.0079 | -0.0282 / +0.0062 | 43/48 |
| layer13 | 0.6383 | 0.6496 | -0.0113 | -0.0349 / +0.0009 | 46/48 |

## Error-budget sensitivity (mean observed − null over pairs)

| Layer | 1% | 5% | 10% | 20% |
|---|---:|---:|---:|---:|
| layer3 | -0.0016 | -0.0042 | -0.0062 | -0.0092 |
| layer8 | -0.0022 | -0.0055 | -0.0079 | -0.0110 |
| layer13 | -0.0034 | -0.0082 | -0.0113 | -0.0148 |

## Per-projection difference at 10% (mean observed − null)

| Layer | gate | up | down |
|---|---:|---:|---:|
| layer3 | -0.0046 | -0.0048 | -0.0090 |
| layer8 | -0.0059 | -0.0062 | -0.0116 |
| layer13 | -0.0094 | -0.0094 | -0.0149 |

## Principal-subspace overlap, all 2016 pairs per layer

Ratio of ||P_i P_j||_F^2 / min(r_i, r_j) to its random-subspace expectation max(r_i, r_j)/n, rank-r subspaces at 10% error. 1.0 means no more overlap than random subspaces.

| Layer | Projection | Mean | 99th pct | Max | Max pair |
|---|---|---:|---:|---:|---|
| layer3 | gate_proj | 1.016 | 1.054 | 1.076 | (51, 56) |
| layer3 | up_proj | 1.014 | 1.050 | 1.072 | (51, 56) |
| layer3 | down_proj | 1.032 | 1.074 | 1.096 | (0, 35) |
| layer8 | gate_proj | 1.019 | 1.091 | 1.134 | (20, 25) |
| layer8 | up_proj | 1.019 | 1.096 | 1.152 | (20, 25) |
| layer8 | down_proj | 1.032 | 1.119 | 1.172 | (20, 25) |
| layer13 | gate_proj | 1.024 | 1.091 | 1.118 | (8, 41) |
| layer13 | up_proj | 1.025 | 1.094 | 1.121 | (11, 19) |
| layer13 | down_proj | 1.030 | 1.099 | 1.131 | (11, 19) |

Recomputed joint ranks are checked against the stored Experiment 0 values (GPU vs CPU SVD may differ by one): mismatches greater than one per layer: layer3=0, layer8=0, layer13=0.

Raw weights only, no whitening or activation weighting. The null tests subspace alignment, not functional similarity or mergeability after neuron alignment.
