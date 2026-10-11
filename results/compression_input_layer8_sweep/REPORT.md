# Single-layer weight compression screen

Baseline NLL: 2.61374381

| Method | Target retained | Actual retained | ΔNLL | Paired block-bootstrap 95% CI |
|---|---:|---:|---:|---|
| svd_input | 60.0% | 59.91% | +0.01775881 | [+0.01132986, +0.02425135] |
| tucker_input | 60.0% | 59.87% | +0.03673109 | [+0.02783522, +0.04673920] |
| svd_input | 40.0% | 39.99% | +0.02478195 | [+0.01619653, +0.03387524] |
| tucker_input | 40.0% | 39.95% | +0.05389244 | [+0.04216940, +0.06754872] |

## Direct paired comparisons

Positive difference means Tucker has higher NLL (worse).

- Tucker minus svd_input (tucker_input) at 60.0%: +0.01897227 [+0.01290353, +0.02546528]
- Tucker minus svd_input (tucker_input) at 40.0%: +0.02911049 [+0.02219853, +0.03616447]

Exploratory validation-set screen, fixed layer and text. Block-bootstrap intervals
do not represent uncertainty across datasets/layers; neighboring blocks can be correlated.
Variants ending in _input use the same shared input whitening; _none is the raw-weight control.
No recovery training. Tucker uses HOSVD; rank policy is recorded per projection.
Input whitening uses pooled uncentered second moments, not full MoE output sensitivity.
Retained fractions count factor payload at model dtype for expert weights in this layer only.
Dense reconstruction is used for NLL. No runtime memory saving or speedup is measured.
Whitening inverses are absorbed into factors before rounding; calibration artifacts are not inference payload.
A negative result does not exclude other tensor methods or recovery training.
