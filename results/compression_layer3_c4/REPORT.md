# Single-layer weight compression screen

Baseline NLL: 2.50262594

| Method | Target retained | Actual retained | ΔNLL | Paired block-bootstrap 95% CI |
|---|---:|---:|---:|---|
| svd_input | 40.0% | 39.99% | +0.02625163 | [+0.02010817, +0.03197567] |
| tucker_input_r64 | 40.0% | 39.95% | +0.05161624 | [+0.04341335, +0.06019390] |
| tucker_input_r48 | 40.0% | 39.83% | +0.05205419 | [+0.04394066, +0.06059524] |

## Direct paired comparisons

Positive difference means Tucker has higher NLL (worse).

- Tucker minus svd_input (tucker_input_r64) at 40.0%: +0.02536461 [+0.02140827, +0.02993921]
- Tucker minus tucker_input_r48 (tucker_input_r64) at 40.0%: -0.00043795 [-0.00228141, +0.00141070]
- Tucker minus svd_input (tucker_input_r48) at 40.0%: +0.02580256 [+0.02169699, +0.03075952]
- Tucker minus tucker_input_r64 (tucker_input_r48) at 40.0%: +0.00043795 [-0.00141070, +0.00228141]

Exploratory validation-set screen, fixed layer and text. Block-bootstrap intervals
do not represent uncertainty across datasets/layers; neighboring blocks can be correlated.
Variants ending in _input use the same shared input whitening; _none is the raw-weight control.
No recovery training. Tucker uses HOSVD; rank policy is recorded per projection.
Input whitening uses pooled uncentered second moments, not full MoE output sensitivity.
Retained fractions count factor payload at model dtype for expert weights in this layer only.
Dense reconstruction is used for NLL. No runtime memory saving or speedup is measured.
Whitening inverses are absorbed into factors before rounding; calibration artifacts are not inference payload.
A negative result does not exclude other tensor methods or recovery training.
