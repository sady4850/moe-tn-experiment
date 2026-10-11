# Single-layer weight compression screen

Baseline NLL: 2.50262594

| Method | Target retained | Actual retained | ΔNLL | Paired block-bootstrap 95% CI |
|---|---:|---:|---:|---|
| svd_input | 40.0% | 39.99% | +0.02548912 | [+0.02038950, +0.03038802] |
| tucker_input_r64 | 40.0% | 39.95% | +0.04999116 | [+0.04291097, +0.05747882] |
| tucker_input_r48 | 40.0% | 39.83% | +0.04906358 | [+0.04102495, +0.05762970] |

## Direct paired comparisons

Positive difference means Tucker has higher NLL (worse).

- Tucker minus svd_input (tucker_input_r64) at 40.0%: +0.02450205 [+0.02098582, +0.02846795]
- Tucker minus tucker_input_r48 (tucker_input_r64) at 40.0%: +0.00092758 [-0.00150163, +0.00318190]
- Tucker minus svd_input (tucker_input_r48) at 40.0%: +0.02357447 [+0.01892375, +0.02900009]
- Tucker minus tucker_input_r64 (tucker_input_r48) at 40.0%: -0.00092758 [-0.00318190, +0.00150163]

Exploratory validation-set screen, fixed layer and text. Block-bootstrap intervals
do not represent uncertainty across datasets/layers; neighboring blocks can be correlated.
Variants ending in _input use the same shared input whitening; _none is the raw-weight control.
No recovery training. Tucker uses HOSVD; rank policy is recorded per projection.
Input whitening uses pooled uncentered second moments, not full MoE output sensitivity.
Retained fractions count factor payload at model dtype for expert weights in this layer only.
Dense reconstruction is used for NLL. No runtime memory saving or speedup is measured.
Whitening inverses are absorbed into factors before rounding; calibration artifacts are not inference payload.
A negative result does not exclude other tensor methods or recovery training.
