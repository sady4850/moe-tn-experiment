# Single-layer weight compression screen

Baseline NLL: 2.89555290

| Method | Target retained | Actual retained | ΔNLL | Paired block-bootstrap 95% CI |
|---|---:|---:|---:|---|
| svd_input | 40.0% | 39.99% | +0.05774123 | [+0.04638091, +0.06812315] |
| tucker_input_r64 | 40.0% | 39.95% | +0.08759704 | [+0.07406033, +0.09987105] |
| tucker_input_r48 | 40.0% | 39.83% | +0.05100194 | [+0.03401225, +0.06577219] |

## Direct paired comparisons

Positive difference means Tucker has higher NLL (worse).

- Tucker minus svd_input (tucker_input_r64) at 40.0%: +0.02985581 [+0.02520377, +0.03436237]
- Tucker minus tucker_input_r48 (tucker_input_r64) at 40.0%: +0.03659510 [+0.03174997, +0.04242933]
- Tucker minus svd_input (tucker_input_r48) at 40.0%: -0.00673929 [-0.01433801, -0.00018769]
- Tucker minus tucker_input_r64 (tucker_input_r48) at 40.0%: -0.03659510 [-0.04242933, -0.03174997]

Exploratory validation-set screen, fixed layer and text. Block-bootstrap intervals
do not represent uncertainty across datasets/layers; neighboring blocks can be correlated.
Variants ending in _input use the same shared input whitening; _none is the raw-weight control.
No recovery training. Tucker uses HOSVD; rank policy is recorded per projection.
Input whitening uses pooled uncentered second moments, not full MoE output sensitivity.
Retained fractions count factor payload at model dtype for expert weights in this layer only.
Dense reconstruction is used for NLL. No runtime memory saving or speedup is measured.
Whitening inverses are absorbed into factors before rounding; calibration artifacts are not inference payload.
A negative result does not exclude other tensor methods or recovery training.
