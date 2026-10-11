# Single-layer weight compression screen

Baseline NLL: 3.07386998

| Method | Target retained | Actual retained | ΔNLL | Paired block-bootstrap 95% CI |
|---|---:|---:|---:|---|
| svd_input | 40.0% | 39.99% | +0.03325513 | [+0.02427946, +0.04289326] |
| tucker_input_r64 | 40.0% | 39.95% | +0.04206533 | [+0.02861036, +0.05596427] |
| tucker_input_r48 | 40.0% | 39.83% | -0.01768668 | [-0.04759685, +0.00888944] |

## Direct paired comparisons

Positive difference means Tucker has higher NLL (worse).

- Tucker minus svd_input (tucker_input_r64) at 40.0%: +0.00881020 [+0.00129448, +0.01598192]
- Tucker minus tucker_input_r48 (tucker_input_r64) at 40.0%: +0.05975201 [+0.04232609, +0.08039260]
- Tucker minus svd_input (tucker_input_r48) at 40.0%: -0.05094181 [-0.07527275, -0.03001441]
- Tucker minus tucker_input_r64 (tucker_input_r48) at 40.0%: -0.05975201 [-0.08039260, -0.04232609]

Exploratory validation-set screen, fixed layer and text. Block-bootstrap intervals
do not represent uncertainty across datasets/layers; neighboring blocks can be correlated.
Variants ending in _input use the same shared input whitening; _none is the raw-weight control.
No recovery training. Tucker uses HOSVD; rank policy is recorded per projection.
Input whitening uses pooled uncentered second moments, not full MoE output sensitivity.
Retained fractions count factor payload at model dtype for expert weights in this layer only.
Dense reconstruction is used for NLL. No runtime memory saving or speedup is measured.
Whitening inverses are absorbed into factors before rounding; calibration artifacts are not inference payload.
A negative result does not exclude other tensor methods or recovery training.
