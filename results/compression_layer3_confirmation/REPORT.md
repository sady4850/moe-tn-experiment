# Single-layer weight compression screen

Baseline NLL: 2.89555290

| Method | Target retained | Actual retained | ΔNLL | Paired block-bootstrap 95% CI |
|---|---:|---:|---:|---|
| svd_input | 40.0% | 39.99% | +0.05235105 | [+0.04595267, +0.05877548] |
| tucker_input_r64 | 40.0% | 39.95% | +0.08061040 | [+0.07164818, +0.08977070] |
| tucker_input_r48 | 40.0% | 39.83% | +0.03832068 | [+0.02122004, +0.05271750] |

## Direct paired comparisons

Positive difference means Tucker has higher NLL (worse).

- Tucker minus svd_input (tucker_input_r64) at 40.0%: +0.02825936 [+0.02360436, +0.03281957]
- Tucker minus tucker_input_r48 (tucker_input_r64) at 40.0%: +0.04228972 [+0.03283978, +0.05374778]
- Tucker minus svd_input (tucker_input_r48) at 40.0%: -0.01403037 [-0.02665317, -0.00342678]
- Tucker minus tucker_input_r64 (tucker_input_r48) at 40.0%: -0.04228972 [-0.05374778, -0.03283978]

Exploratory validation-set screen, fixed layer and text. Block-bootstrap intervals
do not represent uncertainty across datasets/layers; neighboring blocks can be correlated.
Variants ending in _input use the same shared input whitening; _none is the raw-weight control.
No recovery training. Tucker uses HOSVD; rank policy is recorded per projection.
Input whitening uses pooled uncentered second moments, not full MoE output sensitivity.
Retained fractions count factor payload at model dtype for expert weights in this layer only.
Dense reconstruction is used for NLL. No runtime memory saving or speedup is measured.
Whitening inverses are absorbed into factors before rounding; calibration artifacts are not inference payload.
A negative result does not exclude other tensor methods or recovery training.
