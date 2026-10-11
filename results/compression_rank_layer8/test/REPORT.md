# Single-layer weight compression screen

Baseline NLL: 2.78046051

| Method | Target retained | Actual retained | ΔNLL | Paired block-bootstrap 95% CI |
|---|---:|---:|---:|---|
| svd_input | 40.0% | 39.99% | +0.02515722 | [+0.01945332, +0.03080803] |
| tucker_input_r64 | 40.0% | 39.95% | +0.05863168 | [+0.05196661, +0.06521940] |
| tucker_input_r48 | 40.0% | 39.83% | +0.05821990 | [+0.05081074, +0.06559038] |

## Direct paired comparisons

Positive difference means Tucker has higher NLL (worse).

- Tucker minus svd_input (tucker_input_r64) at 40.0%: +0.03347446 [+0.02834626, +0.03838135]
- Tucker minus tucker_input_r48 (tucker_input_r64) at 40.0%: +0.00041178 [-0.00486184, +0.00536395]
- Tucker minus svd_input (tucker_input_r48) at 40.0%: +0.03306268 [+0.02665819, +0.03922830]
- Tucker minus tucker_input_r64 (tucker_input_r48) at 40.0%: -0.00041178 [-0.00536395, +0.00486184]

Exploratory validation-set screen, fixed layer and text. Block-bootstrap intervals
do not represent uncertainty across datasets/layers; neighboring blocks can be correlated.
Variants ending in _input use the same shared input whitening; _none is the raw-weight control.
No recovery training. Tucker uses HOSVD; rank policy is recorded per projection.
Input whitening uses pooled uncentered second moments, not full MoE output sensitivity.
Retained fractions count factor payload at model dtype for expert weights in this layer only.
Dense reconstruction is used for NLL. No runtime memory saving or speedup is measured.
Whitening inverses are absorbed into factors before rounding; calibration artifacts are not inference payload.
A negative result does not exclude other tensor methods or recovery training.
