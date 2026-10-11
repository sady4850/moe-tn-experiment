# Single-layer weight compression screen

Baseline NLL: 3.07386998

| Method | Target retained | Actual retained | ΔNLL | Paired block-bootstrap 95% CI |
|---|---:|---:|---:|---|
| svd_input | 40.0% | 39.99% | +0.03568103 | [+0.01199659, +0.05611126] |
| tucker_input_r64 | 40.0% | 39.95% | +0.08458674 | [+0.06268859, +0.10829095] |
| tucker_input_r48 | 40.0% | 39.83% | +0.10082582 | [+0.08181389, +0.12180375] |

## Direct paired comparisons

Positive difference means Tucker has higher NLL (worse).

- Tucker minus svd_input (tucker_input_r64) at 40.0%: +0.04890571 [+0.04053734, +0.05760438]
- Tucker minus tucker_input_r48 (tucker_input_r64) at 40.0%: -0.01623908 [-0.02520429, -0.00723226]
- Tucker minus svd_input (tucker_input_r48) at 40.0%: +0.06514479 [+0.05251198, +0.07906562]
- Tucker minus tucker_input_r64 (tucker_input_r48) at 40.0%: +0.01623908 [+0.00723226, +0.02520429]

Exploratory validation-set screen, fixed layer and text. Block-bootstrap intervals
do not represent uncertainty across datasets/layers; neighboring blocks can be correlated.
Variants ending in _input use the same shared input whitening; _none is the raw-weight control.
No recovery training. Tucker uses HOSVD; rank policy is recorded per projection.
Input whitening uses pooled uncentered second moments, not full MoE output sensitivity.
Retained fractions count factor payload at model dtype for expert weights in this layer only.
Dense reconstruction is used for NLL. No runtime memory saving or speedup is measured.
Whitening inverses are absorbed into factors before rounding; calibration artifacts are not inference payload.
A negative result does not exclude other tensor methods or recovery training.
