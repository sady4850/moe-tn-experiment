# Single-layer weight compression screen

Baseline NLL: 2.61374381

| Method | Target retained | Actual retained | ΔNLL | Paired block-bootstrap 95% CI |
|---|---:|---:|---:|---|
| svd_input | 80.0% | 79.98% | +0.01311238 | [+0.00889064, +0.01749311] |
| shared_input | 80.0% | 79.96% | +0.03287162 | [+0.02725738, +0.03906894] |
| tucker_input | 80.0% | 79.81% | +0.01907539 | [+0.01408184, +0.02466429] |
| tucker_none | 80.0% | 79.81% | +0.01980425 | [+0.01341128, +0.02606971] |

## Direct paired comparisons

Positive difference means Tucker has higher NLL (worse).

- Tucker minus svd_input (tucker_input) at 80.0%: +0.00596301 [+0.00240421, +0.01008480]
- Tucker minus shared_input (tucker_input) at 80.0%: -0.01379623 [-0.01880890, -0.00884559]
- Tucker minus tucker_none (tucker_input) at 80.0%: -0.00072886 [-0.00512457, +0.00382050]

Exploratory validation-set screen, fixed layer and text. Block-bootstrap intervals
do not represent uncertainty across datasets/layers; neighboring blocks can be correlated.
Variants ending in _input use the same shared input whitening; _none is the raw-weight control.
No recovery training. Tucker uses HOSVD; rank policy is recorded per projection.
Input whitening uses pooled uncentered second moments, not full MoE output sensitivity.
Retained fractions count factor payload at model dtype for expert weights in this layer only.
Dense reconstruction is used for NLL. No runtime memory saving or speedup is measured.
Whitening inverses are absorbed into factors before rounding; calibration artifacts are not inference payload.
A negative result does not exclude other tensor methods or recovery training.
