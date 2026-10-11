# Single-layer weight compression screen

Baseline NLL: 3.07386998

| Method | Target retained | Actual retained | ΔNLL | Paired block-bootstrap 95% CI |
|---|---:|---:|---:|---|
| tucker_input_r64 | 40.0% | 39.95% | +0.04662236 | [+0.03721775, +0.05589476] |
| tucker_input_r48 | 40.0% | 39.83% | +0.03984736 | [+0.02982230, +0.04990606] |
| tucker_input_r32 | 40.0% | 39.85% | +0.03991679 | [+0.02859699, +0.05119519] |
| tucker_input_r24 | 40.0% | 39.96% | +0.04349002 | [+0.03178228, +0.05448554] |

## Direct paired comparisons

Positive difference means Tucker has higher NLL (worse).

- Tucker minus tucker_input_r48 (tucker_input_r64) at 40.0%: +0.00677500 [+0.00279400, +0.01057599]
- Tucker minus tucker_input_r32 (tucker_input_r64) at 40.0%: +0.00670557 [+0.00157985, +0.01180215]
- Tucker minus tucker_input_r24 (tucker_input_r64) at 40.0%: +0.00313234 [-0.00261396, +0.00860397]
- Tucker minus tucker_input_r64 (tucker_input_r48) at 40.0%: -0.00677500 [-0.01057599, -0.00279400]
- Tucker minus tucker_input_r32 (tucker_input_r48) at 40.0%: -0.00006943 [-0.00387164, +0.00402655]
- Tucker minus tucker_input_r24 (tucker_input_r48) at 40.0%: -0.00364266 [-0.00800314, +0.00087691]
- Tucker minus tucker_input_r64 (tucker_input_r32) at 40.0%: -0.00670557 [-0.01180215, -0.00157985]
- Tucker minus tucker_input_r48 (tucker_input_r32) at 40.0%: +0.00006943 [-0.00402655, +0.00387164]
- Tucker minus tucker_input_r24 (tucker_input_r32) at 40.0%: -0.00357323 [-0.00558251, -0.00148699]
- Tucker minus tucker_input_r64 (tucker_input_r24) at 40.0%: -0.00313234 [-0.00860397, +0.00261396]
- Tucker minus tucker_input_r48 (tucker_input_r24) at 40.0%: +0.00364266 [-0.00087691, +0.00800314]
- Tucker minus tucker_input_r32 (tucker_input_r24) at 40.0%: +0.00357323 [+0.00148699, +0.00558251]

Exploratory validation-set screen, fixed layer and text. Block-bootstrap intervals
do not represent uncertainty across datasets/layers; neighboring blocks can be correlated.
Variants ending in _input use the same shared input whitening; _none is the raw-weight control.
No recovery training. Tucker uses HOSVD; rank policy is recorded per projection.
Input whitening uses pooled uncentered second moments, not full MoE output sensitivity.
Retained fractions count factor payload at model dtype for expert weights in this layer only.
Dense reconstruction is used for NLL. No runtime memory saving or speedup is measured.
Whitening inverses are absorbed into factors before rounding; calibration artifacts are not inference payload.
A negative result does not exclude other tensor methods or recovery training.
