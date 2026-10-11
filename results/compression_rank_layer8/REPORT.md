# Tucker expert-rank allocation test

Layer 8; target retained payload 40%; shared input whitening; no training.

## Development (selection only)

| Expert rank | Retained payload | ΔNLL | Projection ranks |
|---:|---:|---:|---|
| 64 | 39.9534% | +0.04662236 | gate_proj: [64, 625, 1260]; up_proj: [64, 625, 1260]; down_proj: [64, 1260, 625] |
| 48 | 39.8277% | +0.03984736 | gate_proj: [48, 720, 1440]; up_proj: [48, 720, 1440]; down_proj: [48, 1440, 720] |
| 32 | 39.8472% | +0.03991679 | gate_proj: [32, 875, 1750]; up_proj: [32, 875, 1750]; down_proj: [32, 1750, 875] |
| 24 | 39.9562% | +0.04349002 | gate_proj: [24, 1005, 2010]; up_proj: [24, 1005, 2010]; down_proj: [24, 2010, 1005] |

Frozen choice: **tucker_input_r48**.

## Untouched test

| Method | ΔNLL |
|---|---:|
| svd_input | +0.02515722 |
| tucker_input_r64 | +0.05863168 |
| tucker_input_r48 | +0.05821990 |

Positive paired differences mean the selected Tucker is worse.

- Selected minus svd_input: +0.03306268 [+0.02665819, +0.03922830].
- Selected minus tucker_input_r64: -0.00041178 [-0.00536395, +0.00486184].

Selection used development only; no alternative ranks were screened on test.
If full rank wins development, selected and control are identical and evaluated once.
Intervals are paired block-bootstrap intervals for fixed text/layer; neighboring blocks may be correlated.
Reduction concerns expert weights in this layer, not the whole model. Dense reconstruction evaluates quality only.
This tests the preregistered expert-rank grid with balanced spatial ranks, not every possible Tucker allocation.
