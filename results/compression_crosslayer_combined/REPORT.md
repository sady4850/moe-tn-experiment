# Cross-layer screen: layers 3 and 13

OLMoE layer-wise compression, 40% retained expert-weight payload; shared input whitening; no recovery.
Exploratory validation blocks 32–63. This is not a new held-out confirmation test.

| Layer | Method | ΔNLL | Method − SVD | Paired 95% CI |
|---:|---|---:|---:|---|
| 3 | svd_input | +0.03325513 | +0.00000000 | [+0.00000000, +0.00000000] |
| 3 | tucker_input_r64 | +0.04206533 | +0.00881020 | [+0.00129448, +0.01598192] |
| 3 | tucker_input_r48 | -0.01768668 | -0.05094181 | [-0.07527275, -0.03001441] |
| 13 | svd_input | +0.03568103 | +0.00000000 | [+0.00000000, +0.00000000] |
| 13 | tucker_input_r64 | +0.08458674 | +0.04890571 | [+0.04053734, +0.05760438] |
| 13 | tucker_input_r48 | +0.10082582 | +0.06514479 | [+0.05251198, +0.07906562] |

Layer 3, Tucker rank 48 improves on SVD in this exploratory sample. Its ΔNLL vs the original model is negative, but the bootstrap interval vs original includes zero: no demonstrated improvement over the original model.
Layer 13, both Tucker variants lose to SVD. This supports a layer-specific hypothesis, not uniform benefit.

Verified identical baseline block losses, source/package versions, checkpoint and data settings. Actual retained payloads are about 39.83–39.99%, within the common 40% ceiling.
Intervals are conditional on fixed text and layers; neighboring blocks can be correlated and comparisons are exploratory, without multiplicity correction.

Next recommended test (not yet launched): freeze layer 3 / Tucker expert rank 48 and compare it against SVD and full-rank Tucker on previously unused test blocks 32–95. Keep calibration and the 40% payload budget fixed. Do not tune on this confirmation set.
