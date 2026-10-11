# Layer 3 / Tucker rank 48: frozen confirmation

Frozen before evaluation: layer 3, Tucker expert rank 48, 40% retained expert-weight payload, same input whitening, no training.
Previously unused WikiText-2 test blocks 32–95: 64 blocks of 512 tokens (32,704 scored next tokens).

| Method | Test ΔNLL |
|---|---:|
| svd_input | +0.05235105 |
| tucker_input_r64 | +0.08061040 |
| tucker_input_r48 | +0.03832068 |

Primary comparison, Tucker r48 minus SVD: **-0.01403037**, paired block-bootstrap 95% CI **[-0.02665317, -0.00342678]**.
Predetermined confirmation criterion passed: negative mean difference with the interval below zero.
Tucker r48 reduces the NLL damage relative to SVD by 26.8%; this is not an accuracy gain. It still has higher NLL than the original model.

Development advantage was -0.05094181 NLL; confirmation advantage is smaller but remains below zero under the declared interval procedure.
All per-projection diagnostics and payload bytes exactly match the development candidates; copied factor checksums are recorded in frozen_candidates.json.

Scope: this layer, budget, calibration and WikiText-2 evaluation. Contiguous blocks may be correlated; intervals do not quantify cross-domain or cross-layer uncertainty. No runtime memory/speed improvement is established.
Next candidate check: keep all factors fixed and compare on a second corpus before training or whole-model expansion.
