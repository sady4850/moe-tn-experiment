# Stage 2: one-shot expert removal and merging, 64 -> 40 experts in every layer (no healing)

Pre-registered in PRUNING_PLAN.md (amendment 2026-10-10, stage 2 definitions).

## Dev (WikiText validation) for all variants

| Family | Variant | Operator | Dev ΔNLL |
|---|---|---|---:|
| tn | tn_prune **(family best)** | prune | +1.1333 |
| tn | tn_merge | merge | +1.9001 |
| tn_whitened | tn_w_prune **(family best)** | prune | +1.4326 |
| tn_whitened | tn_w_merge | merge | +3.0355 |
| cheap | frequency **(family best)** | prune | +0.2207 |
| cheap | reap | prune | +0.4204 |
| cheap | output_similarity | merge | +1.3393 |
| random | random_prune_0 | prune | +0.7426 |
| random | random_prune_1 | prune | +0.9737 |
| random | random_prune_2 | prune | +0.8018 |
| random | random_merge_0 | merge | +1.8768 |
| random | random_merge_1 | merge | +3.4699 |
| random | random_merge_2 | merge | +2.6614 |

Median of the 6 random variants: +1.4253.

## Test

| Model | wikitext_test ΔNLL | c4 ΔNLL | Downstream mean (Δ pts) |
|---|---:|---:|---:|
| base | +0.0000 | +0.0000 | 70.50 (+0.00) |
| tn_prune | +1.2360 | +1.0609 | 43.28 (-27.22) |
| tn_w_prune | +1.5238 | +1.1651 | 45.98 (-24.51) |
| frequency | +0.2251 | +0.8640 | 48.82 (-21.68) |

- **tn (tn_prune) vs cheap (frequency): lose**. NLL difference (test): wikitext_test +1.0109 [+0.9677, +1.0535]; c4 +0.1969 [+0.1621, +0.2336].
- **tn_whitened (tn_w_prune) vs cheap (frequency): lose**. NLL difference (test): wikitext_test +1.2987 [+1.2448, +1.3551]; c4 +0.3010 [+0.2555, +0.3489].

## Pre-registered decision

TN vs cheap: **lose**. TN-whitened vs cheap: **lose**.
Expert-level variants for stage 3: **none (killed)**.

A loss also follows when the TN family best is worse on dev than the median random variant. TN-whitened can replace TN only with a win.
