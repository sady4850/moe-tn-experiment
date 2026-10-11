# Stage 2 on openai/gpt-oss-20b: one-shot expert removal and merging, 32 -> 20 experts in every layer (no healing)

Pre-registered in PRUNING_PLAN.md (amendments 2026-10-10 stage 2 definitions and 2026-10-11 GPT-OSS-20B replication).

## Dev (WikiText validation) for all variants

| Family | Variant | Operator | Dev ΔNLL |
|---|---|---|---:|
| tn | tn_prune **(family best)** | prune | +0.3991 |
| tn | tn_merge | merge | +0.9334 |
| tn_whitened | tn_w_prune **(family best)** | prune | +0.6792 |
| tn_whitened | tn_w_merge | merge | +1.5846 |
| cheap | frequency **(family best)** | prune | -0.0179 |
| cheap | reap | prune | +0.3042 |
| cheap | output_similarity | merge | +2.1876 |
| random | random_prune_0 | prune | +0.5344 |
| random | random_prune_1 | prune | +0.7597 |
| random | random_prune_2 | prune | +0.5928 |
| random | random_merge_0 | merge | +0.8767 |
| random | random_merge_1 | merge | +0.9332 |
| random | random_merge_2 | merge | +1.0445 |

Median of the 6 random variants: +0.8182.

## Test

| Model | wikitext_test ΔNLL | c4 ΔNLL | Downstream mean (Δ pts) |
|---|---:|---:|---:|
| base | +0.0000 | +0.0000 | 65.12 (+0.00) |
| tn_prune | +0.3596 | -0.6433 | 56.93 (-8.19) |
| tn_w_prune | +0.6482 | -0.5220 | 58.60 (-6.52) |
| frequency | +0.0252 | +0.1033 | 63.59 (-1.53) |

- **tn (tn_prune) vs cheap (frequency): tie**. NLL difference (test): wikitext_test +0.3344 [+0.2794, +0.3824]; c4 -0.7466 [-0.7872, -0.7039].
- **tn_whitened (tn_w_prune) vs cheap (frequency): tie**. NLL difference (test): wikitext_test +0.6230 [+0.5630, +0.6721]; c4 -0.6253 [-0.6703, -0.5804].

## Pre-registered decision

TN vs cheap: **tie**. TN-whitened vs cheap: **tie**.
Expert-level variants for stage 3: **tn_prune, frequency**.

A loss also follows when the TN family best is worse on dev than the median random variant. TN-whitened can replace TN only with a win.
