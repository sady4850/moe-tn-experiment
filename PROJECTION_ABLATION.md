# Single-projection Tucker ablation

Authorized 2026-09-10. Three hybrids, evaluated on the existing WikiText and C4 token files:

| Variant | gate | up | down |
|---|---|---|---|
| tucker_gate | Tucker r48 | SVD | SVD |
| tucker_up | SVD | Tucker r48 | SVD |
| tucker_down | SVD | SVD | Tucker r48 |

Reuse original WikiText-calibrated BF16 factors from `results/compression_crosslayer_3`. Fix OLMoE layer 3, 40% per-projection payload ceiling and all ranks. No decomposition, new calibration or training. Count exact achieved factor bytes; the small original SVD/Tucker payload differences remain.

Recompute original baseline on WikiText test blocks 32–95 (64) and C4's 256 saved document windows. Match old baselines exactly. Before hybrid evaluation, reconstruct all-SVD and all-Tucker using the same installer and require exact first-block NLL agreement with prior results on both corpora. Check source hashes before/after and original-model restoration after the six evaluations.

Report each hybrid minus all-SVD using paired per-item bootstrap intervals (2,000 resamples, NumPy seed 17). Compare full-Tucker minus SVD with the sum of the three hybrid-minus-SVD effects. Their difference is the aggregate nonadditivity residual, not an identification of individual pairwise interactions.

A hybrid with negative mean difference on both corpora is an exploratory candidate for new-data confirmation. No multiplicity correction; reused evaluations are diagnostic/selection data. WikiText adjacent blocks may be correlated. Dense reconstruction measures quality, not inference memory or speed.

Log/status `projection_ablation.log` / `projection_ablation.exit`.
Output `results/compression_layer3_projection_ablation`.
Runner `scripts/projection_ablation.py`. Download only JSON/Markdown artifacts.

## Completed outcome

Exit status 0. On WikiText, Tucker gate and up improve on all-SVD while Tucker down
worsens NLL. On C4, all three single-projection replacements worsen NLL. No candidate
has negative mean effect on both corpora. Nonadditivity is measurable on both corpora;
single-replacement effects cannot be summed to predict the full Tucker result.
All provenance, reconstruction controls and restoration checks passed.
Final report: `results/compression_layer3_projection_ablation/REPORT.md`.
