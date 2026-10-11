# Layer 8 compression curve: SVD versus Tucker

OLMoE pinned checkpoint; shared input whitening; no recovery. Reduction is of expert weights in one layer, not the entire model.

| Target reduction | SVD ΔNLL | Tucker ΔNLL | Tucker − SVD | Paired block-bootstrap 95% CI |
|---|---:|---:|---:|---|
| 20% | +0.01311238 | +0.01907539 | +0.00596301 | [+0.00240421, +0.01008480] |
| 40% | +0.01775881 | +0.03673109 | +0.01897227 | [+0.01290353, +0.02546528] |
| 60% | +0.02478195 | +0.05389244 | +0.02911049 | [+0.02219853, +0.03616447] |

SVD has lower NLL damage at all three budgets. The gap grows as compression increases.
All three paired intervals exclude zero, conditional on these fixed validation blocks and layer.
This does not establish performance across layers, models or downstream tasks. Neighboring blocks may be correlated.

Both runs have identical source and packages, model and data settings, baseline block losses, and whitening diagnostics.
Baseline NLL: 2.61374381. Calibration/evaluation token files were copied from the original run.

Tucker uses the fixed full expert rank (64), with balanced input/output ranks. This sweep does not test reducing expert rank.
Actual payload fractions and ranks are recorded in summary.json. All methods stay within their target budgets.

Next candidate experiment, not yet launched: redistribute Tucker budget between expert rank and input/output ranks, select settings on separate development data, then compare against SVD on untouched evaluation text.
