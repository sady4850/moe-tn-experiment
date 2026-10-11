# Single-projection Tucker ablation

Fixed WikiText-calibrated factors, layer 3, 40% per-projection payload ceiling; no training.

| Evaluation | Variant | Retained payload | Delta NLL | Variant minus SVD | Paired 95% CI |
|---|---|---:|---:|---:|---|
| wikitext | all SVD | 39.9902% | +0.05235105 | 0 | — |
| wikitext | tucker_gate | 39.9361% | +0.02927199 | -0.02307906 | [-0.03685129, -0.01205989] |
| wikitext | tucker_up | 39.9361% | +0.02011483 | -0.03223622 | [-0.04096118, -0.02481327] |
| wikitext | tucker_down | 39.9361% | +0.07506857 | +0.02271752 | [+0.01914206, +0.02635348] |
| wikitext | all Tucker r48 | 39.8277% | +0.03832068 | -0.01403037 | [-0.02665317, -0.00342678] |
| c4 | all SVD | 39.9902% | +0.02625163 | 0 | — |
| c4 | tucker_gate | 39.9361% | +0.05044788 | +0.02419625 | [+0.02143184, +0.02739384] |
| c4 | tucker_up | 39.9361% | +0.04452003 | +0.01826841 | [+0.01417420, +0.02294914] |
| c4 | tucker_down | 39.9361% | +0.03605151 | +0.00979988 | [+0.00596800, +0.01418766] |
| c4 | all Tucker r48 | 39.8277% | +0.05205419 | +0.02580256 | [+0.02169699, +0.03075952] |

wikitext nonadditivity (full effect minus sum of single effects): +0.01856739, CI [0.009323352003750588, 0.028762488729099835].

c4 nonadditivity (full effect minus sum of single effects): -0.02646198, CI [-0.03347792051542995, -0.02022072241847526].

Candidates with negative mean on both corpora: []

Paired item percentile bootstrap 2000 resamples, NumPy default_rng(17); exploratory, no multiplicity correction. WikiText blocks can be correlated.

Diagnostic reuse of known evaluations, single layer/calibration/budget. Nonadditivity combines pairwise and higher-order interactions; it does not identify each interaction. New-data confirmation required after selection. Dense reconstruction, no runtime memory/speed claim.

Source SHA256 unchanged, exact original baselines, exact four full-method reconstruction controls, exact final original-model restoration checks. All six hybrids use declared saved projection factors.
