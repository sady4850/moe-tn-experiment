# Calibration-domain diagnostic

Compare WikiText-calibrated (existing) and C4-calibrated (new) factors on both existing evaluation sets. This is diagnostic reuse of observed data, not a fresh confirmation test.

Fixed OLMoE checkpoint, layer 3, BF16, retained factor payload 40%, SVD input and Tucker input r64/r48, balanced spatial ranks, whitening floor 0.001. No rank search or recovery training.

New calibration: 64 documents × 512 tokens from `allenai/c4` English train shard 0 at revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`. First eligible unique hostnames and text hashes, excluding every C4 evaluation hostname and text hash. One random window per document with Python `Random(17)`, no added EOS or cross-document concatenation. Same 32,768-token calibration budget as WikiText.

Evaluation reuses exact token files: C4 256 documents and WikiText test blocks 32–95. Original baselines are recomputed and must match previous runs exactly. New factors are computed once and copied with checksum verification for the other evaluation corpus. Ranks and stored bytes must match the old calibration condition.

Primary contrast on C4 evaluation:

`(Tucker48 − SVD) with C4 calibration − (Tucker48 − SVD) with WikiText calibration`.

Negative values mean recalibration improves Tucker relative to SVD. Report paired per-document percentile-bootstrap 95% CI (2,000 resamples, seed 17), new Tucker–SVD gap, and per-method calibration effects. Report the same contrasts on WikiText, where adjacent blocks may be correlated.

One calibration sample per corpus: corpus effects, sample effects and sampling/packing differences are not separately identified. A negative primary contrast alone does not demonstrate Tucker beating SVD; the new method gap must also be inspected. No inference memory/speed claim.

Log / status: `calibration_crossover.log` / `calibration_crossover.exit`.
Output `results/compression_layer3_calibration_crossover`.
Runner `scripts/calibration_crossover.py`. Save reports/JSON locally, not factor checkpoints.

## Completed outcome

Exit status 0. C4 recalibration closes 8.6% of the Tucker48–SVD gap on C4;
the remaining gap is +0.02357447 NLL [95% CI +0.01892375, +0.02900009].
The relative calibration effect is -0.00222810 [-0.00406248, -0.00025549].
On WikiText, Tucker48's advantage shrinks from -0.01403037 to -0.00673929 NLL.
Thus changing calibration does not reverse the method ordering on either corpus.
All baseline/provenance/rank/payload checks passed. Full report:
`results/compression_layer3_calibration_crossover/REPORT.md`.
