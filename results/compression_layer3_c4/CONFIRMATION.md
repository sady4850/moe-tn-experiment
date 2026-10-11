# Frozen layer-3 cross-corpus confirmation: C4

256 distinct-hostname documents; one 512-token window each; fixed WikiText calibration and factors.

| Method | Delta NLL |
|---|---:|
| svd_input | +0.02625163 |
| tucker_input_r64 | +0.05161624 |
| tucker_input_r48 | +0.05205419 |

Primary Tucker r48 minus SVD: +0.02580256; paired document-bootstrap 95% CI [0.021696989760240235, 0.030759516434193595].
Frozen success criterion passed: False. Wins: 20/256.

Scope: first eligible long pages from one C4 shard, one per hostname. Different sites can still be dependent.
No recovery or recalibration; dense reconstruction measures quality, not inference memory/speed.
Exact dataset revision, sampling, provenance and primary criterion are in PLAN.json.
