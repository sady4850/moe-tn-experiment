# Frozen layer-3 C4 confirmation

Follow-up to the WikiText-2 confirmation, authorized 2026-09-09.

- Model: same pinned OLMoE checkpoint, BF16, layer 3.
- Original model, SVD input, Tucker input r64, Tucker input r48.
- Retained expert-weight factor payload: approximately 40%, unchanged factors.
- Calibration: original WikiText train tokens and moments, copied and SHA256 verified.
- Evaluation: `allenai/c4`, English validation, revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`, shard `en/c4-validation.00000-of-00008.json.gz`.
- First 256 eligible distinct hostnames in shard order, unique text SHA256; documents must contain at least 512 tokens. Strip `www.` when checking hostnames.
- One random 512-token window per document using Python `Random(17)`. No concatenation or added EOS. 131,072 input tokens; 130,816 scored next tokens.
- Primary comparison fixed before evaluation: mean NLL of Tucker r48 minus SVD.
- Success requires a negative mean and paired document-bootstrap 95% percentile interval below zero, 2,000 replicates, NumPy `default_rng(17)`.
- No rank selection, recalibration, factor regeneration or recovery training.
- Different hostnames need not be statistically independent. This sample represents eligible long pages from one shard, not every C4 document. No claim that C4 was absent from model pretraining.
- Dense reconstruction measures quality only, not inference memory or speed.

The wrapper checks original compression source/package versions, refuses factor regeneration and data fallback, freezes the protocol before model evaluation, saves document/window provenance and verifies hashes before/after evaluation. Projection diagnostics and payload must match development exactly.

Log/status: `c4_confirmation.log` / `c4_confirmation.exit`.
Output: `results/compression_layer3_c4`.
Runner: `scripts/c4_layer3_confirmation.py`; launch script: `scripts/run_c4_confirmation.sh`.
Only JSON/Markdown reports should be downloaded, not factor checkpoints.

## Completed outcome

Exit status 0. Tucker r48 minus SVD is +0.02580256 NLL, paired document-bootstrap
95% CI [+0.02169699, +0.03075952]; the transfer criterion failed. Tucker r48 wins
on 20/256 documents. All frozen-artifact and projection-diagnostic checks passed.
Full local results: `results/compression_layer3_c4/CONFIRMATION.md`.
