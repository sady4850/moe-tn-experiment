# Experiment 1: minimal direct-compression screen

Question: at the same factor-storage ceiling, does joint Tucker decomposition
preserve held-out NLL better than independent SVD or a shared matrix basis?

This first stage uses shared input whitening, one layer, one budget and **no
recovery training**. It adapts the input-whitening and balanced-rank approach
from TD-MoE to OLMoE and our controlled runner. It is not a full reproduction of
the paper. Inspect these four trials before investing in training or whole-model
compression.

| Variant | Representation | Input metric |
|---|---|---|
| `svd_input` | Independent SVD | Shared input whitening |
| `shared_input` | Shared matrix basis | Same input whitening |
| `tucker_input` | Joint HOSVD, TD-style balanced ranks | Same input whitening |
| `tucker_none` | Same HOSVD and ranks | Raw weights, whitening ablation |

## Frozen first run

- Same pinned OLMoE checkpoint as Experiment 0; layer 8, all 64 experts.
- Independently compress gate/up/down tensors; preserve router and expert IDs.
- Retain at most **80% of each projection's original weight payload**, counting
  all factors in the model's BF16 dtype. Report exact achieved bytes; integer
  ranks mean budgets need not be identical to the last byte.
- Independent SVD: uniform per-expert rank, singular values absorbed into left factors.
- Shared matrix basis: consider common input and output bases and choose the
  orientation with lower FP32 error in the whitened space under the ceiling.
- Tucker: truncated HOSVD of `[expert, output, input]`. Fix expert rank to the
  number of experts and balance relative input/output ranks following TD-MoE.
  Enforce a strict payload ceiling, unlike the upstream closest-budget search.
  Both Tucker variants have exactly the same ranks. No HOOI or evaluation tuning.
- Calibration: 64 x 512 = 32,768 WikiText-2 **train** tokens. Gate/up use MoE
  block input second moments; down uses pooled actual routed expert intermediate
  inputs. Save per-expert routing counts. Pool samples without router-probability
  weighting; this is a common linear-input metric, not full MoE output sensitivity.
- Normalize uncentered second moments by sample count, symmetrize, floor
  eigenvalues at 0.001 times mean diagonal (at least 1e-12), then compute L via
  Cholesky. Decompose W L; absorb L^-1 into input-side factors using triangular
  solves before BF16 rounding. Whitening matrices are not extra inference payload.
- All decompositions use FP32; factors are rounded to BF16 before reconstruction.
  Reconstruction uses FP32 arithmetic then casts weights to BF16 for NLL.
- 32 fixed 512-token WikiText-2 validation blocks (16,352 scored next tokens).
  Saved token IDs make resumes use exactly the same input. This is exploratory
  validation, not an untouched final test. Reserve the test split for follow-up.
- Primary result: ΔNLL versus original model and paired Tucker–baseline NLL
  differences. Block-bootstrap intervals are exploratory, not population claims.

**The evaluation reconstructs dense weights. Factor bytes are a storage metric;
the run does not demonstrate lower inference memory or faster inference.**

## Run on the box

Use the existing A100-class setup. An A100 40 GB, 64 GB host RAM and about 40 GB
free disk are conservative planning targets, not measured minimum requirements.
Only the GPU run can verify actual peak memory and runtime. No remote GPU has
been used to validate this new runner yet.

Copy this directory, excluding `.venv`, `.uv-cache`, `__pycache__`, and old
`results` (unless deliberately resuming). From its root, inside `tmux`:

```bash
uv sync --locked --extra dev
uv run pytest
uv run moe-tn-compress --config configs/compression_smoke.yaml
nvidia-smi
set -o pipefail
uv run moe-tn-compress --config configs/compression_layer8.yaml 2>&1 | tee compression.log
```

The full first stage is four trials, one shared calibration and the original-model baseline.
For a deliberate pause immediately after the first new trial:

```bash
uv run moe-tn-compress --config configs/compression_layer8.yaml --stop-after 1
```

Resume with the same command without `--stop-after`. Completed trials are skipped;
calibration accumulators, projection factors and individual evaluation blocks are reused. A crash during
one decomposition repeats that projection only. Config, source and package
version changes require a new `output_dir`; there is no implicit overwrite.
Do not run two processes against the same output directory.

## What to inspect

`results/compression_input_layer8/REPORT.md` updates after each completed method.
Each trial directory contains:

- `result.json`: block NLL, ΔNLL, actual payload bytes and projection diagnostics.
- `projection_stats.json`: ranks, orientation, raw and input-weighted errors of rounded reconstruction.
- `{gate,up,down}_proj.pt`: reusable compact factors and their metadata.
- `nll_blocks.json`: resumable evaluation progress.

The run root also saves baseline block losses, calibration/evaluation token IDs,
`calibration_moments.pt`, `whitening_stats.json` (coverage and regularization),
and environment metadata. These calibration files are separate research artifacts,
not part of the compressed inference payload.
Retain factors when copying results back: they can initialize a follow-up recovery
experiment. Trial time is time in the current attempt, not a speed benchmark.

## Decision after this run

1. Check finite NLL, actual budgets and projection errors first. Large degradation
   is a reason to inspect calibration coverage and the approximation,
   not to proceed automatically to more aggressive compression.
2. If Tucker improves over both baselines, examine paired differences and achieved
   bytes, then replicate the same budget on layers 3/13 with new output directories.
3. If shared and Tucker are close, there is no demonstrated extra benefit from
   Tucker. If intervals are too wide, collect more validation text in a separate run.
4. Compare `tucker_input` against `tucker_none` to isolate input whitening at fixed
   ranks. If all input-aware methods degrade badly, consider equal-budget local
   recovery for all three, or a more faithful routed-output objective.
5. Only after a reproducible quality benefit, add stronger compression, joint-layer
   evaluation, quantization comparisons, and real factorized inference measurements.

Random-model smoke results validate plumbing only; they are not evidence of benefit.

## Local validation completed

- 16 tests pass in the isolated locked environment (PyTorch 2.14.0,
  Transformers 5.7.0), including budget accounting, recovery of known separable
  weights, whitening/unwhitening, preservation of high-variance input directions,
  calibration resume, restoration after failure, and changed-config rejection.
- Installed `moe-tn-compress` entry point completes all four variants on the
  tiny random OLMoE and writes factor checkpoints, block losses and the report.
- Real-checkpoint CUDA execution, peak VRAM and runtime remain to be checked on the box.

## TD-MoE provenance and differences

Reference: https://github.com/ust-xu/TD-MoE at commit
`4b93f43e03c1b371e9568d5e245db9141a12f1b0`, especially
`src/tucker_decomposition.py` and `src/data_collection.py`. Attribution and MIT
license are included in `TD_MOE_NOTICE.txt`. No checkout of TD-MoE or TensorLy
installation is required on the box: the small adapted algorithms are self-contained.

Differences are intentional: OLMoE layout support; input-only forward calibration;
normalized FP32 moments with relative eigenvalue flooring; strict rank budget;
Gram-eigendecomposition HOSVD; and inverse transforms fused before factor rounding.
Upstream output whitening uses random-backward statistics and is outside this run.
Our previous energy-based rank policy remains available explicitly as `energy`;
the revised configs use `td_balanced` for both Tucker variants. Old results are
preserved in their original directories and cannot be mixed into this new run.
