# Is TN profiling a good guide for removing layers and experts?

Pre-registered plan, written 2026-10-10 before any quality measurement for these stages.
Changes after a stage has produced outcome data are recorded below as amendments with a date,
never edited in place.

## Question

At a fixed removal budget and with the same compression and healing procedure, does choosing
**which layers or experts to remove** by a tensor-network (TN) profile give a better final model than
cheap standard criteria?

The TN profile is operationalized from the authors' description (see README): effective
mode ranks at a fixed relative reconstruction error. Low rank means compressible / redundant;
high rank means "information-dense, leave it alone".

## Overall decision rule

- **Worth using**: TN beats the best cheap baseline **after healing** (stage 3) at the layer
  or the expert level, by the stage-3 margin below.
- **Not worth the extra machinery**: tie or loss after healing at both levels, or TN killed
  before healing at both levels (kill rules below).
- Scope: the answer is about OLMoE-1B-7B unless stage 4 is run. A positive result on OLMoE is
  the trigger for stage 4, not a claim about GPT-OSS.

Pre-healing results decide only what is worth healing. They do not answer the question alone.

## Common protocol

- Model: `allenai/OLMoE-1B-7B-0924` @ `6d84c48581ec`, BF16, one A100 80GB.
- Calibration: first 64 x 512 WikiText-2 train tokens (as in all earlier runs).
- Development: WikiText-2 validation blocks 0..63 (64 x 512). Used for exhaustive or random
  set evaluation only; no criterion has free parameters to tune on it.
- Test: WikiText-2 test blocks 32..95 and the existing 256 C4 validation documents
  (`results/compression_layer3_c4/documents.json`). Neither is used before the final
  comparison of each stage.
- Downstream (test only): lm-evaluation-harness 0.4.13, zero-shot, full task sets: `arc_easy` (acc),
  `arc_challenge` (acc_norm), `hellaswag` (acc_norm), `piqa` (acc), `winogrande` (acc); report each and
  the unweighted mean in accuracy points.
- Statistics: paired per-block / per-document percentile bootstrap, 2,000 resamples, NumPy
  `default_rng(17)`. No multiplicity correction; the primary comparison of each stage is named
  in advance and everything else is secondary.
- **Margin for "better" before healing**: lower NLL on **both** WikiText test and C4 with both
  95% intervals excluding zero, **and** mean downstream accuracy not lower by more than 0.5 points.

## Stage 0 — does the TN layer score vary at all? (CPU, no quality data)

For each of the 16 layers and each expert projection P in {gate, up, down}, take the raw weight
tensor `[expert=64, out, in]` in FP32. For each mode (expert, out, in) compute the effective rank of
the mode unfolding at relative error 10% (energy kept 1 - 0.1^2), divided by that mode's size.

**TN layer density** `D_l` = mean of these 9 normalized ranks. Attention is excluded: experts are
96% of each block's parameters. Sensitivity: 5% and 20% error.

Output: `D_l` for all layers and its range. No decision is taken from stage 0; it documents whether
the score has dynamic range before any model evaluation.

**Secondary TN score, activation-whitened density `D^w_l`** (added 2026-10-10, before any quality
data, after seeing that raw `D_l` sits near 0.99 on layers 0, 3 and 8). Identical to `D_l`, except each
expert matrix is replaced by `W_e L` before unfolding, where `L` is the Cholesky factor of the input
second moment used in Experiment 1: MoE block input for gate/up, pooled routed expert intermediate
inputs for down; same calibration tokens, normalization and eigenvalue floor (0.001 x mean diagonal).
Computed on the GPU box from the stage 1 calibration pass.

## Stage 1 — layer removal, one shot, no healing

Removal operator: skip the whole transformer block (attention + MoE); the residual stream passes
through unchanged. Every layer is eligible, including the first and last.

Criteria, each a fixed formula computed before any removal is evaluated:

| Criterion | Score per layer | Remove first |
|---|---|---|
| TN (primary) | `D_l` from stage 0 | lowest density |
| TN-whitened (secondary) | `D^w_l` | lowest density |
| BI (ShortGPT block influence) | 1 - mean token cosine(block input, block output) on calibration tokens | lowest BI |
| Random | uniform subsets | — |
| Oracle (reference only) | dev ΔNLL of the set itself | lowest damage |

Sets: one-shot top-k by score, no iterative re-scoring. Budgets k = 2 (**primary**, 12.5% of
layers, close to HyperNova's 4/36 = 11%), k = 1 and k = 4 (secondary).

Development measurements:
- all 16 single-layer removals (ΔNLL on dev) and Spearman of each score with this damage;
- k = 2: **all 120** two-layer sets on dev, so every criterion's choice gets an exact percentile;
- k = 4: 100 random sets on dev (`Random(17)`), plus the criterion sets.

**Primary comparison**: TN set minus BI set at k = 2 on the test corpora.

- **TN wins** if it is better than BI by the margin above.
- **TN loses** if BI is better than TN by the margin, **or** the TN set is worse than the median
  of the 120 two-layer sets on dev.
- Otherwise **tie**, including the case where TN and BI choose the same set ("no added value").

Kill rule: if TN loses at k = 2, the layer-level TN arm is not healed in stage 3.

TN-whitened is evaluated with the same rules and reported alongside. Because it is a second look,
it can replace the primary TN set in stage 3 only if it **wins** against BI by the full margin
(a tie is not enough). If both TN variants qualify, the primary one goes to stage 3.

## Stage 2 — expert removal and merging, one shot, no healing

Budget: 37.5% of experts in every MoE layer (64 -> 40, the HyperNova ratio 128 -> 80), all 16 layers.
Criterion definitions, the merge operator and the neuron-alignment method are frozen in an
amendment **before** stage 2 runs. Committed now:

- Pruning criteria: router usage frequency, REAP-style router-weighted activation norm, a TN score
  (global, correlation-aware, as described by the authors), random (3 seeds).
- Merging: TN-based grouping vs output-similarity grouping (HC-SMoE style) vs random grouping,
  all with the **same** frequency-weighted, neuron-aligned merge operator. No naive unaligned
  averaging.
- **Primary comparison**: best TN variant minus best cheap variant (usage / REAP / output
  similarity), on the test corpora, with the same margin and the same kill rule as stage 1.

## Stage 3 — healing

For each level that survives its kill rule: heal the TN set and the best cheap set with an
identical recipe (distillation from the original model's logits, same tokens, same steps),
**2 seeds each**. Recipe and token budget are frozen in an amendment before stage 3 runs.

- **TN wins after healing** if, on both corpora, mean NLL is lower with the paired interval
  excluding zero (pooling the two seeds), the gap is larger than twice the seed-to-seed
  difference, and mean downstream accuracy is at least 0.5 points higher.
- **Loses** if the cheap set wins by the same rule. Otherwise **tie**.

## Stage 4 — transfer (only if TN wins somewhere in stage 3)

Repeat the winning level on GPT-OSS-20B (same architecture family as GPT-OSS-120B), then
GPT-OSS-120B on an H100. Protocol frozen in an amendment.

## Running stage 1

Stage 0 runs locally (`scripts/layer_tn_density.py`, CPU) and writes `results/layer_tn_density/density.json`;
stage 1 refuses to start if that file changes afterwards. On the box:

```bash
uv sync --locked --extra dev --extra eval
uv run pytest
uv run python scripts/layer_removal.py --config configs/layer_removal_smoke.yaml
uv run python scripts/layer_removal.py --config configs/layer_removal_olmoe.yaml 2>&1 | tee layer_removal.log
```

The runner freezes all criterion sets in `selections.json` before the first evaluation, checks that
original-model NLL reproduces the earlier WikiText and C4 baselines, and resumes from saved per-set
results. Estimated A100 time: about 30 minutes for selections, dev and test, plus 1-2 hours for the
downstream tasks.

## Running stage 2

Needs the stage 1 results directory (baseline checks compare against its original-model NLL) and the
Experiment 0 results for layers 3/8/13 (the pairwise score must reproduce them). On the box:

```bash
uv sync --locked --extra dev --extra eval
uv run pytest
uv run python scripts/expert_removal.py --config configs/expert_removal_smoke.yaml
uv run python scripts/expert_removal.py --config configs/expert_removal_olmoe.yaml 2>&1 | tee expert_removal.log
```

Phases resume from disk: calibration statistics (`stats.pt`), one pairwise file per layer and kind,
then all 13 plans with their neuron permutations, frozen with checksums in `plans/index.json` before
the first evaluation.

GPT-OSS-20B (amendment 2026-10-11) needs an A100 80GB and at least 64GB of RAM (merges temporarily
keep up to ~29GB of original expert weights on the CPU):

```bash
uv run python scripts/expert_removal_gptoss.py --config configs/expert_removal_gptoss_smoke.yaml
uv run python scripts/expert_removal_gptoss.py --config configs/expert_removal_gptoss20b.yaml 2>&1 | tee expert_removal_gptoss.log
```

## Not in scope

Proprietary MPO / tensor-train profiling, quantization, inference speed or memory measurements,
iterative prune-heal schedules.

## Amendments

**2026-10-10, stage 1 runner sanity check (not a criterion change).** On the A10 box the original-model
dev NLL differs from the earlier A100 runs by BF16 kernel noise (per-block mean +3e-4, sd 3e-3, max
1.1e-2; block correlation 0.99996, versus -0.25 for misaligned blocks). The token check now requires block
correlation > 0.999 and |mean difference| < 2e-3 instead of max |diff| < 5e-3. Criterion sets had already
been frozen in `selections.json` and are reused unchanged; no criterion-vs-criterion quality data existed.
All comparisons are paired within one box, so the cross-GPU offset does not enter any verdict.

**2026-10-10, stage 2 definitions (frozen before any stage 2 run).** Written while stage 1 downstream
was still running; stage 1 test NLL was already known (TN loses to BI at every budget). Nothing below is
tuned on expert-level quality data, which does not exist yet.

*Setting.* All 16 MoE layers are compressed at once, 64 -> 40 experts each (24 removed or merged per layer).
Every statistic comes from the **original** model on the stage 1 calibration tokens (64 x 512 WikiText-2
train). Evaluation, margin, corpora and downstream tasks are those of stage 1.

*Operators.*
- **Prune**: the router logits of removed experts are set to -inf before the softmax and top-8, so the
  softmax runs over the 40 remaining experts. Top-k weights are not renormalized (`norm_topk_prob: false`,
  as in the original model).
- **Merge**: experts are grouped into 40 groups. In each group the **anchor** is the most frequently routed
  member. Every other member's intermediate neurons are permuted to the anchor's by the Hungarian algorithm
  (`scipy.optimize.linear_sum_assignment`) maximizing the summed Pearson correlation of intermediate
  activations `act(x W_gate^T) * (x W_up^T)` over all calibration tokens of that layer. Gate/up rows and
  down columns are permuted together. The merged expert is the average of aligned members weighted by
  calibration routing frequency (equal weights if all are zero). The router is unchanged; every original
  expert id routes to its group's merged expert, so when several members are selected their weights add.

*Scores.* "Pairwise incremental rank" is the Experiment 0 score: normalized incremental effective rank
of the two experts stacked along the shared mode (input mode for gate/up, output mode for down) at 10%
relative error, averaged over the three projections. Computed exactly from the eigenvalues of `G_a + G_b`
for all 2016 pairs per layer.

| Variant | Family | Operator | Definition |
|---|---|---|---|
| TN-prune | TN (primary) | prune | Remove the 24 experts with the lowest **mean pairwise incremental rank to the other 63** (most redundant with the rest of the layer). |
| TN-merge | TN (primary) | merge | Average-linkage agglomerative clustering into 40 groups, distance = pairwise incremental rank. |
| TN-w-prune, TN-w-merge | TN-whitened (secondary) | as above | Same, with every expert matrix replaced by `W_e L` (stage 1 whitening factors). |
| Frequency | cheap | prune | Remove the 24 least frequently routed experts (top-8 selections on calibration). |
| REAP | cheap | prune | Remove the 24 lowest `mean over routed tokens of p_j(x) * ||E_j(x)||_2`, with `p_j` the router weight used in the forward pass. |
| Output similarity | cheap | merge | Mean expert output over all calibration tokens (each expert applied to every token); average-linkage clustering into 40 groups on cosine distance (HC-SMoE style). |
| Random | reference | prune / merge | Seeds 0-2 (`Random(17 + seed)` per layer): 24 uniformly random experts removed; or a random permutation whose first 40 experts seed the groups and the other 24 join uniformly random groups. |

Why not the leave-one-out incremental rank of the whole 64-expert stack: stage 0 shows the stacked
unfoldings are near full rank (input-mode normalized rank 0.96-0.98), so removing one of 64 experts
changes the rank by ~0 and the score would be decided by tie-breaking. The mean pairwise score is the
continuous, correlation-aware version of the same idea.

*Selection and comparison.*
- Dev (WikiText validation, 64 blocks) for all 13 variants. The **dev-best member of each family** goes
  to test: TN-best (from TN-prune, TN-merge), cheap-best (from Frequency, REAP, Output similarity) and
  TN-w-best.
- **Primary comparison**: TN-best minus cheap-best on WikiText test and C4, stage 1 margin (both corpora
  with the paired 95% CI below zero, downstream mean not lower by more than 0.5 points).
- **Kill rule**: TN loses if cheap-best is better by the margin, or TN-best has higher dev NLL than the
  median of the 6 random variants. A tie sends both TN-best and cheap-best to stage 3.
- TN-w-best follows the stage 1 second-look rule: it can replace TN-best in stage 3 only with a win.
- Downstream tasks for the original model, TN-best, cheap-best and TN-w-best.

*Expected cost on the A10:* pairwise scores (2 x 16 x 2016 x 3 batched eigensolves of 2048 x 2048) about
1-1.5 h, calibration statistics and alignment about 30 min, 13 dev and 3 test evaluations about 30 min,
4 downstream runs about 30 min.

**2026-10-11, stage 2 replicated on GPT-OSS-20B regardless of the OLMoE outcome (frozen before any
stage 2 quality data on either model).** Written while the OLMoE stage 2 run was computing pairwise ranks.
Reason: GPT-OSS-120B experts share 5-10x more subspace beyond random than OLMoE experts (README), so an
OLMoE-only expert result says little about the authors' model family.

*Model and budget.* `openai/gpt-oss-20b` @ `6cee5e81ee83917806bbde320786a8fb61efebee`: 24 layers, 32 experts
per layer, top-4, 2880 x 2880 experts. MXFP4 experts are dequantized exactly to BF16 (as in
`scripts/null_rank_gptoss.py`) and every variant, including the original-model baseline, runs with BF16
experts. Budget 32 -> 20 experts in every layer (37.5%, the HyperNova ratio). One A100 80GB.

*Architecture-specific operators.* Same definitions as for OLMoE, with these adaptations:
- Router: GPT-OSS takes the top-4 router logits (with bias) and applies softmax over those four. Prune sets
  removed experts' logits to -inf before the top-4. Merge keeps the router and aliases member ids to the
  merged expert, so selected members' weights add.
- Expert biases: merged gate/up biases are permuted with their rows and averaged with the same weights;
  the merged down bias is the same weighted average (output space, no permutation).
- Intermediate activation for alignment and for the whitening of down: the model's own gated unit,
  `(clamp(up) + 1) * gate_c * sigmoid(1.702 * gate_c)` with `gate_c = clamp(gate, max=7)`, gate and up taken
  from the interleaved `gate_up_proj`.
- Biases are not part of any score (TN, frequency, REAP, output similarity are computed as for OLMoE;
  expert outputs used by REAP and output similarity include biases, as in the forward pass).

*Data.* GPT-OSS tokenizer on raw text (no chat template). WikiText calibration / dev / test as for OLMoE
(64 train blocks, validation blocks 0..63, test blocks 32..95, 512 tokens). C4: the OLMoE selection rule is
re-run with the GPT-OSS tokenizer (same shard and revision, first 256 eligible distinct hostnames with at
least 512 tokens, one window per document with `Random(17)`), because document lengths change with the
tokenizer. There is no earlier baseline to reproduce; the Experiment 0 check is OLMoE-only.

*Decision.* Verdicts follow the stage 2 rules on each model separately. For the overall question at the
expert level, **GPT-OSS-20B is decisive**: the expert-level TN arm counts as surviving only if it survives on
GPT-OSS-20B; the OLMoE verdict is reported alongside as evidence of model dependence. Healing (stage 3) on
GPT-OSS-20B, if needed, gets its own amendment.

**2026-10-11, GPT-OSS-20B stage 2 runner fix (not a criterion change, before any GPT-OSS quality data).** The
whitened pairwise incremental rank is negative for 6-48 of 496 pairs in GPT-OSS layers 6, 18, 19, 20 and 23 (min
-0.137; raw scores 0.093-0.286): when one expert carries much less whitened energy, the joint rank at 10% error can
fall below the larger single rank. SciPy's `linkage` rejects negative distances, so building TN-w-merge crashed after
the statistics, all pairwise files and the TN-prune, TN-merge and TN-w-prune plans had been computed. `cluster()` now
subtracts the minimum distance when it is negative. Average linkage is invariant to adding a constant to every
distance (each cluster-to-cluster average shifts by the same constant), so the clustering is the one the frozen
definition specifies; on OLMoE both merge plans are reproduced exactly for all 16 layers, also with shifted distances.
The score definition is unchanged. `PLAN.json` of the first attempt is kept as `PLAN.v1.json`; statistics and pairwise
files are reused, plans are rebuilt and frozen before any evaluation.
