# Does tensor-network profiling help compress MoE models? Controlled tests on OLMoE and GPT-OSS

Tensor-network (TN) compression pipelines for LLMs are sometimes described as using a *TN profile* to decide what to compress. Decompose a weight tensor, or a stack of experts along a shared mode, and read off the mode (bond) ranks needed to reach a fixed reconstruction error. The ranks are taken as a measure of non-redundant information:

- two experts that span nearly the same subspace add little rank when stacked, so they are candidates for merging;
- a layer whose rank stays high under truncation is information-dense and should be kept, while a low-rank one can go.

The question here is whether this profile chooses layers and experts to remove or merge better than cheap, standard criteria: block influence for layers, router frequency or REAP for experts. The repo tests it at small scale, with protocols frozen before each run, paired bootstrap intervals and no healing. The TN side is a transparent proxy (effective mode ranks at a fixed relative error, and TD-MoE style HOSVD), not any proprietary tensorization.

## Answer

- **Which layers to drop: TN does not help.** The TN layer density is nearly flat on OLMoE (0.9858-0.9892) and ranks layer 0, the most damaging layer to drop, as the most removable. ShortGPT block influence wins at every budget.
- **Which experts to prune: TN loses to router frequency.** On OLMoE TN pruning is worse than random. On GPT-OSS-20B it beats random but stays far behind frequency.
- **Which experts to merge: no signal.** The pairwise TN score does not predict real merge damage (router usage does), and TN-guided grouping performs like random grouping.
- **Joint Tucker across experts vs per-expert SVD: SVD wins**, except one OLMoE layer on WikiText, and that win does not transfer to C4.

| Question | Model | TN | Best cheap criterion | Verdict |
|---|---|---|---|---|
| Drop 2 of 16 layers | OLMoE-1B-7B | +2.40 NLL, 39.8 pts | block influence: +0.87 NLL, 62.9 pts | TN loses |
| Prune 37.5% of experts | OLMoE-1B-7B | +1.24 NLL, 43.3 pts | router frequency: +0.23 NLL, 48.8 pts | TN loses |
| Prune 37.5% of experts | GPT-OSS-20B | +0.36 NLL, 56.9 pts | router frequency: +0.03 NLL, 63.6 pts | formal tie, see below |
| Predict merge damage of expert pairs | OLMoE-1B-7B | Spearman -0.01 | router usage: Spearman 0.27 | no TN signal |
| Group experts for merging (37.5%) | GPT-OSS-20B | +0.93 dev NLL | random grouping: +0.88 to +1.04 | random level |
| Compress a layer at 40% payload | OLMoE-1B-7B | joint Tucker | per-expert SVD | SVD wins 2 of 3 layers; the third does not transfer |

NLL is held-out WikiText-2 test ΔNLL; pts is mean zero-shot accuracy over five tasks (original model: 70.5 for OLMoE, 65.1 for GPT-OSS-20B). The GPT-OSS-20B pruning comparison is formally a tie under the pre-registered rule only because TN looks better on C4, and diagnostics trace that to the original model degrading past token 128 on raw web text, not to better compression.

What might still be useful, untested here: on GPT-OSS-20B the TN score carries real signal that is nearly independent of router frequency, so it could work as a secondary criterion. And GPT-OSS experts share 5-10x more subspace than OLMoE experts, spread over the whole layer rather than in near-duplicate pairs, which suits a joint factorization better than choosing pairs to merge.

## Experiments

| Experiment | Question | Models | Report |
|---|---|---|---|
| 0 | Does pairwise effective rank predict merge damage? Random-subspace null | OLMoE (layers 3/8/13), GPT-OSS-120B null | [`results/combined`](results/combined/REPORT.md), [`results/null_rank`](results/null_rank/REPORT.md), [`results/null_rank_gptoss`](results/null_rank_gptoss/REPORT.md) |
| 1 | Joint Tucker vs per-expert SVD at a fixed storage budget | OLMoE (layers 3/8/13) | [COMPRESSION.md](COMPRESSION.md), [RANK_TEST.md](RANK_TEST.md), [C4_CONFIRMATION.md](C4_CONFIRMATION.md) |
| 2 | TN profile as a removal criterion, whole model, pre-registered | OLMoE, GPT-OSS-20B | [PRUNING_PLAN.md](PRUNING_PLAN.md), [layers](results/layer_removal_olmoe/REPORT.md), [experts OLMoE](results/expert_removal_olmoe/REPORT.md), [experts GPT-OSS-20B](results/expert_removal_gptoss20b/REPORT.md) |

Models: `allenai/OLMoE-1B-7B-0924` @ `6d84c48581ece794365f2b8e9cfb043c68ade9c5` (64 experts per layer, top-8) and `openai/gpt-oss-20b` @ `6cee5e81ee83917806bbde320786a8fb61efebee` (32 experts per layer, top-4, MXFP4 experts dequantized to BF16). Calibration on WikiText-2 train; evaluation on held-out WikiText-2 and 256 C4 validation documents.

## Experiment 0: does effective rank find mergeable experts?

Pairs of OLMoE experts are actually merged and the damage measured. The TN incremental rank has no correlation with merge damage; router usage does. A random-rotation null shows OLMoE experts share almost no subspace beyond chance, while GPT-OSS-120B experts share several times more, spread across the layer.

<details>
<summary>Signals, random-subspace null and the GPT-OSS-120B null</summary>

For 48 random expert pairs per layer, each pair is temporarily merged (50/50 weight average, both router IDs point to the merged expert, router unchanged). Ground truth is the held-out ΔNLL of that merge. Each signal is oriented so that lower means "safer to merge". So a useful signal has **positive** Spearman with damage.

TN proxy: for gate/up the pair is stacked along rows (shared input space), for down along columns (shared output space). The score is the incremental effective rank at a fixed relative Frobenius error, averaged over the three projections.

| Signal | Mean within-layer Spearman | 95% CI (cluster bootstrap) |
|---|---:|---:|
| TN incremental rank (10% error) | -0.008 | [-0.206, 0.189] |
| Activation cosine | -0.003 | [-0.156, 0.147] |
| Weight cosine | -0.049 | [-0.220, 0.121] |
| Router usage sum | **0.269** | [0.097, 0.423] |
| Router probability sum | 0.217 | [0.061, 0.369] |
| Router co-routing | -0.113 | [-0.268, 0.049] |

Changing the TN error budget from 1% to 20% gives -0.093 to +0.026, all intervals include zero. Router usage minus TN: +0.277 [0.041, 0.496].

Full report: [`results/combined/REPORT.md`](results/combined/REPORT.md).

### Random-subspace null

Is the near-constant TN score (about 0.64 for every pair) just what unrelated matrices give? For each saved pair, the second expert is rotated by a Haar-random orthogonal matrix in the shared mode (input space for gate/up, output space for down), 6 draws per pair. This keeps both spectra exactly and removes any subspace alignment. Normalized incremental rank, 10% error: 0 would mean identical subspaces.

| Layer | Observed | Random null | Observed - null | Pairs below their null min |
|---:|---:|---:|---:|---:|
| 3 | 0.645 | 0.651 | -0.006 | 46/48 |
| 8 | 0.635 | 0.643 | -0.008 | 43/48 |
| 13 | 0.638 | 0.650 | -0.011 | 46/48 |

- The shared structure is real (the null spread per pair is about 0.0007) but tiny: observed pairs cover 1-2% of the distance from random to identical subspaces, roughly 7-10 dimensions of a ~1500-dimensional joint rank per projection.
- There is no tail of mergeable pairs outside the sample. Over all 2016 pairs per layer, the overlap of rank-r principal subspaces is at most 1.07-1.17 times the random expectation (mean about 1.02).
- The null-adjusted excess does not track merge damage either: Spearman 0.26 / -0.07 / -0.13 on layers 3 / 8 / 13.

Raw weights only, no whitening or activation weighting, and no neuron alignment before merging. Full report: [`results/null_rank/REPORT.md`](results/null_rank/REPORT.md).

### Same null on GPT-OSS-120B

GPT-OSS-120B is the base of a public TN-compressed model, HyperNova-60B. It is a standard `GptOssForCausalLM` with 32 instead of 36 layers, 80 instead of 128 experts per layer and expert width 2560 instead of 2880. So the tensor representation can only have guided what to remove; it is not part of the final model.

The same measurement on GPT-OSS-120B layers 7 / 18 / 29 (MXFP4 experts dequantized exactly, 32 random pairs plus the 5 most-overlapping pairs per layer, 3 rotations each). Because experts are square (2880 x 2880), a stacked pair cannot exceed rank 2880, so the comparison is the shift from the random null as a fraction of the null, not the raw score:

| | OLMoE layers 3 / 8 / 13 | GPT-OSS-120B layers 7 / 18 / 29 |
|---|---|---|
| Random pairs: shift toward identical subspaces | 0.9% / 1.2% / 1.7% | 9.3% / 10.0% / 7.9% |
| Pair-to-pair sd of that shift | 0.5-1.2% | 1.2-2.3% |
| Most-overlapping pairs | n/a | 13-16% |
| Max subspace overlap over all pairs, ratio to random | 1.07-1.17 | 1.04-1.30 |

- GPT-OSS experts share several times more subspace than OLMoE experts. MXFP4 rounding cannot produce this; it pushes weights toward random, not toward each other.
- The sharing is spread over all experts rather than concentrated in near-duplicate pairs: the best pairs reach 13-16%, and single "hub" experts (84 in layer 7, 53 in layer 29) appear in all top pairs. No pair spans nearly the same subspace.
- So the pairwise criterion separates GPT-OSS pairs better than OLMoE pairs, but its range stays narrow. A layer-wide shared component is what a joint (Tucker or shared-basis) factorization could exploit; that is a different use than choosing pairs to merge.

No merge-damage ground truth is measured for GPT-OSS. Full report: [`results/null_rank_gptoss/REPORT.md`](results/null_rank_gptoss/REPORT.md).

Whether TN profiling is a good guide for removing layers and experts is tested in [Experiment 2](#experiment-2-tn-profile-as-a-removal-criterion-pre-registered).

</details>

## Experiment 1: direct compression at a fixed storage budget

All 64 experts of one OLMoE layer compressed at a fixed byte budget. Per-expert whitened SVD beats joint Tucker on layers 8 and 13; Tucker wins on layer 3 for WikiText but loses on C4, even after recalibrating on C4.

<details>
<summary>Compression curves and the layer-3 transfer tests</summary>

All 64 experts of one layer are compressed. Methods share the same input whitening from calibration activations (adapted from [TD-MoE](https://github.com/ust-xu/TD-MoE), see [COMPRESSION.md](COMPRESSION.md) for the exact differences). Budgets count all factor bytes in BF16. Dense weights are reconstructed for evaluation, so this measures quality only, not memory or speed.

- `svd_input`: independent per-expert SVD.
- `tucker_input_rK`: joint HOSVD of the `[expert, output, input]` tensor with expert rank K and balanced input/output ranks.

### Layer 8, compression curve (WikiText-2 validation)

| Reduction | SVD ΔNLL | Tucker (expert rank 64) ΔNLL | Tucker - SVD | 95% CI |
|---|---:|---:|---:|---:|
| 20% | +0.0131 | +0.0191 | +0.0060 | [+0.0024, +0.0101] |
| 40% | +0.0178 | +0.0367 | +0.0190 | [+0.0129, +0.0255] |
| 60% | +0.0248 | +0.0539 | +0.0291 | [+0.0222, +0.0362] |

Lowering the expert rank (64/48/32/24 chosen on development blocks, then frozen) does not change the picture. On the test split the selected rank 48 is still worse than SVD: +0.0331 [+0.0267, +0.0392]. See [RANK_TEST.md](RANK_TEST.md).

### Layers 3 and 13, 40% retained payload (WikiText-2 validation, exploratory)

| Layer | Method | ΔNLL | Method - SVD | 95% CI |
|---:|---|---:|---:|---:|
| 3 | SVD | +0.0333 | | |
| 3 | Tucker r64 | +0.0421 | +0.0088 | [+0.0013, +0.0160] |
| 3 | Tucker r48 | -0.0177 | **-0.0509** | [-0.0753, -0.0300] |
| 13 | SVD | +0.0357 | | |
| 13 | Tucker r64 | +0.0846 | +0.0489 | [+0.0405, +0.0576] |
| 13 | Tucker r48 | +0.1008 | +0.0651 | [+0.0525, +0.0791] |

### Layer 3 follow-up: confirmation and transfer

Layer 3 / Tucker r48 was frozen after the screen above and tested on data it had not seen.

| Test | Tucker r48 - SVD | 95% CI | Result |
|---|---:|---:|---|
| WikiText-2 test blocks (64 x 512) | -0.0140 | [-0.0267, -0.0034] | Tucker better (27% less damage), wins 33/64 blocks |
| C4 validation, 256 documents | +0.0258 | [+0.0217, +0.0308] | SVD better, Tucker wins 20/256 documents |
| C4 eval, factors recalibrated on C4 | +0.0236 | [+0.0189, +0.0290] | Still SVD better, recalibration closes 8.6% of the gap |
| WikiText eval, factors recalibrated on C4 | -0.0067 | [-0.0143, -0.0002] | Tucker advantage shrinks |

Swapping in Tucker for one projection at a time (gate, up or down, the other two SVD) gives no variant that is better on both corpora, and the effects are not additive. See [C4_CONFIRMATION.md](C4_CONFIRMATION.md), [CALIBRATION_CROSSOVER.md](CALIBRATION_CROSSOVER.md), [PROJECTION_ABLATION.md](PROJECTION_ABLATION.md).

</details>

## Experiment 2: TN profile as a removal criterion (pre-registered)

One-shot removal or merging of whole layers or of 37.5% of experts in every layer, scored on held-out NLL and downstream accuracy, with decision margins and kill rules fixed in advance. TN loses to block influence for layers and to router frequency for experts.

<details>
<summary>Protocol, all variants, verdicts and the GPT-OSS C4 diagnostics</summary>

[PRUNING_PLAN.md](PRUNING_PLAN.md) fixes the criteria, budgets, decision margins and kill rules before each stage; later changes are dated amendments. Everything is one-shot with no healing. Test data: WikiText-2 test blocks and 256 C4 validation documents. Downstream: lm-evaluation-harness 0.4.13, zero-shot `arc_easy`, `arc_challenge`, `hellaswag`, `piqa`, `winogrande`, unweighted mean. TN counts as better only if NLL is lower on **both** test corpora with the paired 95% CI below zero, and downstream mean is not lower by more than 0.5 points.

### Layers (OLMoE-1B-7B)

TN layer density: normalized effective mode rank of the `[expert, out, in]` tensor at 10% error, averaged over the three projections and modes; lowest density is removed first. TN-whitened applies the calibration input whitening first. The removal operator skips the whole block.

The density is nearly flat across the 16 layers (0.9858-0.9892), and its ordering goes the wrong way: Spearman with single-layer removal damage is -0.36 (block influence: +0.05). Layer 0 has the lowest density but is by far the most damaging single removal (+2.19 dev ΔNLL). Primary budget, 2 of 16 layers:

| Criterion | Removed | WikiText test ΔNLL | C4 ΔNLL | Downstream mean (base 70.50) |
|---|---|---:|---:|---:|
| TN | 0, 12 | +2.40 | +2.40 | 39.81 |
| TN-whitened | 0, 15 | +3.52 | +3.56 | 39.52 |
| Block influence (ShortGPT) | 13, 14 | +0.87 | +0.60 | 62.87 |
| Oracle (best of all 120 pairs on dev) | 2, 6 | +0.21 | +0.34 | 64.07 |

TN loses to block influence at k = 1, 2 and 4. Full report: [`results/layer_removal_olmoe/REPORT.md`](results/layer_removal_olmoe/REPORT.md).

### Experts (OLMoE-1B-7B and GPT-OSS-20B)

37.5% of experts removed or merged in every layer (64 -> 40 on OLMoE, 32 -> 20 on GPT-OSS-20B, the HyperNova ratio 128 -> 80). TN score: the Experiment 0 pairwise incremental rank for all pairs. TN-prune removes experts with the lowest mean score to the rest of the layer; TN-merge uses it as the distance for average-linkage clustering. All merges share one operator: neuron alignment by Hungarian matching of activation correlations, then a frequency-weighted average. Cheap criteria: router frequency, REAP (router-weighted output norm), HC-SMoE style output similarity for merging; three random seeds for each operator.

Dev ΔNLL (WikiText-2 validation), all variants:

| Variant | OLMoE prune | OLMoE merge | GPT-OSS-20B prune | GPT-OSS-20B merge |
|---|---:|---:|---:|---:|
| Router frequency | **+0.22** | | **-0.02** | |
| REAP | +0.42 | | +0.30 | |
| Output similarity | | +1.34 | | +2.19 |
| TN | +1.13 | +1.90 | +0.40 | +0.93 |
| TN-whitened | +1.43 | +3.04 | +0.68 | +1.58 |
| Random (3 seeds) | +0.74 to +0.97 | +1.88 to +3.47 | +0.53 to +0.76 | +0.88 to +1.04 |

Test, best TN variant vs best cheap variant (both pruning on both models):

| Model | Variant | WikiText test ΔNLL | C4 ΔNLL | Downstream mean |
|---|---|---:|---:|---:|
| OLMoE | original | | | 70.50 |
| OLMoE | TN-prune | +1.24 | +1.06 | 43.28 |
| OLMoE | frequency | +0.23 | +0.86 | 48.82 |
| GPT-OSS-20B | original | | | 65.12 |
| GPT-OSS-20B | TN-prune | +0.36 | -0.64 | 56.93 |
| GPT-OSS-20B | frequency | +0.03 | +0.10 | 63.59 |

- **OLMoE: TN loses.** The raw TN score varies by about 4% across experts and is unrelated to routing frequency, so TN pruning removes about 40% of routed traffic, as random pruning does (frequency pruning: 19%).
- **GPT-OSS-20B: formally a tie**, because TN is worse on WikiText (+0.33 [+0.28, +0.38]) and better on C4 (-0.75 [-0.79, -0.70]). Frequency is better on every downstream task. Here the TN score does carry signal: TN pruning beats all three random prunes, and its removed set overlaps with frequency's only at chance level (5.3 of 12 experts per layer, chance 4.5; REAP vs frequency: 7.8).
- **The GPT-OSS C4 result is not a real win.** Post-hoc diagnostics (not pre-registered, [`diagnostics/`](results/expert_removal_gptoss20b/diagnostics)) show the original model is unusually poor on these raw web-text windows: NLL 4.71 vs 2.90 on WikiText, and it rises past token 128, the sliding-window size (3.95 at positions 16-128, 4.94 at 128-511). TN pruning's whole C4 gain comes from those late positions. Attention is already the eager implementation with sinks, and prepending a special token makes NLL worse on both corpora.
- Merging is worse than pruning for every family on both models.

Stage 3 (healing) has not been run. Full reports: [`results/expert_removal_olmoe/REPORT.md`](results/expert_removal_olmoe/REPORT.md), [`results/expert_removal_gptoss20b/REPORT.md`](results/expert_removal_gptoss20b/REPORT.md).

</details>

## Limitations

- Small models. Experiments 0 and 1 use OLMoE only, with 64 small experts per layer; Experiment 2 adds GPT-OSS-20B. GPT-OSS-120B (128 experts per layer, 2880 x 2880, top-4) is only measured for the subspace null.
- Proxies, not a full TN pipeline. Experiments 0 and 2 use matrix effective rank. Experiment 1 uses one HOSVD with a fixed rank rule. No HOOI, no MPO/tensor-train, no permutation search.
- No healing. Real pipelines retrain after compression. A method that loses before healing can still win after it.
- Experiments 0 and 1 work one layer at a time and report NLL only. Experiment 2 compresses the whole model at a single budget (37.5% of experts, or 1/2/4 of 16 layers) with WikiText-2 calibration, and adds downstream tasks. No inference memory or speed measurements.
- Intervals are conditional on fixed text and layers. Adjacent WikiText blocks may be correlated. No multiplicity correction across the exploratory screens.

## Reproduce

<details>
<summary>Commands for all experiments</summary>

Install into a project-local environment and run unit plus end-to-end smoke tests. The smoke configs build a tiny random OLMoE locally, so they need no downloads. Their numbers are meaningless; they only check the plumbing.

```bash
uv sync --locked --extra dev
uv run pytest
uv run moe-tn --config configs/smoke.yaml --overwrite
uv run moe-tn-compress --config configs/compression_smoke.yaml
```

Experiment 0, one layer (A100-class GPU):

```bash
uv run moe-tn --config configs/olmoe.yaml 2>&1 | tee run.log
uv run moe-tn --config configs/olmoe_layer3.yaml
uv run moe-tn --config configs/olmoe_layer13.yaml
uv run moe-tn-combine \
  --layer layer3=results/olmoe_layer3 \
  --layer layer8=results/olmoe_layer8 \
  --layer layer13=results/olmoe_layer13 \
  --output results/combined \
  --bootstrap-samples 5000
```

Random-subspace null for experiment 0 (CPU only, about 30 minutes on an M1 Pro; downloads only the expert tensors of layers 3/8/13, about 0.8 GB per layer, via HTTP ranges):

```bash
uv run python scripts/null_rank.py
```

The same null on GPT-OSS-120B (CPU only, about 30 minutes per layer on an M1 Pro; downloads about 1.7 GB of packed MXFP4 expert tensors per layer):

```bash
uv run python scripts/null_rank_gptoss.py
```

Experiment 1:

```bash
uv run moe-tn-compress --config configs/compression_layer8.yaml
uv run moe-tn-compress --config configs/compression_layer8_sweep.yaml
uv run python -m moe_tn.rank_run --config configs/compression_rank_layer8.yaml
uv run moe-tn-compress --config configs/compression_crosslayer_3.yaml
uv run moe-tn-compress --config configs/compression_crosslayer_13.yaml
```

The layer-3 follow-ups reuse the saved layer-3 factors, so run them after `compression_crosslayer_3`:

```bash
uv run python scripts/prepare_layer3_confirmation.py
uv run moe-tn-compress --config configs/compression_layer3_confirmation.yaml
scripts/run_c4_confirmation.sh
scripts/run_calibration_crossover.sh
scripts/run_projection_ablation.sh
```

Experiment 2 (stage 0 on CPU; stages 1 and 2 on OLMoE fit an A10 24GB with 32GB RAM; GPT-OSS-20B needs an A100 80GB and at least 64GB RAM). Each runner has a smoke config with a tiny random model:

```bash
uv sync --locked --extra dev --extra eval
uv run python scripts/layer_tn_density.py
uv run python scripts/layer_removal.py --config configs/layer_removal_olmoe.yaml
uv run python scripts/expert_removal.py --config configs/expert_removal_olmoe.yaml
uv run python scripts/expert_removal_gptoss.py --config configs/expert_removal_gptoss20b.yaml
```

Runs are resumable and refuse to mix changed configs or source into an existing output directory.

The `results/` directory here contains only reports and JSON/CSV measurements. Factor checkpoints (`*.pt`, about 1 GB) are not included and are regenerated by the commands above.

</details>

## Attribution and license

The input-whitening and balanced Tucker rank design is adapted from [TD-MoE](https://github.com/ust-xu/TD-MoE) (MIT, see [TD_MOE_NOTICE.txt](TD_MOE_NOTICE.txt)). The rest of the code is MIT, see [LICENSE](LICENSE).
