# Expert-rank allocation: minimal follow-up

Fixed OLMoE checkpoint, layer 8, input whitening, no training. Retain 40% of
expert weight payload (60% reduction in this layer). Test expert ranks
**64, 48, 32, 24**. For each, select balanced spatial ranks under the same strict
payload ceiling. Candidates must use at least 98% of that ceiling. This keeps
the comparison about rank allocation rather than different compression levels.

- Calibration: the same first 64 x 512 WikiText-2 train tokens as earlier runs.
- Development: validation blocks 32–63 (zero based), separate from previously
  inspected validation blocks 0–31. Evaluate the four predefined Tucker candidates.
- Freeze the candidate with minimum development ΔNLL; exact ties favor higher
  expert rank. Save `selection.json` before touching the test split.
- Test: first 32 x 512 WikiText-2 test blocks. Evaluate SVD input, Tucker rank 64,
  and the frozen selected candidate. If rank 64 wins development, evaluate it once.
- No development search for SVD; its rank follows the same fixed payload rule.
- Report paired test differences and block-bootstrap intervals. These concern one
  layer and text set, not general performance across models/tasks.

Run:

```bash
uv run python -m moe_tn.rank_run --config configs/compression_rank_layer8.yaml
```

Output: `results/compression_rank_layer8/REPORT.md`, with separate `development/`
and `test/` checkpoints, `PLAN.json`, `selection.json` and `summary.json`.
The runner can resume each phase using the unchanged config/source. It rejects a
changed plan or frozen selection. Only reports/JSON measurements need copying
back; factors stay on the GPU box and can be recomputed cheaply.

This tests one balanced spatial-rank rule at each expert rank. It does not
exhaustively search all Tucker ranks, optimize expert/channel permutations,
perform HOOI, or use recovery training. Source changes invalidate old runner
resumes, so the box uses a separate checkout for this test.

## Completed result

20 tests and CPU/CUDA smoke workflows passed. Real run completed with exit 0.
Development selected rank 48; test ΔNLL: SVD +0.02515722, full-rank Tucker
+0.05863168, selected Tucker +0.05821990. The selected-vs-full-rank paired
interval includes zero; selected Tucker remains worse than SVD. See
`results/compression_rank_layer8/REPORT.md` for ranks, budgets and intervals.
