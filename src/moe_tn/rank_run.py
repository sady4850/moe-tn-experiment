"""Select Tucker expert rank on development text; evaluate the frozen choice on test."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import yaml

from .compression_run import run, save_json, paired_interval, variant_expert_rank


def phase_configs(config):
    ranks = config["expert_ranks"]
    if not ranks or len(set(ranks)) != len(ranks) or any(not isinstance(r, int) or r < 1 for r in ranks):
        raise ValueError("Expert rank grid must contain distinct positive integers")
    if config["full_expert_rank"] not in ranks:
        raise ValueError("Grid must include the full-rank control")
    d = config["data"]
    if len({d["calibration_split"], d["development_split"], d["evaluation_split"]}) != 3:
        raise ValueError("Calibration, development and test must use different splits")
    if len(config["fractions"]) != 1:
        raise ValueError("This minimal rank test uses one budget")
    if config.get("tucker_rank_policy") != "td_balanced":
        raise ValueError("Rank test requires td_balanced policy")
    root = Path(config["output_dir"])
    dev = copy.deepcopy(config)
    dev["output_dir"] = str(root / "development")
    dev["methods"] = [f"tucker_input_r{r}" for r in ranks]
    dev["data"]["evaluation_split"] = d["development_split"]
    dev["data"]["evaluation_blocks"] = d["development_blocks"]
    dev["data"]["evaluation_offset_blocks"] = d["development_offset_blocks"]
    test = copy.deepcopy(config)
    test["output_dir"] = str(root / "test")
    return dev, test


def choose_candidate(rows, fraction, minimum_fill=0.98):
    for row in rows:
        if not np.isfinite(row["delta_nll"]):
            raise ValueError("Nonfinite development score")
        if not fraction * minimum_fill <= row["retained_fraction"] <= fraction:
            raise ValueError("Candidate misses matched payload budget; inspect ranks")
    if not rows:
        raise ValueError("No development candidates")
    # Tie preference is declared in advance: preserve the larger expert rank.
    return min(rows, key=lambda row: (row["delta_nll"], -variant_expert_rank(row["method"])))


def rank_run(config):
    dev, test = phase_configs(config)
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    plan = root / "PLAN.json"
    if plan.exists() and json.loads(plan.read_text()) != config:
        raise ValueError("Rank-test plan changed: use a new output_dir")
    save_json(plan, config)
    fraction = config["fractions"][0]
    print("PHASE 1: development rank selection", flush=True)
    run(dev)
    rows = [json.loads((Path(dev["output_dir"]) / f"{method}_{fraction:g}" / "result.json").read_text())
            for method in dev["methods"]]
    best = choose_candidate(rows, fraction, config.get("minimum_budget_fill", 0.98))
    selection = {"selected_method": best["method"], "criterion": "minimum development delta_nll",
                 "development_scores": {r["method"]: r["delta_nll"] for r in rows}}
    selection_path = root / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Frozen selection changed; do not reuse test results")
    # This file is frozen before test text is loaded or evaluated.
    save_json(selection_path, selection)
    control = f"tucker_input_r{config['full_expert_rank']}"
    test["methods"] = list(dict.fromkeys(["svd_input", control, best["method"]]))
    print(f"PHASE 2: test frozen choice {best['method']}", flush=True)
    run(test)
    test_rows = {m: json.loads((Path(test["output_dir"]) / f"{m}_{fraction:g}" / "result.json").read_text())
                 for m in test["methods"]}
    chosen = test_rows[best["method"]]
    comparisons = {}
    for method in ("svd_input", control):
        delta = np.array(chosen["block_nll"]) - np.array(test_rows[method]["block_nll"])
        comparisons[method] = {"selected_minus_baseline": float(delta.mean()),
                               "paired_95_ci": paired_interval(delta, config["seed"])}
    save_json(root / "summary.json", {"selection": selection, "comparisons": comparisons,
                                     "test_delta_nll": {m: r["delta_nll"] for m, r in test_rows.items()}})
    lines = ["# Tucker expert-rank allocation test", "",
             f"Layer {config['model']['layer']}; target retained payload {fraction:.0%}; shared input whitening; no training.", "",
             "## Development (selection only)", "",
             "| Expert rank | Retained payload | ΔNLL | Projection ranks |", "|---:|---:|---:|---|"]
    for row in rows:
        ranks = "; ".join(f"{p}: {r['ranks']}" for p, r in row["projections"].items())
        lines.append(f"| {variant_expert_rank(row['method'])} | {row['retained_fraction']:.4%} | {row['delta_nll']:+.8f} | {ranks} |")
    lines += ["", f"Frozen choice: **{best['method']}**.", "", "## Untouched test", "",
              "| Method | ΔNLL |", "|---|---:|"]
    lines += [f"| {m} | {r['delta_nll']:+.8f} |" for m, r in test_rows.items()]
    lines += ["", "Positive paired differences mean the selected Tucker is worse.", ""]
    for method, comparison in comparisons.items():
        lo, hi = comparison["paired_95_ci"]
        lines.append(f"- Selected minus {method}: {comparison['selected_minus_baseline']:+.8f} [{lo:+.8f}, {hi:+.8f}].")
    lines += ["", "Selection used development only; no alternative ranks were screened on test.",
              "If full rank wins development, selected and control are identical and evaluated once.",
              "Intervals are paired block-bootstrap intervals for fixed text/layer; neighboring blocks may be correlated.",
              "Reduction concerns expert weights in this layer, not the whole model. Dense reconstruction evaluates quality only.",
              "This tests the preregistered expert-rank grid with balanced spatial ranks, not every possible Tucker allocation.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(f"Complete: {root / 'REPORT.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    rank_run(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
