"""Stage 2 of PRUNING_PLAN.md: one-shot expert removal and merging on OLMoE, no healing.

Phases, each resumable under output_dir:
  stats      routing frequency, REAP scores, mean expert outputs and whitening factors per layer,
             from the original model on the calibration tokens
  pairwise   pairwise incremental rank for all expert pairs, raw and whitened, one file per layer
  plans      removed experts or merge groups (with neuron permutations and weights) for all
             13 variants, frozen before any quality evaluation
  dev        every variant on WikiText validation; dev-best member of each family
  test       WikiText test and C4 for the original model and the three family winners
  downstream lm-evaluation-harness tasks for the same four models
  report     pre-registered verdicts
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import itertools
import json
from pathlib import Path
import random
import time
from types import MethodType

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
import torch
import torch.nn.functional as F
import yaml

from moe_tn.adapter import PROJECTIONS, OlmoeAdapter
from moe_tn.compression_run import paired_interval, save_json, save_tensor
from moe_tn.model import load_model
from moe_tn.signals import effective_rank_from_singular_values
from moe_tn.whitening import regularized_cholesky
from layer_removal import digest, downstream, evaluate_set, load_corpora, verdict

PRIMARY_ERROR = 0.10
FAMILIES = {
    "tn": ("tn_prune", "tn_merge"),
    "tn_whitened": ("tn_w_prune", "tn_w_merge"),
    "cheap": ("frequency", "reap", "output_similarity"),
    "random": tuple(f"random_{op}_{seed}" for op in ("prune", "merge") for seed in range(3)),
}
VARIANTS = tuple(v for members in FAMILIES.values() for v in members)


def is_merge(variant: str) -> bool:
    return "merge" in variant or variant == "output_similarity"


# ---------------------------------------------------------------------------------------------
# Applying a plan


def _masked_router_forward(self, hidden_states):
    """OlmoeTopKRouter.forward with removed experts' logits set to -inf before softmax / top-k."""
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight).masked_fill(self._removed_mask, float("-inf"))
    router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
    if self.norm_topk_prob:
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
    return router_logits, router_top_value.to(router_logits.dtype), router_indices


def merged_weights(adapter, members: list[dict]) -> dict[str, torch.Tensor]:
    """Frequency-weighted average of neuron-aligned members; each member has expert, perm, weight."""
    merged = {}
    for member in members:
        perm = torch.as_tensor(member["perm"], device=adapter.device)
        aligned = {
            "gate_proj": adapter.weight(member["expert"], "gate_proj").detach().float()[perm],
            "up_proj": adapter.weight(member["expert"], "up_proj").detach().float()[perm],
            "down_proj": adapter.weight(member["expert"], "down_proj").detach().float()[:, perm],
        }
        for projection, value in aligned.items():
            merged[projection] = merged.get(projection, 0) + member["weight"] * value
    return {projection: value.to(adapter.dtype) for projection, value in merged.items()}


@contextmanager
def compressed(model, plan: dict | None):
    """Temporarily apply a prune or merge plan to every MoE layer, then restore exactly."""
    if plan is None:
        yield
        return
    saved: list[tuple[OlmoeAdapter, int, str, torch.Tensor]] = []
    routers = []
    try:
        for layer, layer_plan in enumerate(plan["layers"]):
            adapter = OlmoeAdapter(model, layer)
            if plan["operator"] == "prune":
                router = adapter.block.gate
                assert type(router).__name__ == "OlmoeTopKRouter", type(router)
                mask = torch.zeros(adapter.num_experts, dtype=torch.bool, device=adapter.device)
                mask[layer_plan["removed"]] = True
                router._removed_mask = mask
                router.forward = MethodType(_masked_router_forward, router)
                routers.append(router)
                continue
            for group in layer_plan["groups"]:
                if len(group) == 1:
                    continue
                weights = merged_weights(adapter, group)
                with torch.no_grad():
                    for member in group:
                        for projection in PROJECTIONS:
                            target = adapter.weight(member["expert"], projection)
                            saved.append((adapter, member["expert"], projection, target.detach().cpu().clone()))
                            target.copy_(weights[projection])
        yield
    finally:
        with torch.no_grad():
            for adapter, expert, projection, value in reversed(saved):
                adapter.weight(expert, projection).copy_(value.to(adapter.device))
        for router in routers:
            del router.forward, router._removed_mask


# ---------------------------------------------------------------------------------------------
# Statistics from the original model


@torch.inference_mode()
def capture_moe_inputs(model, blocks) -> list[torch.Tensor]:
    """Inputs of every MoE block over all calibration tokens, kept on CPU in the model dtype."""
    layers = model.model.layers
    captured: list[list[torch.Tensor]] = [[] for _ in layers]
    handles = [layer.mlp.register_forward_pre_hook(
        lambda module, args, index=i: captured[index].append(args[0].reshape(-1, args[0].shape[-1]).cpu()))
        for i, layer in enumerate(layers)]
    device = next(model.parameters()).device
    try:
        for index in range(len(blocks)):
            model(input_ids=blocks[index:index + 1].to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return [torch.cat(chunks) for chunks in captured]


def intermediate(adapter, x: torch.Tensor, expert: int) -> torch.Tensor:
    act = adapter.block.experts.act_fn
    return act(F.linear(x, adapter.weight(expert, "gate_proj"))) * F.linear(x, adapter.weight(expert, "up_proj"))


@torch.inference_mode()
def layer_statistics(adapter, x: torch.Tensor, floor: float, chunk: int = 8192) -> dict:
    """Routing frequency, REAP, mean expert output (all tokens) and whitening factors."""
    num_experts = adapter.num_experts
    _, top_values, top_indices = adapter.block.gate(x)
    frequency = torch.bincount(top_indices.reshape(-1), minlength=num_experts).cpu()
    reap_sum = torch.zeros(num_experts, dtype=torch.float64)
    output_sum = torch.zeros(num_experts, x.shape[1], dtype=torch.float64)
    hidden_moment = torch.zeros(x.shape[1], x.shape[1], device=adapter.device)
    intermediate_size = adapter.weight(0, "down_proj").shape[1]
    down_moment = torch.zeros(intermediate_size, intermediate_size, device=adapter.device)
    for start in range(0, len(x), chunk):
        xf = x[start:start + chunk].float()
        hidden_moment += xf.T @ xf
    for expert in range(num_experts):
        routed_mask = (top_indices == expert)
        for start in range(0, len(x), chunk):
            xs = x[start:start + chunk]
            h = intermediate(adapter, xs, expert)
            y = F.linear(h, adapter.weight(expert, "down_proj"))
            output_sum[expert] += y.float().sum(0).double().cpu()
            mask = routed_mask[start:start + chunk]
            routed = mask.any(-1)
            if routed.any():
                weight = (top_values[start:start + chunk].float() * mask).sum(-1)[routed]
                reap_sum[expert] += (weight * y[routed].float().norm(dim=-1)).sum().double().cpu()
                hr = h[routed].float()
                down_moment += hr.T @ hr
    rows = len(x)
    l_hidden, hidden_stats = regularized_cholesky(hidden_moment / rows, floor)
    l_down, down_stats = regularized_cholesky(down_moment / (rows * adapter.top_k), floor)
    return {
        "frequency": frequency,
        "reap": (reap_sum / frequency.clamp_min(1).double()).float(),
        "mean_output": (output_sum / rows).float(),
        "l_hidden": l_hidden.cpu(), "l_down": l_down.cpu(),
        "whitening": {"hidden": hidden_stats, "down": down_stats},
    }


# ---------------------------------------------------------------------------------------------
# Pairwise incremental rank


def shared_mode_gram(matrix: torch.Tensor, projection: str) -> torch.Tensor:
    # gate/up share the input mode (rows stacked); down shares the output mode (columns stacked).
    return matrix @ matrix.T if projection == "down_proj" else matrix.T @ matrix


def rank_from_eigenvalues(values: torch.Tensor, error: float) -> int:
    singular = values.clamp_min(0).flip(-1).sqrt().cpu()
    return effective_rank_from_singular_values(singular, error)


@torch.inference_mode()
def pairwise_incremental_rank(matrices: dict[str, list[torch.Tensor]], batch: int = 16) -> np.ndarray:
    """Experiment 0 score for all pairs: mean over projections of (joint - max) / max effective rank."""
    num_experts = len(matrices["gate_proj"])
    pairs = list(itertools.combinations(range(num_experts), 2))
    score = np.zeros((num_experts, num_experts))
    for projection in PROJECTIONS:
        grams = torch.stack([shared_mode_gram(m.float(), projection) for m in matrices[projection]])
        ranks = [rank_from_eigenvalues(torch.linalg.eigvalsh(g), PRIMARY_ERROR) for g in grams]
        for start in range(0, len(pairs), batch):
            chunk = pairs[start:start + batch]
            a = torch.tensor([p[0] for p in chunk], device=grams.device)
            b = torch.tensor([p[1] for p in chunk], device=grams.device)
            joint = torch.linalg.eigvalsh(grams[a] + grams[b])
            for (i, j), values in zip(chunk, joint):
                reference = max(ranks[i], ranks[j])
                value = (rank_from_eigenvalues(values, PRIMARY_ERROR) - reference) / max(reference, 1)
                score[i, j] += value / len(PROJECTIONS)
                score[j, i] += value / len(PROJECTIONS)
        del grams
    return score


# ---------------------------------------------------------------------------------------------
# Plans


def lowest(scores, count: int) -> list[int]:
    return sorted(sorted(range(len(scores)), key=lambda e: (float(scores[e]), e))[:count])


def cluster(distance: np.ndarray, groups: int) -> list[list[int]]:
    condensed = squareform(np.asarray(distance, dtype=float), checks=False)
    # Average linkage is unchanged by adding a constant to every distance; SciPy rejects negative ones.
    condensed = condensed - min(condensed.min(), 0.0)
    labels = cut_tree(linkage(condensed, method="average"), n_clusters=groups).reshape(-1)
    result = [sorted(np.flatnonzero(labels == g).tolist()) for g in range(groups)]
    assert all(result) and sum(map(len, result)) == len(distance)
    return sorted(result)


def random_groups(rng: random.Random, num_experts: int, groups: int) -> list[list[int]]:
    order = rng.sample(range(num_experts), num_experts)
    result = [[seed] for seed in order[:groups]]
    for expert in order[groups:]:
        result[rng.randrange(groups)].append(expert)
    return sorted(sorted(g) for g in result)


@torch.inference_mode()
def align_group(adapter, x: torch.Tensor, group: list[int], frequency: torch.Tensor) -> list[dict]:
    """Anchor = most frequent member; others permuted to it by Hungarian matching of activation correlations."""
    anchor = min(group, key=lambda e: (-int(frequency[e]), e))
    total = float(sum(int(frequency[e]) for e in group))

    def standardized(expert):
        h = intermediate(adapter, x, expert).float()
        h = h - h.mean(0)
        return h / h.norm(dim=0).clamp_min(1e-12)

    reference = standardized(anchor)
    members = []
    for expert in group:
        weight = int(frequency[expert]) / total if total > 0 else 1.0 / len(group)
        if expert == anchor:
            perm = list(range(reference.shape[1]))
        else:
            correlation = (reference.T @ standardized(expert)).cpu().numpy()
            rows, columns = linear_sum_assignment(-correlation)
            perm = columns[np.argsort(rows)].tolist()
        members.append({"expert": expert, "perm": perm, "weight": weight})
    return members


def build_plans(model, inputs, stats, pairwise, keep: int, seed: int) -> dict[str, dict]:
    num_layers = len(stats)
    num_experts = len(stats[0]["frequency"])
    remove = num_experts - keep
    plans = {}
    for variant in VARIANTS:
        # One generator per random variant, Random(17 + seed), drawn layer by layer in order.
        rng = random.Random(seed + int(variant[-1])) if variant.startswith("random") else None
        layers = []
        for layer in range(num_layers):
            s = stats[layer]
            if variant in ("tn_prune", "tn_w_prune"):
                matrix = pairwise["raw" if variant == "tn_prune" else "whitened"][layer]
                redundancy = matrix.sum(1) / (num_experts - 1)
                layers.append({"removed": lowest(redundancy, remove)})
            elif variant == "frequency":
                layers.append({"removed": lowest(s["frequency"].tolist(), remove)})
            elif variant == "reap":
                layers.append({"removed": lowest(s["reap"].tolist(), remove)})
            elif variant.startswith("random_prune"):
                layers.append({"removed": sorted(rng.sample(range(num_experts), remove))})
            else:
                if variant in ("tn_merge", "tn_w_merge"):
                    groups = cluster(pairwise["raw" if variant == "tn_merge" else "whitened"][layer], keep)
                elif variant == "output_similarity":
                    mean = F.normalize(s["mean_output"].double(), dim=1)
                    groups = cluster((1 - mean @ mean.T).clamp_min(0).fill_diagonal_(0).numpy(), keep)
                else:
                    groups = random_groups(rng, num_experts, keep)
                adapter = OlmoeAdapter(model, layer)
                x = inputs[layer].to(adapter.device)
                layers.append({"groups": [align_group(adapter, x, g, s["frequency"]) for g in groups]})
        plans[variant] = {"operator": "merge" if is_merge(variant) else "prune", "layers": layers}
        print(f"[plans] {variant} built", flush=True)
    return plans


# ---------------------------------------------------------------------------------------------


def check_baseline(current: np.ndarray, previous_path: Path, label: str) -> None:
    previous = np.array(json.loads(previous_path.read_text()))
    correlation = float(np.corrcoef(current, previous)[0, 1])
    shift = float((current - previous).mean())
    print(f"[check] {label} vs {previous_path}: correlation {correlation:.6f}, mean diff {shift:+.2e}", flush=True)
    assert correlation > 0.999 and abs(shift) < 2e-3, "Token blocks differ from stage 1"


def run(config: dict) -> None:
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "PLAN.json"
    plan = {"config": config, "script_sha256": digest(Path(__file__)),
            "layer_removal_sha256": digest(Path(__file__).with_name("layer_removal.py"))}
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, "Config or scripts changed"
    else:
        save_json(plan_path, plan)

    model, tokenizer = load_model(config["model"], config["seed"])
    if config.get("device"):
        model.to(config["device"])
    num_layers = len(model.model.layers)
    corpora = load_corpora(config, model, tokenizer, output)

    # --- statistics -----------------------------------------------------------------------
    stats_path = output / "stats.pt"
    inputs = None
    if stats_path.exists():
        stats = torch.load(stats_path, weights_only=False)
    else:
        inputs = capture_moe_inputs(model, corpora["calibration"])
        stats = []
        for layer in range(num_layers):
            adapter = OlmoeAdapter(model, layer)
            stats.append(layer_statistics(adapter, inputs[layer].to(adapter.device),
                                          config.get("whitening_relative_floor", 1e-3)))
            print(f"[stats] layer {layer}", flush=True)
        save_tensor(stats_path, stats)

    # --- pairwise incremental rank --------------------------------------------------------
    pairwise = {"raw": [], "whitened": []}
    started = time.time()
    for layer in range(num_layers):
        adapter = OlmoeAdapter(model, layer)
        for kind in ("raw", "whitened"):
            path = output / "pairwise" / f"{kind}_layer{layer}.json"
            if not path.exists():
                matrices = {}
                for projection in PROJECTIONS:
                    factor = stats[layer]["l_down" if projection == "down_proj" else "l_hidden"].to(adapter.device)
                    matrices[projection] = [
                        adapter.weight(e, projection).detach().float() @ factor if kind == "whitened"
                        else adapter.weight(e, projection).detach().float() for e in range(adapter.num_experts)]
                path.parent.mkdir(parents=True, exist_ok=True)
                save_json(path, pairwise_incremental_rank(matrices).tolist())
                elapsed = time.time() - started
                print(f"[pairwise] layer {layer} {kind} done, {elapsed / 60:.1f} min", flush=True)
            pairwise[kind].append(np.array(json.loads(path.read_text())))
    for layer, directory in config.get("experiment0_dirs", {}).items():
        import csv
        rows = list(csv.DictReader((Path(directory) / "results.csv").open()))
        diff = [abs(pairwise["raw"][int(layer)][int(r["expert_first"]), int(r["expert_second"])]
                    - float(r[f"tn_delta_rank_normalized_eps_{PRIMARY_ERROR:.4f}"])) for r in rows]
        print(f"[check] layer {layer} pairwise vs Experiment 0: mean |diff| {np.mean(diff):.4f}, "
              f"max {np.max(diff):.4f}", flush=True)
        assert np.mean(diff) < 0.003, "Pairwise score does not reproduce Experiment 0"

    # --- plans, frozen before quality evaluation --------------------------------------------
    plans_dir = output / "plans"
    index_path = plans_dir / "index.json"
    if not index_path.exists():
        if inputs is None:
            inputs = capture_moe_inputs(model, corpora["calibration"])
        plans = build_plans(model, inputs, stats, pairwise, config["keep_experts"], config["seed"])
        plans_dir.mkdir(parents=True, exist_ok=True)
        for variant, value in plans.items():
            save_json(plans_dir / f"{variant}.json", value)
        save_json(index_path, {v: digest(plans_dir / f"{v}.json") for v in VARIANTS})
    index = json.loads(index_path.read_text())
    plans = {}
    for variant in VARIANTS:
        assert digest(plans_dir / f"{variant}.json") == index[variant], f"Plan {variant} changed"
        plans[variant] = json.loads((plans_dir / f"{variant}.json").read_text())
    del inputs
    print("[plans] frozen", flush=True)

    # --- dev ------------------------------------------------------------------------------
    def evaluate(variant, corpus):
        with compressed(model, plans.get(variant)):
            return evaluate_set(model, corpora[corpus], [], output / corpus / f"{variant}.json")

    dev = {"base": evaluate("base", "dev")}
    for check in config.get("baseline_checks", []):
        current = evaluate("base", check["corpus"])
        check_baseline(current, Path(check["file"]), check["corpus"])
    for variant in VARIANTS:
        dev[variant] = evaluate(variant, "dev")
        print(f"[dev] {variant}: ΔNLL {dev[variant].mean() - dev['base'].mean():+.4f}", flush=True)
    best = {family: min(members, key=lambda v: dev[v].mean())
            for family, members in FAMILIES.items() if family != "random"}
    save_json(output / "family_best.json", best)

    # --- test and downstream --------------------------------------------------------------
    chosen = ["base", *dict.fromkeys(best.values())]
    test_corpora = [c for c in ("wikitext_test", "c4") if c in corpora]
    test = {c: {v: evaluate(v, c) for v in chosen} for c in test_corpora}
    down = None
    if config.get("downstream"):
        d = config["downstream"]
        down = {}
        for variant in chosen:
            with compressed(model, plans.get(variant)):
                down[variant] = downstream(model, tokenizer, [], d["tasks"], d["batch_size"],
                                           output / "downstream" / f"{variant}.json", d.get("limit"))
    write_report(output, dev, best, test, down)


def write_report(output: Path, dev, best, test, down) -> None:
    base = dev["base"].mean()
    random_median = float(np.median([dev[v].mean() for v in FAMILIES["random"]]))
    lines = ["# Stage 2: one-shot expert removal and merging, 64 -> 40 experts in every layer (no healing)", "",
             "Pre-registered in PRUNING_PLAN.md (amendment 2026-10-10, stage 2 definitions).", "",
             "## Dev (WikiText validation) for all variants", "",
             "| Family | Variant | Operator | Dev ΔNLL |", "|---|---|---|---:|"]
    for family, members in FAMILIES.items():
        for variant in members:
            mark = " **(family best)**" if best.get(family) == variant else ""
            lines.append(f"| {family} | {variant}{mark} | {'merge' if is_merge(variant) else 'prune'} | "
                         f"{dev[variant].mean() - base:+.4f} |")
    lines += ["", f"Median of the 6 random variants: {random_median - base:+.4f}.", "",
              "## Test", "",
              "| Model | " + " | ".join(f"{c} ΔNLL" for c in test) + (" | Downstream mean (Δ pts) |" if down else " |"),
              "|---|" + "---:|" * len(test) + ("---:|" if down else "")]
    for variant in ["base", *dict.fromkeys(best.values())]:
        cells = [f"{test[c][variant].mean() - test[c]['base'].mean():+.4f}" for c in test]
        extra = f" | {down[variant]['mean']:.2f} ({down[variant]['mean'] - down['base']['mean']:+.2f})" if down else ""
        lines.append(f"| {variant} | " + " | ".join(cells) + extra + " |")
    summary = {"family_best": best, "random_median_dev_delta": random_median - base}
    lines.append("")
    for family in ("tn", "tn_whitened"):
        candidate, reference = best[family], best["cheap"]
        result = verdict(candidate, reference, test, down, dev[candidate].mean(), random_median)
        summary[f"{family}_vs_cheap"] = result
        contrasts = "; ".join(
            f"{c} {(test[c][candidate] - test[c][reference]).mean():+.4f} "
            f"[{paired_interval(test[c][candidate] - test[c][reference], 17)[0]:+.4f}, "
            f"{paired_interval(test[c][candidate] - test[c][reference], 17)[1]:+.4f}]" for c in test)
        lines.append(f"- **{family} ({candidate}) vs cheap ({reference}): {result}**. "
                     f"NLL difference (test): {contrasts}.")
    primary, secondary = summary["tn_vs_cheap"], summary["tn_whitened_vs_cheap"]
    # Stage 3 heals the surviving TN variant together with the cheap winner as its comparator.
    if primary in ("win", "tie"):
        arm = [best["tn"], best["cheap"]]
    elif secondary == "win":
        arm = [best["tn_whitened"], best["cheap"]]
    else:
        arm = []
    summary["stage3_expert_arm"] = arm
    lines += ["", "## Pre-registered decision", "",
              f"TN vs cheap: **{primary}**. TN-whitened vs cheap: **{secondary}**.",
              f"Expert-level variants for stage 3: **{', '.join(arm) if arm else 'none (killed)'}**.", "",
              "A loss also follows when the TN family best is worse on dev than the median random variant. "
              "TN-whitened can replace TN only with a win."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    save_json(output / "summary.json", summary)
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    run(yaml.safe_load(parser.parse_args().config.read_text()))


if __name__ == "__main__":
    main()
