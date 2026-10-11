"""Stage 1 of PRUNING_PLAN.md: one-shot layer removal on OLMoE, no healing.

Phases, each resumable under output_dir:
  selections  TN (stage 0), TN-whitened and BI scores and sets, frozen before any evaluation
  dev         baseline, every set for small budgets, random sets otherwise, plus criterion sets
  test        WikiText test and C4 for the baseline and the criterion / oracle sets
  downstream  lm-evaluation-harness zero-shot tasks for the same sets
  report      pre-registered verdicts
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import itertools
import json
from math import comb
from pathlib import Path
import random
import time
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from moe_tn.adapter import PROJECTIONS, OlmoeAdapter
from moe_tn.compression_run import paired_interval, save_json, save_tensor
from moe_tn.data import synthetic_blocks, text_blocks
from moe_tn.evaluate import evaluate_nll
from moe_tn.model import load_model
from moe_tn.signals import effective_rank_from_singular_values
from moe_tn.whitening import collect_moments, regularized_cholesky
from layer_tn_density import ERRORS, MODES, PRIMARY, mode_singular_values

C4_URL = ("https://huggingface.co/datasets/allenai/c4/resolve/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/"
          "en/c4-validation.00000-of-00008.json.gz")
CRITERIA = ("tn", "tn_whitened", "bi")
DOWNSTREAM_MARGIN = 0.5  # accuracy points


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(layers) -> str:
    return "base" if not layers else "drop_" + "_".join(str(layer) for layer in sorted(layers))


class _Skip(torch.nn.Module):
    """Stands in for a decoder layer: the residual stream passes through unchanged."""

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


@contextmanager
def removed(model, layers):
    stack = model.model.layers
    originals = {layer: stack[layer] for layer in layers}
    try:
        for layer in layers:
            stack[layer] = _Skip()
        yield
    finally:
        for layer, module in originals.items():
            stack[layer] = module


# ---------------------------------------------------------------------------------------------
# Scores


def density(tensors: dict[str, torch.Tensor]) -> dict:
    """Mean normalized effective mode rank of [expert, out, in] tensors (stage 0 definition)."""
    normalized = {}
    for projection, tensor in tensors.items():
        sizes = dict(zip(MODES, tensor.shape))
        for mode, values in mode_singular_values(tensor).items():
            for error in ERRORS:
                rank = effective_rank_from_singular_values(values.cpu(), error)
                normalized[f"{projection}/{mode}/{error:.2f}"] = rank / sizes[mode]
    result = {"normalized_rank": normalized}
    for error in ERRORS:
        result[f"density_{error:.2f}"] = float(np.mean(
            [normalized[f"{p}/{m}/{error:.2f}"] for p in tensors for m in MODES]))
    return result


def raw_tensors(adapter) -> dict[str, torch.Tensor]:
    return {p: torch.stack([adapter.weight(e, p).float() for e in range(adapter.num_experts)])
            for p in PROJECTIONS}


def whitened_tensors(model, adapter, calibration, floor) -> tuple[dict[str, torch.Tensor], dict]:
    state = collect_moments(model, adapter, calibration, adapter.device)
    stats, factors = {}, {}
    for name, projections in (("hidden", ("gate_proj", "up_proj")), ("down", ("down_proj",))):
        moment = state[f"{name}_sum"].to(adapter.device) / state[f"{name}_rows"]
        chol, stats[name] = regularized_cholesky(moment, floor)
        for projection in projections:
            factors[projection] = chol
    tensors = {p: torch.stack([adapter.weight(e, p).float() @ factors[p] for e in range(adapter.num_experts)])
               for p in PROJECTIONS}
    return tensors, stats


@torch.inference_mode()
def block_influence(model, blocks) -> list[float]:
    """ShortGPT BI: 1 - mean token cosine between each decoder layer's input and output."""
    layers = model.model.layers
    device = next(model.parameters()).device
    sums = torch.zeros(len(layers), dtype=torch.float64)
    count = 0

    def hook(index):
        def record(module, args, kwargs, output):
            x = args[0] if args else kwargs["hidden_states"]
            y = output[0] if isinstance(output, tuple) else output
            sums[index] += F.cosine_similarity(x.float(), y.float(), dim=-1).double().sum().cpu()
        return record

    handles = [layer.register_forward_hook(hook(i), with_kwargs=True) for i, layer in enumerate(layers)]
    try:
        for index in range(len(blocks)):
            model(input_ids=blocks[index:index + 1].to(device), use_cache=False)
            count += blocks.shape[1]
    finally:
        for handle in handles:
            handle.remove()
    return (1.0 - sums / count).tolist()


def lowest_first(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda layer: (scores[layer], layer))


# ---------------------------------------------------------------------------------------------
# Data


def c4_blocks(tokenizer, documents_path: Path, size: int) -> torch.Tensor:
    documents = json.loads(documents_path.read_text())
    wanted = {d["row"]: d for d in documents}
    windows = {}
    for attempt in range(6):
        try:
            with urllib.request.urlopen(C4_URL, timeout=120) as response, \
                    gzip.GzipFile(fileobj=response) as stream:
                for row_index, line in enumerate(stream):
                    if row_index not in wanted:
                        continue
                    row, document = json.loads(line), wanted[row_index]
                    assert hashlib.sha256(row["text"].encode()).hexdigest() == document["text_sha256"]
                    ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
                    assert len(ids) == document["document_tokens"], row_index
                    windows[row_index] = ids[document["window_start"]:document["window_start"] + size]
                    if len(windows) == len(wanted):
                        break
            break
        except OSError as error:
            print(f"C4 download retry {attempt + 1}: {error}", flush=True)
            time.sleep(30 * (attempt + 1))
    assert len(windows) == len(wanted), "C4 documents missing"
    return torch.tensor([windows[d["row"]] for d in documents], dtype=torch.long)


def load_corpora(config, model, tokenizer, output: Path) -> dict[str, torch.Tensor]:
    path = output / "token_blocks.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    d, seed = config["data"], config["seed"]
    if tokenizer is None:
        def synthetic(offset, count):
            return synthetic_blocks(seed=seed + offset, vocab_size=model.config.vocab_size,
                                    count=count, block_size=d["block_size"])
        corpora = {"calibration": synthetic(1, d["calibration_blocks"]), "dev": synthetic(2, d["dev_blocks"]),
                   "wikitext_test": synthetic(3, d["test_blocks"])}
    else:
        corpora = {
            "calibration": text_blocks(tokenizer, d, d["calibration_split"], d["calibration_blocks"]),
            "dev": text_blocks(tokenizer, d, d["dev_split"], d["dev_blocks"]),
            "wikitext_test": text_blocks(tokenizer, d, d["test_split"],
                                         d["test_offset_blocks"] + d["test_blocks"])[d["test_offset_blocks"]:],
        }
        if config.get("c4_documents"):
            corpora["c4"] = c4_blocks(tokenizer, Path(config["c4_documents"]), d["block_size"])
    save_tensor(path, corpora)
    return corpora


# ---------------------------------------------------------------------------------------------
# Evaluation


def evaluate_set(model, blocks, layers, path: Path) -> np.ndarray:
    if path.exists():
        values = json.loads(path.read_text())
        if len(values) == len(blocks):
            return np.array(values)
    with removed(model, layers):
        values = [evaluate_nll(model, blocks[i:i + 1])["nll"] for i in range(len(blocks))]
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, values)
    return np.array(values)


def downstream(model, tokenizer, layers, tasks: dict[str, str], batch_size: int, path: Path,
               limit: int | None = None) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    with removed(model, layers):
        output = simple_evaluate(model=HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size),
                                 tasks=list(tasks), num_fewshot=0, limit=limit, log_samples=False, random_seed=17,
                                 numpy_random_seed=17, torch_random_seed=17, fewshot_random_seed=17)
    scores = {task: 100 * float(output["results"][task][f"{metric},none"]) for task, metric in tasks.items()}
    result = {"scores": scores, "mean": float(np.mean(list(scores.values())))}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, result)
    return result


def dev_sets(num_layers: int, budgets, exhaustive_limit: int, random_sets: int, seed: int) -> dict[int, list]:
    sets = {}
    for k in budgets:
        if comb(num_layers, k) <= exhaustive_limit:
            sets[k] = [list(c) for c in itertools.combinations(range(num_layers), k)]
        else:
            rng, chosen = random.Random(seed), set()
            while len(chosen) < random_sets:
                chosen.add(tuple(sorted(rng.sample(range(num_layers), k))))
            sets[k] = [list(c) for c in sorted(chosen)]
    return sets


# ---------------------------------------------------------------------------------------------
# Verdicts


def better(a: str, b: str, test: dict, down: dict | None) -> bool:
    """Pre-registered margin: lower NLL on every test corpus with the 95% CI below zero,
    and mean downstream accuracy not lower by more than DOWNSTREAM_MARGIN points."""
    if a == b:
        return False
    for corpus in test:
        delta = test[corpus][a] - test[corpus][b]
        if not (delta.mean() < 0 and paired_interval(delta, 17)[1] < 0):
            return False
    return down is None or down[a]["mean"] >= down[b]["mean"] - DOWNSTREAM_MARGIN


def verdict(candidate: str, reference: str, test, down, dev_nll: float, dev_median: float) -> str:
    if candidate == reference:
        return "tie (same set)"
    if better(candidate, reference, test, down):
        return "win"
    if better(reference, candidate, test, down) or dev_nll > dev_median:
        return "lose"
    return "tie"


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------------------------


def run(config: dict) -> None:
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    plan = {"config": config, "script_sha256": digest(Path(__file__))}
    if config.get("tn_density_file"):
        plan["tn_density_sha256"] = digest(Path(config["tn_density_file"]))
    plan_path = output / "PLAN.json"
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, "Config, script or stage 0 file changed"
    else:
        save_json(plan_path, plan)

    model, tokenizer = load_model(config["model"], config["seed"])
    if config.get("device"):
        model.to(config["device"])
    num_layers = len(model.model.layers)
    corpora = load_corpora(config, model, tokenizer, output)
    budgets = config["budgets"]

    # --- selections, frozen before any quality evaluation -----------------------------------
    selections_path = output / "selections.json"
    if not selections_path.exists():
        floor = config.get("whitening_relative_floor", 1e-3)
        raw, whitened, whitening_stats = [], [], []
        for layer in range(num_layers):
            adapter = OlmoeAdapter(model, layer)
            raw.append(density(raw_tensors(adapter))[f"density_{PRIMARY:.2f}"])
            tensors, stats = whitened_tensors(model, adapter, corpora["calibration"], floor)
            whitened.append(density(tensors)[f"density_{PRIMARY:.2f}"])
            whitening_stats.append(stats)
            print(f"[selections] layer {layer}: raw {raw[-1]:.4f}, whitened {whitened[-1]:.4f}", flush=True)
        if config.get("tn_density_file"):
            stage0 = json.loads(Path(config["tn_density_file"]).read_text())
            tn = [stage0["layers"][str(layer)][f"density_{PRIMARY:.2f}"] for layer in range(num_layers)]
        else:
            tn = raw
        scores = {"tn": tn, "tn_whitened": whitened, "bi": block_influence(model, corpora["calibration"]),
                  "tn_recomputed_on_device": raw}
        sets = {c: {str(k): sorted(lowest_first(scores[c])[:k]) for k in budgets} for c in CRITERIA}
        save_json(selections_path, {"scores": scores, "sets": sets, "whitening_stats": whitening_stats,
                                    "rule": "one-shot top-k lowest score, ties by layer index"})
    selections = json.loads(selections_path.read_text())
    sets = selections["sets"]
    print("[selections] frozen:", json.dumps(sets), flush=True)

    # Original-model NLL must reproduce earlier runs on the shared token blocks.
    for check in config.get("baseline_checks", []):
        previous = np.array([row["nll"] for row in json.loads(Path(check["file"]).read_text())])
        current = evaluate_set(model, corpora[check["corpus"]], [], output / check["corpus"] / "base.json")
        current = current[check["start"]:check["end"]]
        correlation = float(np.corrcoef(current, previous)[0, 1])
        shift = float((current - previous).mean())
        print(f"[check] {check['corpus']} vs {check['file']}: block correlation {correlation:.6f}, "
              f"mean diff {shift:+.2e}, max |diff| {np.abs(current - previous).max():.2e}", flush=True)
        # Wrong tokens decorrelate block NLLs; BF16 kernels on another GPU type add only ~3e-3 noise per block.
        assert correlation > 0.999 and abs(shift) < 2e-3, "Token blocks differ from earlier runs"

    # --- dev ------------------------------------------------------------------------------
    dev = {"base": evaluate_set(model, corpora["dev"], [], output / "dev" / "base.json")}
    candidates = dev_sets(num_layers, budgets, config["exhaustive_limit"], config["random_sets"], config["seed"])
    for k in budgets:
        for criterion in CRITERIA:
            if sets[criterion][str(k)] not in candidates[k]:
                candidates[k].append(sets[criterion][str(k)])
        for index, layers in enumerate(candidates[k]):
            dev[key(layers)] = evaluate_set(model, corpora["dev"], layers, output / "dev" / f"{key(layers)}.json")
            if (index + 1) % 20 == 0:
                print(f"[dev] k={k}: {index + 1}/{len(candidates[k])}", flush=True)
    oracle = {str(k): min(candidates[k], key=lambda s: dev[key(s)].mean()) for k in budgets}
    save_json(output / "oracle.json", oracle)

    # --- test and downstream --------------------------------------------------------------
    chosen = {"base": []}
    for k in budgets:
        for criterion in (*CRITERIA, "oracle"):
            layers = oracle[str(k)] if criterion == "oracle" else sets[criterion][str(k)]
            chosen[key(layers)] = layers
    test_corpora = [c for c in ("wikitext_test", "c4") if c in corpora]
    test = {c: {name: evaluate_set(model, corpora[c], layers, output / c / f"{name}.json")
                for name, layers in chosen.items()} for c in test_corpora}
    down = None
    if config.get("downstream"):
        d = config["downstream"]
        down = {name: downstream(model, tokenizer, layers, d["tasks"], d["batch_size"],
                                 output / "downstream" / f"{name}.json", d.get("limit"))
                for name, layers in chosen.items()}
    write_report(output, config, selections, dev, candidates, oracle, test, down, num_layers)


def write_report(output, config, selections, dev, candidates, oracle, test, down, num_layers) -> None:
    budgets, primary = config["budgets"], config["primary_k"]
    scores, sets = selections["scores"], selections["sets"]
    base = dev["base"].mean()
    single = [dev[key([layer])].mean() - base for layer in range(num_layers)]
    lines = ["# Stage 1: one-shot layer removal (no healing)", "",
             "Pre-registered in PRUNING_PLAN.md. Lower score = removed first.", "",
             "## Layer scores and single-layer removal damage (dev ΔNLL)", "",
             "| Layer | TN density | TN-whitened | BI | Dev ΔNLL if removed alone |", "|---:|---:|---:|---:|---:|"]
    for layer in range(num_layers):
        lines.append(f"| {layer} | {scores['tn'][layer]:.4f} | {scores['tn_whitened'][layer]:.4f} | "
                     f"{scores['bi'][layer]:.4f} | {single[layer]:+.4f} |")
    lines += ["", "Spearman of score with single-removal damage (positive = useful ordering): " + ", ".join(
        f"{c} {spearman(scores[c], single):+.3f}" for c in CRITERIA) + ".", ""]

    summary = {"verdicts": {}, "sets": {}}
    for k in budgets:
        evaluated = np.array([dev[key(s)].mean() for s in candidates[k]])
        median = float(np.median(evaluated))
        exhaustive = comb(num_layers, k) <= config["exhaustive_limit"]
        lines += [f"## k = {k}{' (primary)' if k == primary else ''}", "",
                  f"Dev reference: {len(evaluated)} {'(all)' if exhaustive else 'random + criterion'} sets, "
                  f"median ΔNLL {median - base:+.4f}, best {evaluated.min() - base:+.4f} "
                  f"(oracle {oracle[str(k)]}).", "",
                  "| Criterion | Set | Dev ΔNLL | Dev percentile | " +
                  " | ".join(f"{c} ΔNLL" for c in test) + (" | Downstream mean (Δ pts) |" if down else " |"),
                  "|---|---|---:|---:|" + "---:|" * len(test) + ("---:|" if down else "")]
        rows = {**{c: sets[c][str(k)] for c in CRITERIA}, "oracle": oracle[str(k)]}
        for criterion, layers in rows.items():
            name = key(layers)
            percentile = float((evaluated <= dev[name].mean()).mean())
            cells = [f"{test[c][name].mean() - test[c]['base'].mean():+.4f}" for c in test]
            extra = (f" | {down[name]['mean']:.2f} ({down[name]['mean'] - down['base']['mean']:+.2f})"
                     if down else "")
            lines.append(f"| {criterion} | {layers} | {dev[name].mean() - base:+.4f} | {percentile:.0%} | "
                         + " | ".join(cells) + extra + " |")
        bi = key(sets["bi"][str(k)])
        lines.append("")
        for criterion in ("tn", "tn_whitened"):
            name = key(sets[criterion][str(k)])
            result = verdict(name, bi, test, down, dev[name].mean(), median)
            summary["verdicts"][f"{criterion}_vs_bi_k{k}"] = result
            contrasts = "; ".join(
                f"{c} {(test[c][name] - test[c][bi]).mean():+.4f} "
                f"[{paired_interval(test[c][name] - test[c][bi], 17)[0]:+.4f}, "
                f"{paired_interval(test[c][name] - test[c][bi], 17)[1]:+.4f}]" for c in test) if name != bi else "same set"
            lines.append(f"- **{criterion} vs BI: {result}**. NLL difference (test): {contrasts}.")
        summary["sets"][str(k)] = rows
        lines.append("")
    tn_primary = summary["verdicts"][f"tn_vs_bi_k{primary}"]
    tnw_primary = summary["verdicts"][f"tn_whitened_vs_bi_k{primary}"]
    to_heal = ("tn" if tn_primary in ("win", "tie") else
               "tn_whitened" if tnw_primary == "win" else None)
    summary["primary"] = {"tn": tn_primary, "tn_whitened": tnw_primary, "stage3_tn_arm": to_heal}
    lines += ["## Pre-registered decision", "",
              f"Primary (k = {primary}) TN vs BI: **{tn_primary}**. TN-whitened vs BI: **{tnw_primary}**.",
              f"Layer-level TN arm for stage 3: **{to_heal or 'none (killed)'}**.", "",
              "Margin: lower NLL on every test corpus with the paired 95% CI below zero, and mean downstream "
              f"accuracy not lower by more than {DOWNSTREAM_MARGIN} points. A loss also follows when the TN set is "
              "worse than the dev median. TN-whitened can replace TN only with a win."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    save_json(output / "summary.json", summary)
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    run(yaml.safe_load(parser.parse_args().config.read_text()))


if __name__ == "__main__":
    main()
