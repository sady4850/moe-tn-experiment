"""Resumable single-layer screen. Reconstructed dense weights test quality only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .adapter import OlmoeAdapter, PROJECTIONS
from .compress import METHODS, decompose, reconstruct_expert
from .data import synthetic_blocks, text_blocks
from .evaluate import evaluate_nll
from .model import load_model
from .whitening import collect_moments, regularized_cholesky


def variant(name):
    base = re.sub(r"_r[1-9][0-9]*$", "", name)
    method, _, whitening = base.partition("_")
    if method not in METHODS or whitening not in ("", "none", "input"):
        raise ValueError(f"Unknown compression variant: {name}")
    if base != name and method != "tucker":
        raise ValueError("Explicit expert ranks only apply to Tucker")
    return method, whitening or "none"


def variant_expert_rank(name):
    match = re.search(r"_r([1-9][0-9]*)$", name)
    return int(match.group(1)) if match else None


def save_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_tensor(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


@contextmanager
def original_layer(adapter):
    originals = {p: torch.stack([adapter.weight(e, p).detach().cpu().clone()
                                for e in range(adapter.num_experts)]) for p in PROJECTIONS}
    try:
        yield originals
    finally:
        with torch.no_grad():
            for p, weights in originals.items():
                for e, weight in enumerate(weights):
                    adapter.weight(e, p).copy_(weight)


def evaluate_cached(model, blocks, path):
    rows = json.loads(path.read_text()) if path.exists() else []
    for index in range(len(rows), len(blocks)):
        result = evaluate_nll(model, blocks[index:index + 1])
        rows.append(result)
        save_json(path, rows)
        if (index + 1) % 8 == 0 or index + 1 == len(blocks):
            print(f"  evaluation {index + 1}/{len(blocks)}", flush=True)
    return np.array([row["nll"] for row in rows])


def paired_interval(deltas, seed):
    generator = np.random.default_rng(seed)
    means = np.array([generator.choice(deltas, len(deltas), replace=True).mean()
                      for _ in range(2000)])
    return np.quantile(means, [0.025, 0.975]).tolist()


def report(output: Path, baseline, results, seed):
    lines = ["# Single-layer weight compression screen", "",
             f"Baseline NLL: {baseline.mean():.8f}", "",
             "| Method | Target retained | Actual retained | ΔNLL | Paired block-bootstrap 95% CI |",
             "|---|---:|---:|---:|---|"]
    for row in results:
        delta = np.array(row["block_nll"]) - baseline
        interval = paired_interval(delta, seed)
        lines.append(f"| {row['method']} | {row['fraction']:.1%} | "
                     f"{row['retained_fraction']:.2%} | {delta.mean():+.8f} | "
                     f"[{interval[0]:+.8f}, {interval[1]:+.8f}] |")
    lines += ["", "## Direct paired comparisons", "",
              "Positive difference means Tucker has higher NLL (worse).", ""]
    for tn in (r for r in results if variant(r["method"])[0] == "tucker"):
        for other in results:
            if other is tn or other["fraction"] != tn["fraction"]:
                continue
            delta = np.array(tn["block_nll"]) - np.array(other["block_nll"])
            lo, hi = paired_interval(delta, seed)
            lines.append(f"- Tucker minus {other['method']} ({tn['method']}) at {tn['fraction']:.1%}: "
                         f"{delta.mean():+.8f} [{lo:+.8f}, {hi:+.8f}]")
    lines += ["", "Exploratory validation-set screen, fixed layer and text. Block-bootstrap intervals",
              "do not represent uncertainty across datasets/layers; neighboring blocks can be correlated.",
              "Variants ending in _input use the same shared input whitening; _none is the raw-weight control.",
              "No recovery training. Tucker uses HOSVD; rank policy is recorded per projection.",
              "Input whitening uses pooled uncentered second moments, not full MoE output sensitivity.",
              "Retained fractions count factor payload at model dtype for expert weights in this layer only.",
              "Dense reconstruction is used for NLL. No runtime memory saving or speedup is measured.",
              "Whitening inverses are absorbed into factors before rounding; calibration artifacts are not inference payload.",
              "A negative result does not exclude other tensor methods or recovery training.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def run(config, stop_after=None):
    for method in config["methods"]:
        variant(method)
    if any(variant(m)[1] == "input" for m in config["methods"]):
        d = config["data"]
        if d.get("calibration_blocks", 0) < 1:
            raise ValueError("Input whitening requires calibration blocks")
        if config["model"]["id"] != "__tiny_random_olmoe__" and (
            d.get("calibration_split") == d["evaluation_split"] or not d.get("calibration_split")
        ):
            raise ValueError("Calibration and evaluation must use different splits")
    if not config["fractions"] or any(not 0 < f <= 1 for f in config["fractions"]):
        raise ValueError("Fractions must be in (0, 1]")
    if config["data"].get("evaluation_offset_blocks", 0) < 0:
        raise ValueError("Evaluation offset must be nonnegative")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run(config, output, stop_after)


def _run(config, output, stop_after):
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    code_hash = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        code_hash.update(source.name.encode() + source.read_bytes())
    signature = {"config": config, "code_sha256": code_hash.hexdigest(),
                 "versions": {p: importlib.metadata.version(p)
                              for p in ("torch", "transformers", "datasets", "numpy")}}
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text())["signature"] != signature:
            raise ValueError("Config/code/packages changed: use a new output_dir")
    print("Loading model", flush=True)
    model, tokenizer = load_model(config["model"], seed)
    model.eval()
    adapter = OlmoeAdapter(model, config["model"]["layer"])
    if adapter.device.type == "mps":
        decomposition_device = torch.device("cpu")
    else:
        decomposition_device = adapter.device
    if not metadata_path.exists():
        save_json(metadata_path, {"signature": signature, "device": str(adapter.device),
                                 "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                                 "resolved_model_commit": getattr(model.config, "_commit_hash", None),
                                 "model_dtype": str(adapter.dtype),
                                 "model_parameters": sum(p.numel() for p in model.parameters())})
    block_path = output / "evaluation_blocks.pt"
    if block_path.exists():
        blocks = torch.load(block_path, weights_only=True)
    else:
        d = config["data"]
        offset = int(d.get("evaluation_offset_blocks", 0))
        count = int(d["evaluation_blocks"]) + offset
        if tokenizer is None:
            blocks = synthetic_blocks(seed=seed + 2, vocab_size=model.config.vocab_size,
                                      count=count, block_size=d["block_size"])
        else:
            blocks = text_blocks(tokenizer, d, d["evaluation_split"], count)
        blocks = blocks[offset:]
        save_tensor(block_path, blocks)
    print("Baseline NLL", flush=True)
    baseline = evaluate_cached(model, blocks, output / "baseline_blocks.json")
    cholesky = {}
    if any(variant(m)[1] == "input" for m in config["methods"]):
        print("Shared input calibration", flush=True)
        d = config["data"]
        calibration_path = output / "calibration_blocks.pt"
        if calibration_path.exists():
            calibration = torch.load(calibration_path, weights_only=True)
        else:
            if tokenizer is None:
                calibration = synthetic_blocks(seed=seed + 1, vocab_size=model.config.vocab_size,
                                               count=d["calibration_blocks"], block_size=d["block_size"])
            else:
                calibration = text_blocks(tokenizer, d, d["calibration_split"], d["calibration_blocks"])
            save_tensor(calibration_path, calibration)
        moment_path = output / "calibration_moments.pt"
        state = torch.load(moment_path, weights_only=True) if moment_path.exists() else None
        state = collect_moments(model, adapter, calibration, decomposition_device, state,
                                lambda s: save_tensor(moment_path, s))
        whitening_stats = {"expert_counts": state["expert_counts"].tolist(),
                           "hidden_rows": state["hidden_rows"], "down_rows": state["down_rows"]}
        for key, projections in (("hidden", ("gate_proj", "up_proj")), ("down", ("down_proj",))):
            moment = state[f"{key}_sum"].to(decomposition_device) / state[f"{key}_rows"]
            chol, stats = regularized_cholesky(moment, config.get("whitening_relative_floor", 1e-3))
            whitening_stats[key] = stats
            for projection in projections:
                cholesky[projection] = chol
        save_json(output / "whitening_stats.json", whitening_stats)
    results = []
    completed_this_run = 0
    for fraction in config["fractions"]:
        for method in config["methods"]:
            base_method, whitening = variant(method)
            trial = output / f"{method}_{fraction:g}"
            trial.mkdir(exist_ok=True)
            result_path = trial / "result.json"
            if result_path.exists():
                results.append(json.loads(result_path.read_text()))
                report(output, baseline, results, seed)
                continue
            print(f"Compressing {method}, retained fraction {fraction:g}", flush=True)
            started = time.monotonic()
            diagnostics = {}
            with original_layer(adapter) as originals, torch.no_grad():
                for projection in PROJECTIONS:
                    print(f"  {projection}", flush=True)
                    factor_path = trial / f"{projection}.pt"
                    if factor_path.exists():
                        artifact = torch.load(factor_path, weights_only=True)
                    else:
                        weights = originals[projection].to(decomposition_device, torch.float32)
                        artifact = decompose(weights, base_method, fraction, adapter.dtype,
                                             input_cholesky=cholesky.get(projection) if whitening == "input" else None,
                                             rank_policy=config.get("tucker_rank_policy", "energy"),
                                             expert_rank=variant_expert_rank(method))
                        save_tensor(factor_path, artifact)
                        del weights
                    info = {k: v for k, v in artifact.items() if k != "factors"}
                    # Transfer factors once per projection, not once per expert.
                    artifact["factors"] = {k: v.to(decomposition_device, torch.float32)
                                           for k, v in artifact["factors"].items()}
                    error, norm, weighted_error, weighted_norm = 0.0, 0.0, 0.0, 0.0
                    for expert in range(adapter.num_experts):
                        reconstructed = reconstruct_expert(artifact, expert, decomposition_device).to(adapter.dtype)
                        original = originals[projection][expert].to(decomposition_device, torch.float32)
                        error += (reconstructed.float() - original).square().sum().item()
                        norm += original.square().sum().item()
                        if projection in cholesky:
                            l = cholesky[projection]
                            weighted_error += ((reconstructed.float() - original) @ l).square().sum().item()
                            weighted_norm += (original @ l).square().sum().item()
                        adapter.weight(expert, projection).copy_(reconstructed)
                    info["relative_frobenius_error"] = (error / max(norm, 1e-30)) ** 0.5
                    if projection in cholesky:
                        info["relative_input_weighted_error"] = (weighted_error / max(weighted_norm, 1e-30)) ** 0.5
                    diagnostics[projection] = info
                    del artifact, reconstructed, original
                    save_json(trial / "projection_stats.json", diagnostics)
                nll = evaluate_cached(model, blocks, trial / "nll_blocks.json")
            stored = sum(v["stored_parameters"] for v in diagnostics.values())
            original_count = sum(v["original_parameters"] for v in diagnostics.values())
            result = {"method": method, "fraction": fraction,
                      "retained_fraction": stored / original_count,
                      "stored_bytes": sum(v["stored_bytes"] for v in diagnostics.values()),
                      "block_nll": nll.tolist(), "delta_nll": float((nll - baseline).mean()),
                      "seconds_this_attempt": time.monotonic() - started,
                      "projections": diagnostics}
            save_json(result_path, result)
            results.append(result)
            report(output, baseline, results, seed)
            print(f"  ΔNLL={result['delta_nll']:+.8f}; report: {output / 'REPORT.md'}", flush=True)
            completed_this_run += 1
            if stop_after is not None and completed_this_run >= stop_after:
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stop-after", type=int, help="Stop after N newly completed method/budget trials")
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    run(yaml.safe_load(Path(args.config).read_text()), args.stop_after)


if __name__ == "__main__":
    main()
