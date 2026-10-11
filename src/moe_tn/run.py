from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

from .adapter import OlmoeAdapter
from .analyze import analyze
from .config import load_config
from .data import build_data
from .evaluate import evaluate_nll
from .model import load_model
from .signals import capture_layer_inputs, compute_signals, sample_pairs


GENERATED_FILES = (
    "metadata.json",
    "pairs.json",
    "signals.json",
    "baseline.json",
    "damages.jsonl",
    "results.json",
    "results.csv",
    "summary.json",
    "REPORT.md",
)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metadata(config: dict[str, Any], model: torch.nn.Module) -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda_version": torch.version.cuda,
        }
    return {
        "config": config,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": gpu,
    }


def combine_rows(
    signals: list[dict[str, Any]], damages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_pair = {
        (row["expert_first"], row["expert_second"]): row for row in damages
    }
    combined = []
    for signal in signals:
        key = (signal["expert_first"], signal["expert_second"])
        if key in by_pair:
            combined.append({**signal, **by_pair[key]})
    return combined


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    def number(value: float | None, digits: int = 4) -> str:
        return "undefined" if value is None else f"{value:.{digits}f}"

    metrics = summary["metrics"]
    lines = [
        "# MoE TN-proxy experiment report",
        "",
        f"Pairs evaluated: {summary['num_pairs']}",
        f"Primary reconstruction error: {summary['primary_reconstruction_error']:.1%}",
        "",
        "| Signal | Spearman vs merge damage | 95% bootstrap CI | top-k mean ΔNLL |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        "tn_incremental_rank",
        "activation_cosine",
        "weight_cosine",
        "router_usage_sum",
        "router_probability_sum",
        "router_co_routing",
    ):
        metric = metrics[name]
        low, high = metric["spearman_bootstrap_95_ci"]
        lines.append(
            f"| {name} | {number(metric['spearman_with_damage'])} | "
            f"[{number(low)}, {number(high)}] | {metric['top_k_mean_delta_nll']:.6g} |"
        )
    reference = metrics["random_pair_reference"]
    lines.extend(
        [
            "",
            f"Random-pair mean ΔNLL: {reference['mean_delta_nll']:.6g}",
            "",
            summary["interpretation"],
            "",
            "This tests an effective-rank TN proxy for expert-pair selection, not tensorization in general.",
        ]
    )
    lines.extend(
        [
            "",
            "## TN error-budget sensitivity",
            "",
            "| Relative reconstruction error | Spearman vs merge damage | 95% bootstrap CI |",
            "|---:|---:|---:|",
        ]
    )
    for name, metric in metrics.items():
        prefix = "tn_incremental_rank_eps_"
        if not name.startswith(prefix):
            continue
        error = float(name.removeprefix(prefix))
        low, high = metric["spearman_bootstrap_95_ci"]
        lines.append(
            f"| {error:.1%} | {number(metric['spearman_with_damage'])} | "
            f"[{number(low)}, {number(high)}] |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(config: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in GENERATED_FILES:
            target = output_dir / name
            if target.exists():
                target.unlink()

    set_seed(int(config["seed"]))
    started = time.perf_counter()
    print("[1/6] Loading model and tokenizer", flush=True)
    model, tokenizer = load_model(config["model"], int(config["seed"]))
    model.eval()
    adapter = OlmoeAdapter(model, int(config["model"]["layer"]))
    print(
        f"      {sum(p.numel() for p in model.parameters()):,} parameters; "
        f"{adapter.num_experts} experts; device={adapter.device}; dtype={adapter.dtype}",
        flush=True,
    )

    print("[2/6] Preparing fixed calibration/evaluation blocks", flush=True)
    calibration, evaluation = build_data(config, model, tokenizer)
    write_json(output_dir / "metadata.json", metadata(config, model))

    pair_path = output_dir / "pairs.json"
    if pair_path.exists():
        pairs = [tuple(pair) for pair in read_json(pair_path)]
    else:
        pairs = sample_pairs(
            adapter.num_experts,
            int(config["experiment"]["num_pairs"]),
            int(config["seed"]),
        )
        write_json(pair_path, pairs)

    signal_path = output_dir / "signals.json"
    if signal_path.exists():
        print("[3/6] Reusing saved signals", flush=True)
        signals = read_json(signal_path)
    else:
        print("[3/6] Capturing common expert inputs and computing signals", flush=True)
        hidden = capture_layer_inputs(
            model,
            adapter,
            calibration,
            int(config["experiment"]["calibration_tokens"]),
        )
        signals = compute_signals(
            adapter,
            pairs,
            hidden,
            [float(value) for value in config["experiment"]["reconstruction_errors"]],
            int(config["experiment"]["activation_chunk_tokens"]),
        )
        write_json(signal_path, signals)
        del hidden

    baseline_path = output_dir / "baseline.json"
    if baseline_path.exists():
        print("[4/6] Reusing saved baseline", flush=True)
        baseline = read_json(baseline_path)
    else:
        print("[4/6] Evaluating unmodified baseline", flush=True)
        baseline = evaluate_nll(
            model, evaluation, int(config["experiment"]["eval_batch_size"])
        )
        write_json(baseline_path, baseline)
    print(f"      baseline NLL={baseline['nll']:.6f}", flush=True)

    print("[5/6] Measuring actual merge damage", flush=True)
    damage_path = output_dir / "damages.jsonl"
    damages = read_jsonl(damage_path)
    completed = {(row["expert_first"], row["expert_second"]) for row in damages}
    alpha = float(config["experiment"]["merge_alpha"])
    for index, (first, second) in enumerate(pairs, start=1):
        if (first, second) in completed:
            continue
        pair_started = time.perf_counter()
        with adapter.merged_pair(first, second, alpha=alpha):
            merged = evaluate_nll(
                model, evaluation, int(config["experiment"]["eval_batch_size"])
            )
        row = {
            "expert_first": first,
            "expert_second": second,
            "merged_nll": merged["nll"],
            "merged_perplexity": merged["perplexity"],
            "delta_nll": merged["nll"] - baseline["nll"],
            "evaluation_tokens": merged["tokens"],
            "seconds": time.perf_counter() - pair_started,
        }
        append_jsonl(damage_path, row)
        damages.append(row)
        print(
            f"      pair {index}/{len(pairs)} ({first},{second}) "
            f"ΔNLL={row['delta_nll']:+.6g} in {row['seconds']:.1f}s",
            flush=True,
        )

    print("[6/6] Computing ranking statistics", flush=True)
    combined = combine_rows(signals, damages)
    if len(combined) != len(pairs):
        raise RuntimeError(f"Only {len(combined)}/{len(pairs)} pairs have complete results")
    write_json(output_dir / "results.json", combined)
    write_csv(output_dir / "results.csv", combined)
    summary = analyze(
        combined,
        float(config["experiment"]["primary_reconstruction_error"]),
        int(config["experiment"]["bootstrap_samples"]),
        int(config["seed"]),
    )
    summary["wall_seconds"] = time.perf_counter() - started
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "REPORT.md", summary)
    print(f"Done in {summary['wall_seconds'] / 60:.1f} min: {output_dir / 'REPORT.md'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument(
        "--overwrite", action="store_true", help="Discard known outputs and start from scratch"
    )
    args = parser.parse_args()
    run_experiment(load_config(args.config), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
