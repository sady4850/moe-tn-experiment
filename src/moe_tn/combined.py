from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .analyze import spearman


BASE_PREDICTORS = {
    "activation_cosine": lambda row: -row["activation_cosine"],
    "weight_cosine": lambda row: -row["weight_cosine"],
    "router_usage_sum": lambda row: (
        row["router_usage_first"] + row["router_usage_second"]
    ),
    "router_probability_sum": lambda row: (
        row["router_mean_probability_first"]
        + row["router_mean_probability_second"]
    ),
    "router_co_routing": lambda row: -row["router_co_routing"],
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _pair(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["expert_first"]), int(row["expert_second"])


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _mean_layer_spearman(
    predictors: list[np.ndarray], damages: list[np.ndarray], indices: np.ndarray | None = None
) -> float:
    estimates = []
    for predictor, damage in zip(predictors, damages, strict=True):
        if indices is not None:
            predictor = predictor[indices]
            damage = damage[indices]
        estimate = spearman(predictor, damage)
        if math.isfinite(estimate):
            estimates.append(estimate)
    return float(np.mean(estimates)) if estimates else math.nan


def _cluster_bootstrap_ci(
    predictors: list[np.ndarray],
    damages: list[np.ndarray],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    estimates = []
    pair_count = len(damages[0])
    for _ in range(samples):
        indices = generator.integers(0, pair_count, pair_count)
        estimate = _mean_layer_spearman(predictors, damages, indices)
        if math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return math.nan, math.nan
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _cluster_bootstrap_difference_ci(
    baseline_predictors: list[np.ndarray],
    tn_predictors: list[np.ndarray],
    damages: list[np.ndarray],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    estimates = []
    pair_count = len(damages[0])
    for _ in range(samples):
        indices = generator.integers(0, pair_count, pair_count)
        baseline = _mean_layer_spearman(baseline_predictors, damages, indices)
        tn = _mean_layer_spearman(tn_predictors, damages, indices)
        difference = baseline - tn
        if math.isfinite(difference):
            estimates.append(difference)
    if not estimates:
        return math.nan, math.nan
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def analyze_layers(
    layers: dict[str, list[dict[str, Any]]], bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    if len(layers) < 2:
        raise ValueError("Combined analysis requires at least two layers")

    ordered_layers: dict[str, list[dict[str, Any]]] = {}
    expected_pairs: list[tuple[int, int]] | None = None
    for name, rows in layers.items():
        by_pair = {_pair(row): row for row in rows}
        pairs = sorted(by_pair)
        if expected_pairs is None:
            expected_pairs = pairs
        elif pairs != expected_pairs:
            raise ValueError(f"Layer {name!r} does not contain the same expert pairs")
        ordered_layers[name] = [by_pair[pair] for pair in pairs]

    assert expected_pairs is not None
    first_rows = next(iter(ordered_layers.values()))
    tn_fields = sorted(
        key
        for key in first_rows[0]
        if key.startswith("tn_delta_rank_normalized_eps_")
        and all(key in row for rows in ordered_layers.values() for row in rows)
    )
    predictor_functions = dict(BASE_PREDICTORS)
    for field in tn_fields:
        error = field.removeprefix("tn_delta_rank_normalized_eps_")
        predictor_functions[f"tn_incremental_rank_eps_{error}"] = (
            lambda row, field=field: row[field]
        )

    damages = {
        name: np.asarray([row["delta_nll"] for row in rows], dtype=np.float64)
        for name, rows in ordered_layers.items()
    }
    top_k = max(1, min(5, len(expected_pairs) // 4))
    metrics: dict[str, Any] = {}
    stored_predictors: dict[str, list[np.ndarray]] = {}
    for offset, (metric_name, predictor_function) in enumerate(
        predictor_functions.items()
    ):
        predictor_by_layer = {
            name: np.asarray(
                [predictor_function(row) for row in rows], dtype=np.float64
            )
            for name, rows in ordered_layers.items()
        }
        predictors = list(predictor_by_layer.values())
        stored_predictors[metric_name] = predictors
        damage_arrays = list(damages.values())
        per_layer: dict[str, Any] = {}
        relative_top_k = []
        for name in ordered_layers:
            predictor = predictor_by_layer[name]
            damage = damages[name]
            selected = np.argsort(predictor, kind="stable")[:top_k]
            random_mean = float(damage.mean())
            top_k_mean = float(damage[selected].mean())
            relative_top_k.append(top_k_mean - random_mean)
            per_layer[name] = {
                "spearman_with_damage": _finite_or_none(spearman(predictor, damage)),
                "random_pair_mean_delta_nll": random_mean,
                "top_k_mean_delta_nll": top_k_mean,
                "top_k_minus_random_mean_delta_nll": top_k_mean - random_mean,
                "selected_pairs": [list(expected_pairs[index]) for index in selected],
            }
        aggregate = _mean_layer_spearman(predictors, damage_arrays)
        low, high = _cluster_bootstrap_ci(
            predictors, damage_arrays, bootstrap_samples, seed + offset
        )
        metrics[metric_name] = {
            "mean_within_layer_spearman": _finite_or_none(aggregate),
            "cluster_bootstrap_95_ci": [
                _finite_or_none(low),
                _finite_or_none(high),
            ],
            "mean_top_k_minus_random_delta_nll": float(np.mean(relative_top_k)),
            "top_k": top_k,
            "per_layer": per_layer,
        }

    primary_name = "tn_incremental_rank_eps_0.1000"
    if primary_name in metrics:
        metrics["tn_incremental_rank"] = metrics[primary_name]
    comparisons: dict[str, Any] = {}
    if primary_name in stored_predictors:
        tn_predictors = stored_predictors[primary_name]
        tn_aggregate = metrics[primary_name]["mean_within_layer_spearman"]
        assert tn_aggregate is not None
        for offset, baseline_name in enumerate(BASE_PREDICTORS):
            baseline_aggregate = metrics[baseline_name]["mean_within_layer_spearman"]
            assert baseline_aggregate is not None
            low, high = _cluster_bootstrap_difference_ci(
                stored_predictors[baseline_name],
                tn_predictors,
                list(damages.values()),
                bootstrap_samples,
                seed + 100 + offset,
            )
            comparisons[baseline_name] = {
                "mean_spearman_difference_baseline_minus_tn": (
                    baseline_aggregate - tn_aggregate
                ),
                "cluster_bootstrap_95_ci": [
                    _finite_or_none(low),
                    _finite_or_none(high),
                ],
            }
    return {
        "interpretation": (
            "All predictors are oriented so lower means safer to merge. Positive "
            "Spearman therefore means the signal tracks greater merge damage. The "
            "primary aggregate is the mean of within-layer correlations; it does not "
            "pool incomparable damage scales across layers."
        ),
        "bootstrap_note": (
            "The cluster bootstrap resamples the same expert-pair identities in every "
            "layer. Its interval quantifies pair-sampling uncertainty for these fixed "
            "layers, not uncertainty over all model layers."
        ),
        "layers": list(ordered_layers),
        "num_pairs_per_layer": len(expected_pairs),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "metrics": metrics,
        "comparisons_vs_primary_tn": comparisons,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    def number(value: float | None, digits: int = 4) -> str:
        return "undefined" if value is None else f"{value:.{digits}f}"

    metrics = summary["metrics"]
    lines = [
        "# Cross-layer MoE merge-signal report",
        "",
        f"Layers: {', '.join(summary['layers'])}",
        f"Pairs per layer: {summary['num_pairs_per_layer']}",
        "",
        "| Signal | Mean within-layer Spearman | Cluster-bootstrap 95% CI | Mean top-5 ΔNLL vs random |",
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
        low, high = metric["cluster_bootstrap_95_ci"]
        lines.append(
            f"| {name} | {number(metric['mean_within_layer_spearman'])} | "
            f"[{number(low)}, {number(high)}] | "
            f"{metric['mean_top_k_minus_random_delta_nll']:+.6g} |"
        )

    lines.extend(
        [
            "",
            "## Direct comparison with primary TN signal",
            "",
            "Positive values favor the baseline over TN.",
            "",
            "| Baseline | Mean Spearman difference | Cluster-bootstrap 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for name, comparison in summary["comparisons_vs_primary_tn"].items():
        low, high = comparison["cluster_bootstrap_95_ci"]
        lines.append(
            f"| {name} | "
            f"{number(comparison['mean_spearman_difference_baseline_minus_tn'])} | "
            f"[{number(low)}, {number(high)}] |"
        )

    lines.extend(
        [
            "",
            "## Per-layer Spearman correlations",
            "",
            "| Signal | " + " | ".join(summary["layers"]) + " |",
            "|---|" + "---:|" * len(summary["layers"]),
        ]
    )
    for name in (
        "tn_incremental_rank",
        "activation_cosine",
        "weight_cosine",
        "router_usage_sum",
        "router_probability_sum",
        "router_co_routing",
    ):
        values = [
            number(metrics[name]["per_layer"][layer]["spearman_with_damage"])
            for layer in summary["layers"]
        ]
        lines.append(f"| {name} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## TN reconstruction-error sensitivity",
            "",
            "| Relative reconstruction error | Mean within-layer Spearman | Cluster-bootstrap 95% CI |",
            "|---:|---:|---:|",
        ]
    )
    for name, metric in metrics.items():
        prefix = "tn_incremental_rank_eps_"
        if not name.startswith(prefix):
            continue
        error = float(name.removeprefix(prefix))
        low, high = metric["cluster_bootstrap_95_ci"]
        lines.append(
            f"| {error:.1%} | {number(metric['mean_within_layer_spearman'])} | "
            f"[{number(low)}, {number(high)}] |"
        )

    lines.extend(
        [
            "",
            summary["interpretation"],
            "",
            summary["bootstrap_note"],
            "",
            "This is a falsification test of one effective-rank TN proxy for selecting "
            "expert pairs. It is not a test of every tensorization method or of "
            "post-merge recovery training.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine fixed-pair layer replications")
    parser.add_argument(
        "--layer",
        action="append",
        required=True,
        metavar="NAME=RESULT_DIR",
        help="Named result directory containing results.json; repeat for each layer",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    layers: dict[str, list[dict[str, Any]]] = {}
    for specification in args.layer:
        if "=" not in specification:
            parser.error(f"invalid --layer {specification!r}; expected NAME=RESULT_DIR")
        name, directory = specification.split("=", 1)
        if not name or name in layers:
            parser.error(f"invalid or duplicate layer name {name!r}")
        layers[name] = _read_json(Path(directory) / "results.json")

    summary = analyze_layers(layers, args.bootstrap_samples, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_json(args.output / "summary.json", summary)
    write_report(args.output / "REPORT.md", summary)
    print(args.output / "REPORT.md")


if __name__ == "__main__":
    main()
