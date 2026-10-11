from __future__ import annotations

import math
from typing import Any

import numpy as np


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return math.nan
    ranked_left = _rankdata(left)
    ranked_right = _rankdata(right)
    if ranked_left.std() == 0 or ranked_right.std() == 0:
        return math.nan
    return float(np.corrcoef(ranked_left, ranked_right)[0, 1])


def bootstrap_spearman(
    predictor: np.ndarray,
    damage: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(samples):
        indices = generator.integers(0, len(damage), len(damage))
        estimate = spearman(predictor[indices], damage[indices])
        if not math.isnan(estimate):
            estimates.append(estimate)
    if not estimates:
        return math.nan, math.nan
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def analyze(
    rows: list[dict[str, Any]],
    primary_error: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    error_label = f"{primary_error:.4f}"
    damage = np.asarray([row["delta_nll"] for row in rows], dtype=np.float64)
    predictors = {
        "weight_cosine": -np.asarray([row["weight_cosine"] for row in rows]),
        "activation_cosine": -np.asarray([row["activation_cosine"] for row in rows]),
        "router_usage_sum": np.asarray(
            [row["router_usage_first"] + row["router_usage_second"] for row in rows]
        ),
        "router_probability_sum": np.asarray(
            [
                row["router_mean_probability_first"]
                + row["router_mean_probability_second"]
                for row in rows
            ]
        ),
        "router_co_routing": -np.asarray([row["router_co_routing"] for row in rows]),
    }
    rank_fields = sorted(
        key
        for key in rows[0]
        if key.startswith("tn_delta_rank_normalized_eps_")
    )
    for field in rank_fields:
        error = field.removeprefix("tn_delta_rank_normalized_eps_")
        predictors[f"tn_incremental_rank_eps_{error}"] = np.asarray(
            [row[field] for row in rows]
        )
    top_k = max(1, min(5, len(rows) // 4))
    metrics: dict[str, Any] = {}
    for offset, (name, predictor) in enumerate(predictors.items()):
        correlation = spearman(predictor, damage)
        low, high = bootstrap_spearman(
            predictor, damage, bootstrap_samples, seed + offset
        )
        selected = np.argsort(predictor)[:top_k]
        metrics[name] = {
            "spearman_with_damage": _finite_or_none(correlation),
            "spearman_bootstrap_95_ci": [_finite_or_none(low), _finite_or_none(high)],
            "top_k": top_k,
            "top_k_mean_delta_nll": float(damage[selected].mean()),
            "top_k_worst_delta_nll": float(damage[selected].max()),
            "selected_pairs": [
                [int(rows[index]["expert_first"]), int(rows[index]["expert_second"])]
                for index in selected
            ],
        }
    primary_name = f"tn_incremental_rank_eps_{error_label}"
    metrics["tn_incremental_rank"] = metrics[primary_name]
    metrics["random_pair_reference"] = {
        "mean_delta_nll": float(damage.mean()),
        "median_delta_nll": float(np.median(damage)),
        "best_observed_delta_nll": float(damage.min()),
        "worst_observed_delta_nll": float(damage.max()),
    }
    return {
        "interpretation": (
            "Predictors are oriented so lower means 'safer to merge'; positive Spearman "
            "therefore means the signal correctly tracks greater merge damage."
        ),
        "primary_reconstruction_error": primary_error,
        "num_pairs": len(rows),
        "metrics": metrics,
    }
