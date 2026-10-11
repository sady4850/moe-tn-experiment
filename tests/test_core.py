from __future__ import annotations

import torch

from moe_tn.adapter import OlmoeAdapter, PROJECTIONS
from moe_tn.analyze import analyze, spearman
from moe_tn.combined import analyze_layers
from moe_tn.model import tiny_random_olmoe
from moe_tn.signals import effective_rank_from_singular_values, sample_pairs


def test_effective_rank_obeys_relative_error_budget() -> None:
    singular_values = torch.tensor([4.0, 3.0])
    assert effective_rank_from_singular_values(singular_values, 0.60) == 1
    assert effective_rank_from_singular_values(singular_values, 0.59) == 2


def test_pair_sampling_is_deterministic_and_unique() -> None:
    first = sample_pairs(8, 12, seed=7)
    second = sample_pairs(8, 12, seed=7)
    assert first == second
    assert len(set(first)) == 12
    assert all(left < right for left, right in first)


def test_merge_context_sets_both_slots_and_restores() -> None:
    model = tiny_random_olmoe(seed=3)
    adapter = OlmoeAdapter(model, layer_index=1)
    originals = {
        (expert, projection): adapter.weight(expert, projection).detach().clone()
        for expert in (0, 1)
        for projection in PROJECTIONS
    }
    with adapter.merged_pair(0, 1, alpha=0.5):
        for projection in PROJECTIONS:
            assert torch.equal(adapter.weight(0, projection), adapter.weight(1, projection))
    for key, original in originals.items():
        assert torch.equal(adapter.weight(*key), original)


def test_spearman_orientation() -> None:
    import numpy as np

    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])) == 1.0
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([30.0, 20.0, 10.0])) == -1.0


def test_constant_predictor_is_serializable_as_null() -> None:
    rows = [
        {
            "expert_first": 0,
            "expert_second": index + 1,
            "delta_nll": float(index),
            "weight_cosine": 0.0,
            "activation_cosine": 0.0,
            "router_co_routing": 0.0,
            "router_usage_first": 0.0,
            "router_usage_second": 0.0,
            "router_mean_probability_first": 0.0,
            "router_mean_probability_second": 0.0,
            "tn_delta_rank_normalized_eps_0.1000": 0.0,
        }
        for index in range(3)
    ]
    summary = analyze(rows, primary_error=0.1, bootstrap_samples=5, seed=1)
    assert summary["metrics"]["tn_incremental_rank"]["spearman_with_damage"] is None


def test_combined_analysis_averages_within_layer_correlations() -> None:
    def rows(damages: list[float]) -> list[dict[str, float | int]]:
        return [
            {
                "expert_first": index,
                "expert_second": index + 10,
                "delta_nll": damage,
                "weight_cosine": -float(index),
                "activation_cosine": -float(index),
                "router_co_routing": -float(index),
                "router_usage_first": float(index),
                "router_usage_second": 0.0,
                "router_mean_probability_first": float(index),
                "router_mean_probability_second": 0.0,
                "tn_delta_rank_normalized_eps_0.1000": float(index),
            }
            for index, damage in enumerate(damages)
        ]

    summary = analyze_layers(
        {"ascending": rows([0.0, 1.0, 2.0]), "descending": rows([2.0, 1.0, 0.0])},
        bootstrap_samples=10,
        seed=4,
    )
    assert summary["metrics"]["tn_incremental_rank"]["mean_within_layer_spearman"] == 0.0
    assert (
        summary["comparisons_vs_primary_tn"]["router_usage_sum"]
        ["mean_spearman_difference_baseline_minus_tn"]
        == 0.0
    )
