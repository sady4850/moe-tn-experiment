import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from expert_removal import (  # noqa: E402
    align_group, cluster, compressed, merged_weights, pairwise_incremental_rank, random_groups,
)
from moe_tn.adapter import PROJECTIONS, OlmoeAdapter  # noqa: E402
from moe_tn.model import tiny_random_olmoe  # noqa: E402
from moe_tn.signals import _joint_matrix, effective_rank_from_singular_values  # noqa: E402


def _model():
    model = tiny_random_olmoe(17)
    model.eval()
    return model


def _weights(model):
    return {(l, e, p): OlmoeAdapter(model, l).weight(e, p).detach().clone()
            for l in range(len(model.model.layers)) for e in range(6) for p in PROJECTIONS}


@torch.inference_mode()
def test_prune_masks_router_and_restores():
    model = _model()
    adapter = OlmoeAdapter(model, 0)
    x = torch.randn(64, 32, generator=torch.Generator().manual_seed(0))
    original = adapter.block.gate(x)
    empty = {"operator": "prune", "layers": [{"removed": []} for _ in range(3)]}
    with compressed(model, empty):
        for a, b in zip(adapter.block.gate(x), original):
            assert torch.equal(a, b)
    plan = {"operator": "prune", "layers": [{"removed": [1, 4]} for _ in range(3)]}
    with compressed(model, plan):
        _, values, indices = adapter.block.gate(x)
        assert not torch.isin(indices, torch.tensor([1, 4])).any()
        assert torch.isfinite(values).all()
    assert "forward" not in vars(adapter.block.gate)
    for a, b in zip(adapter.block.gate(x), original):
        assert torch.equal(a, b)


def test_merge_alignment_recovers_permutation_and_restores():
    model = _model()
    adapter = OlmoeAdapter(model, 1)
    perm = torch.randperm(24, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        adapter.weight(1, "gate_proj").copy_(adapter.weight(0, "gate_proj")[perm])
        adapter.weight(1, "up_proj").copy_(adapter.weight(0, "up_proj")[perm])
        adapter.weight(1, "down_proj").copy_(adapter.weight(0, "down_proj")[:, perm])
    before = _weights(model)
    x = torch.randn(512, 32, generator=torch.Generator().manual_seed(4))
    members = align_group(adapter, x, [0, 1], torch.tensor([5, 5, 0, 0, 0, 0]))
    assert [m["weight"] for m in members] == [0.5, 0.5]
    merged = merged_weights(adapter, members)
    for projection in PROJECTIONS:
        assert torch.allclose(merged[projection], adapter.weight(0, projection), atol=1e-6)
    plan = {"operator": "merge",
            "layers": [{"groups": [[{"expert": e, "perm": list(range(24)), "weight": 1.0}] for e in range(6)]},
                       {"groups": [members] + [[{"expert": e, "perm": list(range(24)), "weight": 1.0}]
                                               for e in range(2, 6)]},
                       {"groups": [[{"expert": e, "perm": list(range(24)), "weight": 1.0}] for e in range(6)]}]}
    with compressed(model, plan):
        for projection in PROJECTIONS:
            assert torch.allclose(adapter.weight(1, projection), adapter.weight(0, projection), atol=1e-6)
    after = _weights(model)
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_pairwise_matches_experiment0_svd_score():
    generator = torch.Generator().manual_seed(5)
    experts = 4
    matrices = {"gate_proj": [torch.randn(24, 32, generator=generator) for _ in range(experts)],
                "up_proj": [torch.randn(24, 32, generator=generator) for _ in range(experts)],
                "down_proj": [torch.randn(32, 24, generator=generator) for _ in range(experts)]}
    matrices["gate_proj"][3] = matrices["gate_proj"][2].clone()
    matrices["up_proj"][3] = matrices["up_proj"][2].clone()
    matrices["down_proj"][3] = matrices["down_proj"][2].clone()
    score = pairwise_incremental_rank(matrices, batch=2)
    assert np.allclose(score, score.T) and score[2, 3] == 0

    def reference(i, j):
        values = []
        for p in PROJECTIONS:
            rank = [effective_rank_from_singular_values(torch.linalg.svdvals(matrices[p][k]), 0.10) for k in (i, j)]
            joint = effective_rank_from_singular_values(
                torch.linalg.svdvals(_joint_matrix(matrices[p][i], matrices[p][j], p)), 0.10)
            values.append((joint - max(rank)) / max(rank))
        return sum(values) / len(values)

    for i in range(experts):
        for j in range(i + 1, experts):
            assert abs(score[i, j] - reference(i, j)) < 1e-9


def test_grouping_helpers():
    distance = np.array([[0, 1, 5, 5], [1, 0, 5, 5], [5, 5, 0, 1], [5, 5, 1, 0]], dtype=float)
    assert cluster(distance, 2) == [[0, 1], [2, 3]]
    assert cluster(distance - 3, 2) == [[0, 1], [2, 3]]
    rng = np.random.default_rng(17)
    upper = np.triu(rng.uniform(-0.2, 0.3, (32, 32)), 1)
    shifted = upper + upper.T
    assert cluster(shifted, 20) == cluster(shifted + 1, 20)
    groups = random_groups(random.Random(17), 64, 40)
    assert len(groups) == 40 and sorted(e for g in groups for e in g) == list(range(64))
