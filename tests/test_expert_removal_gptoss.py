import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from expert_removal_gptoss import (  # noqa: E402
    PROJECTIONS, GptOssAdapter, align_group, compressed, load_gptoss, merged_parameters,
)


def _model():
    model, _ = load_gptoss({"seed": 17, "model": {"id": "__tiny_random_gptoss__"}, "device": "cpu"})
    return model


def _snapshot(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


@torch.inference_mode()
def test_adapter_matches_module_forward():
    model = _model()
    adapter = GptOssAdapter(model, 0)
    x = torch.randn(10, 32, generator=torch.Generator().manual_seed(0))
    for expert in range(adapter.num_experts):
        indices = torch.full((10, 2), expert)
        weights = torch.tensor([[1.0, 0.0]]).expand(10, 2)
        assert torch.allclose(adapter.experts(x, indices, weights), adapter.output(x, expert), atol=1e-6)
        gate = x @ adapter.weight(expert, "gate_proj").T + adapter.bias(expert, "gate_proj")
        up = x @ adapter.weight(expert, "up_proj").T + adapter.bias(expert, "up_proj")
        gated = (up.clamp(-7, 7) + 1) * gate.clamp(max=7) * torch.sigmoid(1.702 * gate.clamp(max=7))
        assert torch.allclose(gated, adapter.intermediate(x, expert), atol=1e-6)


def test_weight_views_write_through():
    model = _model()
    adapter = GptOssAdapter(model, 1)
    with torch.no_grad():
        adapter.weight(2, "up_proj").fill_(3.0)
        adapter.bias(2, "down_proj").fill_(-1.0)
    assert torch.all(adapter.experts.gate_up_proj[2][:, 1::2] == 3.0)
    assert not torch.any(adapter.experts.gate_up_proj[2][:, 0::2] == 3.0)
    assert torch.all(adapter.experts.down_proj_bias[2] == -1.0)


@torch.inference_mode()
def test_prune_masks_router_and_restores():
    model = _model()
    adapter = GptOssAdapter(model, 0)
    x = torch.randn(64, 32, generator=torch.Generator().manual_seed(1))
    original = adapter.router(x)
    with compressed(model, {"operator": "prune", "layers": [{"removed": []}, {"removed": []}]}):
        assert all(torch.equal(a, b) for a, b in zip(adapter.router(x), original))
    with compressed(model, {"operator": "prune", "layers": [{"removed": [0, 3]}, {"removed": [1]}]}):
        _, scores, indices = adapter.router(x)
        assert not torch.isin(indices, torch.tensor([0, 3])).any()
        assert torch.allclose(scores.sum(-1), torch.ones(64))
    assert "forward" not in vars(adapter.router)
    assert all(torch.equal(a, b) for a, b in zip(adapter.router(x), original))


def test_merge_recovers_permutation_with_biases_and_restores():
    model = _model()
    adapter = GptOssAdapter(model, 1)
    perm = torch.randperm(24, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        for projection in ("gate_proj", "up_proj"):
            adapter.weight(1, projection).copy_(adapter.weight(0, projection)[perm])
            adapter.bias(1, projection).copy_(adapter.bias(0, projection)[perm])
        adapter.weight(1, "down_proj").copy_(adapter.weight(0, "down_proj")[:, perm])
        adapter.bias(1, "down_proj").copy_(adapter.bias(0, "down_proj"))
    before = _snapshot(model)
    x = torch.randn(512, 32, generator=torch.Generator().manual_seed(3))
    members = align_group(adapter, x, [0, 1], torch.tensor([3, 3, 0, 0, 0, 0]))
    merged = merged_parameters(adapter, members)
    for projection in PROJECTIONS:
        assert torch.allclose(merged[projection][0], adapter.weight(0, projection), atol=1e-6)
        assert torch.allclose(merged[projection][1], adapter.bias(0, projection), atol=1e-6)
    single = [[{"expert": e, "perm": list(range(24)), "weight": 1.0}] for e in range(2, 6)]
    plan = {"operator": "merge", "layers": [{"groups": single}, {"groups": [members, *single]}]}
    with compressed(model, plan):
        assert torch.allclose(adapter.weight(1, "gate_proj"), adapter.weight(0, "gate_proj"), atol=1e-6)
    after = _snapshot(model)
    assert all(torch.equal(before[k], after[k]) for k in before)
