import json

import pytest
import torch

from moe_tn.compress import balanced_ranks, decompose, reconstruct_expert
from moe_tn.rank_run import phase_configs, choose_candidate, rank_run
from moe_tn.model import tiny_random_olmoe


def test_expert_rank_reallocates_budget_to_spatial_modes():
    shape = (64, 1024, 2048)
    budget = int(0.4 * 64 * 1024 * 2048)
    previous_area = 0
    for rank in (64, 48, 32, 24):
        re, ro, ri = balanced_ranks(shape, budget, rank)
        assert re == rank
        count = re * ro * ri + 64 * re + 1024 * ro + 2048 * ri
        assert budget * 0.98 <= count <= budget
        assert ro * ri > previous_area
        previous_area = ro * ri


def test_reduced_expert_rank_preserves_known_shared_structure():
    torch.manual_seed(23)
    weights = torch.einsum("e,o,i->eoi", torch.randn(6), torch.randn(12), torch.randn(16))
    artifact = decompose(weights, "tucker", 0.4, torch.float32,
                         rank_policy="td_balanced", expert_rank=2)
    reconstructed = torch.stack([reconstruct_expert(artifact, e, weights.device) for e in range(6)])
    assert artifact["ranks"][0] == 2
    torch.testing.assert_close(reconstructed, weights, atol=1e-5, rtol=1e-5)


def test_selection_checks_budget_and_prefers_lower_development_loss():
    rows = [{"method": "tucker_input_r64", "delta_nll": 0.03, "retained_fraction": 0.4},
            {"method": "tucker_input_r32", "delta_nll": 0.02, "retained_fraction": 0.398}]
    assert choose_candidate(rows, 0.4)["method"] == "tucker_input_r32"
    rows[1]["retained_fraction"] = 0.2
    with pytest.raises(ValueError, match="budget"):
        choose_candidate(rows, 0.4)


def test_rank_workflow_freezes_selection_before_test(tmp_path, monkeypatch):
    import moe_tn.rank_run as workflow
    import moe_tn.compression_run as runner
    monkeypatch.setattr(runner, "load_model", lambda cfg, seed: (tiny_random_olmoe(seed), None))
    config = {"seed": 17, "model": {"id": "__tiny_random_olmoe__", "layer": 1},
              "data": {"calibration_split": "train", "calibration_blocks": 2,
                       "development_split": "validation", "development_offset_blocks": 2,
                       "development_blocks": 2, "evaluation_split": "test",
                       "evaluation_blocks": 2, "block_size": 8},
              "expert_ranks": [6, 3], "full_expert_rank": 6, "fractions": [0.4],
              "minimum_budget_fill": 0.9, "tucker_rank_policy": "td_balanced", "output_dir": str(tmp_path)}
    calls = []
    def checked_run(phase):
        calls.append(phase)
        if phase["output_dir"].endswith("/test"):
            selected = json.loads((tmp_path / "selection.json").read_text())["selected_method"]
            assert set(phase["methods"]) == {"svd_input", "tucker_input_r6", selected}
        runner.run(phase)
    monkeypatch.setattr(workflow, "run", checked_run)
    rank_run(config)
    assert calls[0]["data"]["evaluation_split"] == "validation"
    assert calls[0]["data"]["evaluation_offset_blocks"] == 2
    assert calls[1]["data"]["evaluation_split"] == "test"
    assert (tmp_path / "REPORT.md").exists()
    dev = torch.load(tmp_path / "development/evaluation_blocks.pt", weights_only=True)
    test = torch.load(tmp_path / "test/evaluation_blocks.pt", weights_only=True)
    assert not torch.equal(dev, test)
    with pytest.raises(ValueError, match="different splits"):
        phase_configs({**config, "data": {**config["data"], "development_split": "test"}})
