import json

import pytest
import torch

from moe_tn.adapter import OlmoeAdapter, PROJECTIONS
from moe_tn.compress import decompose, reconstruct_expert
from moe_tn.compression_run import original_layer, run
from moe_tn.model import tiny_random_olmoe


@pytest.mark.parametrize("method", ["svd", "shared", "tucker"])
def test_compression_recovers_known_separable_tensor_within_budget(method):
    torch.manual_seed(3)
    # Exact rank-one structure in every mode must survive every representation.
    w = torch.einsum("e,o,i->eoi", torch.randn(4), torch.randn(8), torch.randn(6))
    artifact = decompose(w, method, 0.6, torch.float32)
    reconstructed = torch.stack([reconstruct_expert(artifact, e, w.device) for e in range(4)])
    assert artifact["stored_parameters"] <= int(w.numel() * 0.6)
    assert artifact["stored_bytes"] == sum(v.numel() * v.element_size()
                                           for v in artifact["factors"].values())
    torch.testing.assert_close(reconstructed, w, atol=1e-5, rtol=1e-5)


def test_layer_is_restored_after_failed_trial():
    adapter = OlmoeAdapter(tiny_random_olmoe(1), 1)
    before = {p: torch.stack([adapter.weight(e, p).detach().clone()
                             for e in range(adapter.num_experts)]) for p in PROJECTIONS}
    with pytest.raises(RuntimeError), original_layer(adapter), torch.no_grad():
        for p in PROJECTIONS:
            adapter.weight(0, p).zero_()
        raise RuntimeError("injected failure")
    for p in PROJECTIONS:
        for e in range(adapter.num_experts):
            assert torch.equal(adapter.weight(e, p), before[p][e])


def test_runner_resumes_and_rejects_changed_config(tmp_path, monkeypatch):
    import moe_tn.compression_run as runner
    monkeypatch.setattr(runner, "load_model", lambda cfg, seed: (tiny_random_olmoe(seed), None))
    config = {"seed": 17, "model": {"id": "__tiny_random_olmoe__", "layer": 1},
              "data": {"evaluation_blocks": 2, "calibration_blocks": 2, "block_size": 8},
              "methods": ["svd_input", "shared_input", "tucker_input", "tucker_none"], "fractions": [0.8],
              "tucker_rank_policy": "td_balanced",
              "output_dir": str(tmp_path)}
    run(config, stop_after=1)
    first = tmp_path / "svd_input_0.8" / "result.json"
    timestamp = first.stat().st_mtime_ns
    run(config)
    assert first.stat().st_mtime_ns == timestamp
    for method in config["methods"]:
        result = json.loads((tmp_path / f"{method}_0.8" / "result.json").read_text())
        assert result["retained_fraction"] <= 0.8
        assert len(result["block_nll"]) == 2
    assert "Tucker minus shared" in (tmp_path / "REPORT.md").read_text()
    input_result = json.loads((tmp_path / "tucker_input_0.8/result.json").read_text())
    raw_result = json.loads((tmp_path / "tucker_none_0.8/result.json").read_text())
    for p in PROJECTIONS:
        assert input_result["projections"][p]["ranks"] == raw_result["projections"][p]["ranks"]
    stats = json.loads((tmp_path / "whitening_stats.json").read_text())
    assert sum(stats["expert_counts"]) == stats["down_rows"] == stats["hidden_rows"] * 2
    with pytest.raises(ValueError, match="changed"):
        run({**config, "fractions": [0.5]})


@pytest.mark.parametrize("method", ["svd", "shared", "tucker"])
def test_input_whitening_inverse_recovers_separable_weights(method):
    from moe_tn.whitening import regularized_cholesky
    torch.manual_seed(19)
    w = torch.einsum("e,o,i->eoi", torch.randn(4), torch.randn(8), torch.randn(6))
    x = torch.randn(20, 6) * torch.arange(1, 7)
    l, _ = regularized_cholesky(x.T @ x / len(x))
    artifact = decompose(w, method, 0.8, torch.float32, l, "td_balanced")
    recovered = torch.stack([reconstruct_expert(artifact, e, w.device) for e in range(4)])
    torch.testing.assert_close(recovered, w, rtol=1e-4, atol=1e-4)
    assert artifact["stored_parameters"] <= int(w.numel() * 0.8)
    assert all(v.dtype == torch.float32 for v in artifact["factors"].values())


def test_whitened_svd_preserves_high_variance_direction():
    # Rank-one raw SVD preserves the larger weight; input-aware SVD should instead
    # preserve the direction dominating output error on anisotropic inputs.
    w = torch.diag(torch.tensor([3., 1.])).unsqueeze(0)
    l = torch.diag(torch.tensor([1., 10.]))
    raw = decompose(w, "svd", 1.0, torch.float32)
    aware = decompose(w, "svd", 1.0, torch.float32, l)
    raw_w = reconstruct_expert(raw, 0, w.device)
    aware_w = reconstruct_expert(aware, 0, w.device)
    assert ((aware_w - w[0]) @ l).square().sum() < ((raw_w - w[0]) @ l).square().sum()
    torch.testing.assert_close(aware_w, torch.diag(torch.tensor([0., 1.])))


def test_calibration_resume_preserves_moments(tmp_path):
    from moe_tn.whitening import collect_moments, regularized_cholesky
    model = tiny_random_olmoe(7).eval()
    adapter = OlmoeAdapter(model, 1)
    blocks = torch.randint(3, 128, (2, 8))
    first = collect_moments(model, adapter, blocks[:1], torch.device("cpu"))
    resumed = collect_moments(model, adapter, blocks, torch.device("cpu"), first)
    full = collect_moments(model, adapter, blocks, torch.device("cpu"))
    for key in ("hidden_sum", "down_sum", "expert_counts"):
        torch.testing.assert_close(resumed[key], full[key])
    # Singular moments must be handled explicitly through regularization.
    l, diagnostics = regularized_cholesky(torch.ones(4, 4))
    assert torch.isfinite(l).all() and diagnostics["clamped_eigenvalues"] > 0
