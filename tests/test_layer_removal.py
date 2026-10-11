import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from layer_removal import block_influence, better, dev_sets, lowest_first, removed, verdict  # noqa: E402
from moe_tn.model import tiny_random_olmoe  # noqa: E402


def _model():
    model = tiny_random_olmoe(17)
    model.eval()
    return model


@torch.inference_mode()
def test_removed_skips_block_and_restores():
    model = _model()
    ids = torch.randint(3, 128, (1, 16), generator=torch.Generator().manual_seed(0))
    original = model(input_ids=ids).logits
    layers = list(model.model.layers)
    with removed(model, [0, 1, 2]):
        skipped = model(input_ids=ids).logits
        expected = model.lm_head(model.model.norm(model.model.embed_tokens(ids)))
        assert torch.allclose(skipped, expected, atol=1e-6)
    assert all(a is b for a, b in zip(model.model.layers, layers))
    assert torch.allclose(model(input_ids=ids).logits, original)


def test_block_influence_is_zero_for_identity_layer():
    model = _model()
    blocks = torch.randint(3, 128, (2, 16), generator=torch.Generator().manual_seed(1))
    with removed(model, [1]):
        scores = block_influence(model, blocks)
    assert abs(scores[1]) < 1e-6
    assert scores[0] > 1e-4 and scores[2] > 1e-4


def test_selection_and_candidate_sets():
    assert lowest_first([0.3, 0.1, 0.1, 0.5]) == [1, 2, 0, 3]
    sets = dev_sets(16, [1, 2, 4], exhaustive_limit=200, random_sets=100, seed=17)
    assert len(sets[1]) == 16 and len(sets[2]) == 120
    assert len(sets[4]) == 100 and len({tuple(s) for s in sets[4]}) == 100
    assert sets[4] == dev_sets(16, [4], 200, 100, 17)[4]


def test_verdict_rules():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, 64)
    test = {"wiki": {"a": 2.0 + noise, "b": 2.1 + noise}, "c4": {"a": 3.0 + noise, "b": 3.1 + noise}}
    down = {"a": {"mean": 50.0}, "b": {"mean": 50.2}}
    assert better("a", "b", test, down)
    assert not better("b", "a", test, down)
    assert verdict("a", "b", test, down, dev_nll=1.0, dev_median=2.0) == "win"
    assert verdict("b", "a", test, down, dev_nll=1.0, dev_median=2.0) == "lose"
    # Better NLL but a downstream drop beyond the margin is not a win.
    assert verdict("a", "b", test, {"a": {"mean": 49.0}, "b": {"mean": 50.0}}, 1.0, 2.0) == "tie"
    # Winning on one corpus only is a tie; worse than the dev median is a loss.
    mixed = {"wiki": test["wiki"], "c4": {"a": 3.1 + noise, "b": 3.0 + noise}}
    assert verdict("a", "b", mixed, down, 1.0, 2.0) == "tie"
    assert verdict("a", "b", mixed, down, 3.0, 2.0) == "lose"
    assert verdict("a", "a", test, down, 1.0, 2.0) == "tie (same set)"
