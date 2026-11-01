"""Tests for loop-wise evaluation and the three stopping rules."""

import torch

from looped_hessian import data
from looped_hessian.evaluate import loop_profile, summarize_length
from looped_hessian.model import LoopedGPT, ModelConfig


def test_loop_profile_matches_direct_forward():
    torch.manual_seed(0)
    model = LoopedGPT(ModelConfig(d_model=16, n_layer=1, n_head=2))
    x, y = data.make_eval_set(3, 10, seed=1)
    prof = loop_profile(model, x, y, max_loops=4, chunk_size=4, device="cpu")
    assert prof["correct"].shape == (4, 10) and prof["confidence"].shape == (4, 10)
    mask = y != data.IGNORE
    for k in range(1, 5):
        pred = model(x, torch.full((10,), k)).argmax(-1)
        expected = ((pred == y) | ~mask).all(dim=1)
        assert torch.equal(prof["correct"][k - 1], expected)


def test_stopping_rules():
    # Three loop counts, four problems: K=2 is best by accuracy, K=3 is most confident.
    correct = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
    confidence = torch.tensor([[-3.0] * 4, [-2.0] * 4, [-0.5] * 4])
    s = summarize_length(correct, confidence, n=1)
    assert s["oracle"]["k"] == 2 and s["oracle"]["acc"] == 0.75
    assert s["max_confidence"]["k"] == 3 and s["max_confidence"]["acc"] == 0.25
    assert s["fixed_n"]["k"] == 1 and s["fixed_n"]["acc"] == 0.0
    assert s["acc_per_k"] == [0.0, 0.75, 0.25]
    assert s["oracle"]["correct"] == [1, 1, 1, 0]
