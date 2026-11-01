"""Invariants of LoopedGPT that the training and evaluation code rely on."""

import torch

from looped_hessian import data
from looped_hessian.model import LoopedGPT, ModelConfig


def _tiny(seed=0):
    torch.manual_seed(seed)
    return LoopedGPT(ModelConfig(d_model=16, n_layer=2, n_head=2)).double()


def test_per_sample_loop_gather_matches_separate_runs():
    model = _tiny()
    x, _ = data.make_batch(torch.tensor([3, 3, 3]), torch.Generator().manual_seed(0))
    loops = torch.tensor([1, 4, 2])
    batched = model(x, loops)
    # Each sample's logits must equal running that sample alone with its own K.
    for i in range(3):
        alone = model(x[i : i + 1], torch.tensor([int(loops[i])]))
        torch.testing.assert_close(batched[i], alone[0])


def test_right_padding_does_not_change_real_positions():
    model = _tiny()
    x, _ = data.make_batch(torch.tensor([2]), torch.Generator().manual_seed(3))
    padded = torch.cat([x, torch.full((1, 5), data.PAD)], dim=1)
    k = torch.tensor([3])
    # Causal attention: appended PAD tokens cannot affect earlier logits.
    torch.testing.assert_close(model(padded, k)[:, : x.shape[1]], model(x, k))


def test_iterate_loops_matches_forward_at_each_k():
    model = _tiny()
    x, _ = data.make_batch(torch.full((2,), 4), torch.Generator().manual_seed(5))
    for k, logits in model.iterate_loops(x, 5):
        torch.testing.assert_close(logits, model(x, torch.full((2,), k)))


def test_weights_are_tied_and_shared_across_loops():
    model = _tiny()
    assert model.lm_head.weight is model.wte.weight
    # Parameter count must not depend on the number of loops (weight tying across depth).
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == sum(p.numel() for p in _tiny(1).parameters())
