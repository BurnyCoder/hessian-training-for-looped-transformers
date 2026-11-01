"""Curvature estimators vs ground truth from finite-difference Hessians (float64, tiny model).

Global context: the manipulation check and every reported sharpness number rely on these
estimators. Ground truth is built from central differences of gradients, which does not
use the double-backward code under test.
"""

import pytest
import torch

from looped_hessian import curvature, data
from looped_hessian.model import LoopedGPT, ModelConfig
from looped_hessian.regularizers import loss_fn


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    model = LoopedGPT(ModelConfig(d_model=8, n_layer=1, n_head=2)).double()
    x, y = data.make_batch(torch.tensor([1, 2, 2, 1, 3]), torch.Generator().manual_seed(0))
    loops = torch.tensor([1, 2, 3, 2, 3])
    params = curvature.trainable_params(model)
    flat = torch.cat([p.detach().reshape(-1) for p in params])

    def grad_at(theta):
        # Gradient of the full-batch loss at flat parameter vector theta.
        curvature.set_flat(params, theta)
        g = torch.autograd.grad(loss_fn(model, x, y, loops), params)
        return torch.cat([gi.reshape(-1) for gi in g])

    h = 1e-5
    cols = []
    for i in range(flat.numel()):
        e = torch.zeros_like(flat)
        e[i] = h
        cols.append((grad_at(flat + e) - grad_at(flat - e)) / (2 * h))
    curvature.set_flat(params, flat)
    hess = torch.stack(cols, dim=1)
    hess = 0.5 * (hess + hess.T)
    return model, (x, y, loops), params, flat, hess


def test_hvp_matches_finite_difference_hessian(setup):
    model, batch, params, flat, hess = setup
    v = torch.randn(flat.numel(), dtype=torch.float64)
    vs = curvature.unflatten_like(v, params)
    hv = curvature.hvp(model, batch, vs, chunk_size=64)
    torch.testing.assert_close(torch.cat([t.reshape(-1) for t in hv]), hess @ v, rtol=1e-5, atol=1e-7)


def test_hvp_is_invariant_to_chunking(setup):
    model, batch, params, flat, hess = setup
    vs = curvature.unflatten_like(torch.randn(flat.numel(), dtype=torch.float64), params)
    a = curvature.hvp(model, batch, vs, chunk_size=2)
    b = curvature.hvp(model, batch, vs, chunk_size=64)
    for ai, bi in zip(a, b):
        torch.testing.assert_close(ai, bi)


def test_hutchinson_trace_is_unbiased_within_4_standard_errors(setup):
    model, batch, params, flat, hess = setup
    out = curvature.hutchinson_trace(model, batch, n_probes=3000, seed=0, chunk_size=64)
    exact = torch.trace(hess).item()
    assert abs(out["trace"] - exact) < 4 * out["se"]
    # Per-group diagonal-block traces add up to the total estimate.
    assert abs(sum(g["trace"] for g in out["groups"].values()) - out["trace"]) < 1e-8


def test_dominant_eigenvalue_matches_eigendecomposition(setup):
    model, batch, params, flat, hess = setup
    eig = torch.linalg.eigvalsh(hess)
    dominant = eig[eig.abs().argmax()].item()
    est = curvature.dominant_eigenvalue(model, batch, iters=1000, tol=1e-9, seed=0, chunk_size=64)
    assert abs(est - dominant) / abs(dominant) < 2e-2


def test_fisher_trace_matches_exact_ggn_trace(setup):
    model, batch, params, flat, hess = setup
    x, y, loops = batch
    # Exact GGN trace for mean cross-entropy: (1/N) sum_t sum_c p_t(c) ||grad log p_t(c)||^2.
    logits = model(x, loops)
    mask = y != data.IGNORE
    logp = torch.log_softmax(logits[mask], dim=-1)
    total = 0.0
    for t in range(logp.shape[0]):
        for c in range(logp.shape[1]):
            g = torch.autograd.grad(logp[t, c], params, retain_graph=True)
            total += logp[t, c].exp().item() * sum((gi**2).sum().item() for gi in g)
    exact = total / logp.shape[0]
    out = curvature.fisher_trace(model, batch, n_samples=3000, seed=0, chunk_size=64)
    assert abs(out["trace"] - exact) < 4 * out["se"]


def test_adaptive_sharpness_is_zero_at_zero_radius_and_restores_weights(setup):
    model, batch, params, flat, hess = setup
    out = curvature.adaptive_sharpness(model, batch, rho=0.0, n_draws=3, seed=0, chunk_size=64)
    assert abs(out["sharpness"]) < 1e-12
    curvature.adaptive_sharpness(model, batch, rho=0.05, n_draws=3, seed=0, chunk_size=64)
    # Perturbations must be undone exactly.
    torch.testing.assert_close(torch.cat([p.detach().reshape(-1) for p in params]), flat, rtol=0, atol=0)
