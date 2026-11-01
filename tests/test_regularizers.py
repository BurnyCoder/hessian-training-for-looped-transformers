"""Tests for the training-step strategies, in particular the exact Hutchinson penalty."""

import torch

from looped_hessian import curvature, data
from looped_hessian.model import LoopedGPT, ModelConfig
from looped_hessian.regularizers import BaselineStep, HutchinsonStep, hutchinson_penalty


def test_penalty_gradient_is_exact_on_a_cubic():
    # f(w) = sum a_i w_i^3 / 6  =>  H = diag(a w), and v^T H v = sum a_i w_i for Rademacher v
    # (v_i^2 = 1), so the penalty's gradient is exactly a, regardless of the probe.
    a = torch.tensor([1.0, -2.0, 3.0, 0.5], dtype=torch.float64)
    w = torch.tensor([0.3, -1.2, 2.0, 0.7], dtype=torch.float64, requires_grad=True)
    pen = hutchinson_penalty((a * w**3 / 6).sum(), [w], torch.Generator().manual_seed(0))
    torch.testing.assert_close(pen.detach(), (a * w).sum().detach())
    pen.backward()
    torch.testing.assert_close(w.grad, a)


def _tiny_model_and_batch():
    torch.manual_seed(0)
    model = LoopedGPT(ModelConfig(d_model=8, n_layer=1, n_head=2)).double()
    x, y = data.make_batch(torch.tensor([2, 3, 1, 2]), torch.Generator().manual_seed(1))
    return model, (x, y, torch.tensor([2, 3, 1, 4]))


def test_zero_lambda_reproduces_baseline_gradients():
    model, batch = _tiny_model_and_batch()
    BaselineStep()(model, *batch)
    ref = [p.grad.clone() for p in curvature.trainable_params(model)]
    model.zero_grad(set_to_none=True)
    HutchinsonStep(lam=0.0, sub_batch=2, seed=0)(model, *batch)
    for r, p in zip(ref, curvature.trainable_params(model)):
        torch.testing.assert_close(p.grad, r)


def test_third_order_gradient_through_looped_model_matches_finite_differences():
    # Directional derivative of v^T H(theta) v along u: autograd (third order through MATH
    # SDPA, LayerNorm, GELU and the loop) vs a central difference of the penalty itself.
    model, (x, y, loops) = _tiny_model_and_batch()
    params = curvature.trainable_params(model)
    from looped_hessian.regularizers import loss_fn

    def penalty():
        return hutchinson_penalty(loss_fn(model, x, y, loops), params, torch.Generator().manual_seed(3))

    pen = penalty()
    grads = torch.autograd.grad(pen, params)
    u = [torch.randn_like(p) for p in params]
    analytic = sum((g * ui).sum() for g, ui in zip(grads, u)).item()
    flat = torch.cat([p.detach().reshape(-1) for p in params])
    uf = torch.cat([ui.reshape(-1) for ui in u])
    h = 1e-5
    curvature.set_flat(params, flat + h * uf)
    plus = penalty().item()
    curvature.set_flat(params, flat - h * uf)
    minus = penalty().item()
    curvature.set_flat(params, flat)
    assert abs(analytic - (plus - minus) / (2 * h)) < 1e-5 * max(1.0, abs(analytic))


def test_hutchinson_step_reports_loss_and_penalty():
    model, batch = _tiny_model_and_batch()
    out = HutchinsonStep(lam=1e-3, sub_batch=2, seed=0)(model, *batch)
    assert set(out) >= {"loss", "vhv"}
    assert all(p.grad is not None for p in curvature.trainable_params(model))
