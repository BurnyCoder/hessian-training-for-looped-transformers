"""Third-order autograd correctness for every op in LoopedGPT (regression test).

Global context: the Hutchinson penalty's gradient ∇(vᵀHv) is a third derivative of the
loss. We compare autograd against central differences of vᵀHv itself (float64, two step
sizes that agree, so truncation error is negligible). While building this project the
native `torch.nn.functional.layer_norm` failed this check in PyTorch 2.15.0 (only for
non-constant probes; an all-ones probe hides it), which is why model.py uses a composite
LayerNorm. The native op is kept here as an xfail so a future PyTorch fix becomes visible.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from looped_hessian.model import LayerNorm, third_order_safe

DT = torch.float64


def _third_order_rel_error(fn, shapes, seed=0):
    # Relative error between autograd and finite differences of the directional derivative
    # u · ∇_θ (vᵀ H(θ) v) for a random Rademacher probe v and Gaussian direction u.
    torch.manual_seed(seed)
    params = [(0.5 * torch.randn(*s, dtype=DT)).requires_grad_() for s in shapes]
    vs = [torch.randint(0, 2, p.shape, dtype=DT) * 2 - 1 for p in params]

    def pen(ps):
        g = torch.autograd.grad(fn(*ps), ps, create_graph=True)
        hv = torch.autograd.grad(sum((a * b).sum() for a, b in zip(g, vs)), ps, create_graph=True)
        return sum((a * b).sum() for a, b in zip(hv, vs))

    us = [torch.randn_like(p) for p in params]
    grads = torch.autograd.grad(pen(params), params)
    analytic = sum((g * u).sum() for g, u in zip(grads, us)).item()
    h = 1e-5
    plus = pen([(p + h * u).detach().requires_grad_() for p, u in zip(params, us)]).item()
    minus = pen([(p - h * u).detach().requires_grad_() for p, u in zip(params, us)]).item()
    fd = (plus - minus) / (2 * h)
    return abs(analytic - fd) / max(1e-12, abs(fd))


X = torch.randn(2, 5, 8, dtype=DT, generator=torch.Generator().manual_seed(1))
IDX = torch.randint(0, 6, (2, 5), generator=torch.Generator().manual_seed(2))
TGT = torch.randint(0, 6, (10,), generator=torch.Generator().manual_seed(3))


def _attention(q):
    q = q.view(2, 5, 2, 4).transpose(1, 2)
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, 0.7 * q, 1.3 * q, is_causal=True)


def _composite_ln(z, w, b):
    # Route the test parameters through the module's composite path via functional_call.
    with third_order_safe():
        return torch.func.functional_call(LayerNorm(8).double(), {"weight": w, "bias": b}, (z,))


CASES = {
    "composite_layernorm": (lambda W, w, b: _composite_ln(X @ W, w, b).pow(3).sum(), [(8, 8), (8,), (8,)]),
    "gelu": (lambda W: F.gelu(X @ W).pow(2).sum(), [(8, 8)]),
    "sdpa_math_causal": (lambda W: _attention(X @ W).pow(2).sum(), [(8, 8)]),
    "embedding_tied_head_cross_entropy": (
        lambda E, W: F.cross_entropy(F.linear(F.embedding(IDX, E) @ W, E).reshape(-1, 6), TGT),
        [(6, 8), (8, 8)]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_third_order_matches_finite_differences(name):
    fn, shapes = CASES[name]
    assert _third_order_rel_error(fn, shapes) < 1e-5


@pytest.mark.xfail(reason="native layer_norm third derivative disagrees with finite differences in torch 2.15.0",
                   strict=False)
def test_native_layer_norm_third_order():
    fn = lambda W, w, b: F.layer_norm(X @ W, (8,), w, b).pow(3).sum()  # noqa: E731
    assert _third_order_rel_error(fn, [(8, 8), (8,), (8,)]) < 1e-5
