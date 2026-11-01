"""Parameter-space linear-algebra helpers shared by the penalty and the curvature metrics.

Local: parameters are handled as lists of tensors (one per nn.Parameter), the same layout
`torch.autograd.grad` returns, so vectors such as probes v and products Hv need no
flattening. The Hessian-vector product is the "vjp over grad" (double backward) construction
from the PyTorch tutorial "Jacobians, Hessians, hvp, vhp, and more"
(https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html): Hv = d/dθ (∇L · v).

Global: `regularizers.py` uses these for the Hutchinson training penalty (with
create_graph=True so the penalty itself can be differentiated), and `curvature.py` uses them
for end-of-training measurements (no third-order graph needed).
"""

import torch


def trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    # model.parameters() yields each shared tensor once, so the tied embedding/head counts once.
    return [p for p in model.parameters() if p.requires_grad]


def dot(xs: list[torch.Tensor], ys: list[torch.Tensor]) -> torch.Tensor:
    # Euclidean inner product of two parameter-shaped tensor lists.
    return sum((x * y).sum() for x, y in zip(xs, ys))


def rademacher_like(params: list[torch.Tensor], gen: torch.Generator) -> list[torch.Tensor]:
    """i.i.d. ±1 entries; E[v v^T] = I, so E[v^T H v] = tr(H) (Hutchinson 1990)."""
    return [torch.randint(0, 2, p.shape, generator=gen, device=p.device).to(p.dtype) * 2 - 1
            for p in params]


def hvp_from_grads(grads, params, vs, create_graph: bool = False, retain_graph: bool = True):
    """Hessian-vector product from first-order gradients that were built with create_graph=True."""
    # Differentiating the scalar ∇L · v w.r.t. θ gives H v (H symmetric).
    return torch.autograd.grad(dot(grads, vs), params, create_graph=create_graph,
                               retain_graph=retain_graph)


def unflatten_like(flat: torch.Tensor, params: list[torch.Tensor]) -> list[torch.Tensor]:
    # Split a flat vector into tensors shaped like `params` (inverse of torch.cat of reshapes).
    out, i = [], 0
    for p in params:
        out.append(flat[i : i + p.numel()].view_as(p))
        i += p.numel()
    return out


@torch.no_grad()
def set_flat(params: list[torch.Tensor], flat: torch.Tensor) -> None:
    # Overwrite parameters in place from a flat vector (used by finite-difference tests).
    for p, v in zip(params, unflatten_like(flat, params)):
        p.copy_(v)
