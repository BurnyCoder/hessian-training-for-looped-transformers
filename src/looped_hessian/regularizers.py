"""Training-step strategies: plain loss (baseline) and loss + exact Hutchinson tr(H) penalty.

Local: every strategy is a callable `step(model, x, y, loops) -> metrics` that leaves the
gradient of its training objective in `p.grad`; the training loop then clips, steps the
optimizer and logs. Strategies differ only in the objective, so data, optimizer, schedule,
clipping and initialisation are shared code paths across arms.

Global: the Hutchinson arm optimizes L(θ) + λ · tr(∇²L_sub(θ)), estimated by
λ · vᵀ H v with one fresh Rademacher probe per step (Hutchinson 1990; used as a training
regularizer by SEHT, Liu, Yu & Lin 2022, arXiv:2208.05924). Hv is a double backward, and
the gradient of the penalty is a third backward pass through that graph.
"""

import torch
import torch.nn.functional as F

from looped_hessian.data import IGNORE
from looped_hessian.linalg import dot, hvp_from_grads, rademacher_like, trainable_params
from looped_hessian.model import third_order_safe


def loss_fn(model, x: torch.Tensor, y: torch.Tensor, loops: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over labelled (answer) positions; IGNORE positions are skipped."""
    device = next(model.parameters()).device
    logits = model(x.to(device), loops.to(device))
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.to(device).reshape(-1),
                           ignore_index=IGNORE)


def hutchinson_penalty(loss: torch.Tensor, params: list[torch.Tensor],
                       gen: torch.Generator) -> torch.Tensor:
    """Differentiable single-probe estimate vᵀ H v of tr(∇²loss).

    Both autograd calls keep their graphs (create_graph=True), so `penalty.backward()`
    produces ∇_θ(vᵀ H(θ) v), whose expectation over v is ∇_θ tr(H(θ)).
    """
    grads = torch.autograd.grad(loss, params, create_graph=True)
    vs = rademacher_like(params, gen)
    hv = hvp_from_grads(grads, params, vs, create_graph=True)
    return dot(hv, vs)


class BaselineStep:
    """Objective L: one forward/backward on the full batch."""

    def __call__(self, model, x, y, loops) -> dict:
        loss = loss_fn(model, x, y, loops)
        loss.backward()
        return {"loss": loss.item()}


class HutchinsonStep:
    """Objective L + λ · vᵀ H_sub v, where H_sub is the Hessian of the loss on the first
    `sub_batch` problems of the batch.

    Amortization (preregistration addendum 1): the penalty is applied on every `every`-th
    step with weight `every · λ`, so its expected contribution per step is still λ ∇tr(H)
    (the same lazy-curvature idea as Sophia's Hessian estimates every k = 10 steps,
    arXiv:2305.14342). `every = 1` is the plain per-step penalty.
    """

    def __init__(self, lam: float, sub_batch: int, seed: int, every: int = 1):
        self.lam = lam
        self.sub_batch = sub_batch
        self.seed = seed
        self.every = every
        self.calls = 0
        # Probe generator is created lazily on the parameters' device and is separate from
        # the data generator, so the baseline and this arm see identical batches.
        self.gen = None

    def __call__(self, model, x, y, loops) -> dict:
        params = trainable_params(model)
        if self.gen is None:
            self.gen = torch.Generator(device=params[0].device).manual_seed(self.seed)
        # 1) ∇L on the full batch, accumulated into .grad (frees the first graph early).
        loss = loss_fn(model, x, y, loops)
        loss.backward()
        out = {"loss": loss.item()}
        apply = self.calls % self.every == 0
        self.calls += 1
        if apply:
            # 2) every·λ ∇(vᵀ H_sub v), accumulated into the same .grad buffers. The sub-batch
            #    forward uses composite LayerNorm, the only op whose native third derivative is wrong.
            s = self.sub_batch
            with third_order_safe():
                penalty = hutchinson_penalty(loss_fn(model, x[:s], y[:s], loops[:s]), params, self.gen)
                (self.every * self.lam * penalty).backward()
            out["vhv"] = penalty.item()
        return out


def make_step(arm: str, lam: float = 0.0, sub_batch: int = 16, seed: int = 0, every: int = 1):
    # Factory used by the training loop; arm names match the preregistration.
    if arm == "baseline":
        return BaselineStep()
    if arm == "hutchinson":
        return HutchinsonStep(lam=lam, sub_batch=sub_batch, seed=seed, every=every)
    raise ValueError(f"unknown arm {arm!r}")
