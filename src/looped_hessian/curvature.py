"""End-of-training curvature measurements (manipulation check and secondary metrics).

Local: every measurement is of the full-batch mean loss L on a fixed measurement batch,
evaluated in chunks to bound memory. Because L = Σ_c w_c L_c with w_c = (labelled tokens in
chunk c) / (all labelled tokens), Hessians, gradients and losses combine exactly with the
same weights, so results do not depend on the chunk size.

Metrics (preregistration Sec. 3):
- `hutchinson_trace`: tr(H) = E[vᵀHv], Rademacher v (Hutchinson 1990), with standard error,
  and per-parameter-group diagonal-block traces (Σ_g v_gᵀ(Hv)_g, Zhang et al. 2024,
  arXiv:2402.16788, report strongly block-heterogeneous transformer Hessians).
- `dominant_eigenvalue`: power iteration on H as in PyHessian (Yao et al. 2020,
  arXiv:1912.07145, github.com/amirgholami/PyHessian, MIT); returns the eigenvalue of
  largest magnitude.
- `fisher_trace`: trace of the Fisher = Gauss-Newton matrix for softmax cross-entropy,
  tr(G) = N · E||∇L̂||² where L̂ uses labels sampled from the model and N counts the sampled
  labels (answer tokens): the batched Gauss-Newton-Bartlett estimator of Sophia
  (arXiv:2305.14342, Alg. 2); Liu et al. 2023 (arXiv:2210.14199, App. B.1) use the
  per-example Bartlett form. Cross terms vanish because each sampled-label score has zero
  mean under the model.
- `adaptive_sharpness`: average-case adaptive sharpness E_ε[L(θ + ε⊙|θ|)] − L(θ),
  ε ~ N(0, ρ²I) (Andriushchenko et al. 2023, arXiv:2302.07011, Eq. 1 with c = |θ|), which is
  invariant to function-preserving multiplicative reparametrizations f(θ⊙c) = f(θ), unlike
  tr(H) (Dinh et al. 2017, arXiv:1703.04933).
- `weight_norms`: per-group L2 norms, to detect tr(H) reductions bought by norm growth.
"""

import math

import torch

from looped_hessian.data import IGNORE
from looped_hessian.linalg import (dot, hvp_from_grads, rademacher_like, set_flat,  # noqa: F401
                                   trainable_params, unflatten_like)
from looped_hessian.regularizers import loss_fn


def param_groups(model) -> dict[str, list[int]]:
    """Indices (into `trainable_params(model)`) of embedding / attention / mlp / layernorm."""
    groups = {"embedding": [], "attention": [], "mlp": [], "layernorm": []}
    # named_parameters() has the same order and tied-weight de-duplication as parameters().
    for i, (name, _) in enumerate(model.named_parameters()):
        key = ("embedding" if name.startswith("wte") else "attention" if ".attn." in name
               else "mlp" if ".mlp." in name else "layernorm")
        groups[key].append(i)
    return groups


def _chunks(batch, chunk_size: int):
    # Yield (chunk, weight) with weights proportional to labelled-token counts (sum to 1).
    x, y, loops = batch
    total = (y != IGNORE).sum().item()
    for i in range(0, x.shape[0], chunk_size):
        sl = slice(i, i + chunk_size)
        yield (x[sl], y[sl], loops[sl]), (y[sl] != IGNORE).sum().item() / total


def _mean_se(values: torch.Tensor) -> dict:
    # Monte Carlo mean and standard error of the mean.
    return {"trace": values.mean().item(), "se": values.std(unbiased=True).item() / math.sqrt(len(values))}


def hvp(model, batch, vs, chunk_size: int = 64) -> list[torch.Tensor]:
    """Exact H v for the full-batch mean loss, accumulated over chunks."""
    params = trainable_params(model)
    out = [torch.zeros_like(p) for p in params]
    for chunk, w in _chunks(batch, chunk_size):
        grads = torch.autograd.grad(loss_fn(model, *chunk), params, create_graph=True)
        for o, h in zip(out, hvp_from_grads(grads, params, vs, retain_graph=False)):
            o.add_(h.detach(), alpha=w)
    return out


def hutchinson_trace(model, batch, n_probes: int, seed: int, chunk_size: int = 64) -> dict:
    """Hutchinson estimate of tr(H) with per-probe standard error and per-group block traces."""
    params = trainable_params(model)
    groups = param_groups(model)
    per_group = {g: torch.zeros(n_probes, dtype=torch.float64) for g in groups}
    for chunk, w in _chunks(batch, chunk_size):
        grads = torch.autograd.grad(loss_fn(model, *chunk), params, create_graph=True)
        # Re-seed per chunk so every chunk sees the same probe sequence v_1..v_n; the
        # weighted sum over chunks is then vᵀHv for the full-batch Hessian.
        gen = torch.Generator(device=params[0].device).manual_seed(seed)
        for k in range(n_probes):
            vs = rademacher_like(params, gen)
            hv = hvp_from_grads(grads, params, vs)
            for g, idx in groups.items():
                per_group[g][k] += w * dot([hv[i] for i in idx], [vs[i] for i in idx]).item()
    total = sum(per_group.values())
    return {**_mean_se(total), "n_probes": n_probes,
            "groups": {g: _mean_se(v) for g, v in per_group.items()}}


def dominant_eigenvalue(model, batch, iters: int, tol: float, seed: int, chunk_size: int = 64) -> float:
    """Power iteration: v <- Hv / ||Hv||, eigenvalue = vᵀHv; stops on relative change < tol."""
    params = trainable_params(model)
    gen = torch.Generator(device=params[0].device).manual_seed(seed)
    vs = [torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype) for p in params]
    norm = math.sqrt(dot(vs, vs).item())
    vs = [v / norm for v in vs]
    eig = None
    for _ in range(iters):
        hv = hvp(model, batch, vs, chunk_size)
        new = dot(hv, vs).item()
        norm = math.sqrt(dot(hv, hv).item())
        vs = [h / norm for h in hv]
        if eig is not None and abs(new - eig) <= tol * (abs(eig) + 1e-12):
            return new
        eig = new
    return eig


def fisher_trace(model, batch, n_samples: int, seed: int, chunk_size: int = 64) -> dict:
    """tr(Fisher) = N · E||∇L̂||², L̂ = mean CE against labels sampled from the model."""
    params = trainable_params(model)
    device = params[0].device
    gen = torch.Generator(device=device).manual_seed(seed)
    n_tokens = (batch[1] != IGNORE).sum().item()
    estimates = torch.zeros(n_samples, dtype=torch.float64)
    for k in range(n_samples):
        total_grad = [torch.zeros_like(p) for p in params]
        for (x, y, loops), w in _chunks(batch, chunk_size):
            logits = model(x.to(device), loops.to(device))[(y != IGNORE).to(device)]
            # Sample one label per answer position from the model's own predictive distribution.
            sampled = torch.multinomial(torch.softmax(logits.detach(), -1), 1, generator=gen).squeeze(1)
            loss = torch.nn.functional.cross_entropy(logits, sampled)
            for t, g in zip(total_grad, torch.autograd.grad(loss, params)):
                t.add_(g, alpha=w)
        estimates[k] = n_tokens * dot(total_grad, total_grad).item()
    return _mean_se(estimates)


@torch.no_grad()
def full_loss(model, batch, chunk_size: int = 64) -> float:
    # Full-batch mean loss assembled from chunk losses with token-count weights.
    return sum(w * loss_fn(model, *chunk).item() for chunk, w in _chunks(batch, chunk_size))


@torch.no_grad()
def adaptive_sharpness(model, batch, rho: float, n_draws: int, seed: int, chunk_size: int = 64) -> dict:
    """Average-case adaptive sharpness; parameters are restored exactly afterwards."""
    params = trainable_params(model)
    backup = [p.detach().clone() for p in params]
    base = full_loss(model, batch, chunk_size)
    gen = torch.Generator(device=params[0].device).manual_seed(seed)
    diffs = torch.zeros(n_draws, dtype=torch.float64)
    for k in range(n_draws):
        for p, b in zip(params, backup):
            # θ + ε ⊙ |θ| with ε ~ N(0, ρ²): noise scaled elementwise by weight magnitude.
            p.copy_(b + rho * torch.randn(b.shape, generator=gen, device=b.device, dtype=b.dtype) * b.abs())
        diffs[k] = full_loss(model, batch, chunk_size) - base
    for p, b in zip(params, backup):
        p.copy_(b)
    out = _mean_se(diffs)
    return {"sharpness": out["trace"], "se": out["se"], "rho": rho}


def weight_norms(model) -> dict[str, float]:
    # L2 norm of each parameter group (concatenated) and of all parameters.
    params = trainable_params(model)
    norms = {g: math.sqrt(sum(params[i].detach().pow(2).sum().item() for i in idx))
             for g, idx in param_groups(model).items()}
    norms["total"] = math.sqrt(sum(p.detach().pow(2).sum().item() for p in params))
    return norms
