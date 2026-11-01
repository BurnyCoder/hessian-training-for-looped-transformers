"""Curvature measurement phase: every preregistered sharpness metric for one checkpoint.

Local: all metrics use the same fixed 512-problem batch from the training distribution,
read out at K = n, with fixed probe seeds, so different models are measured identically
(preregistration Sec. 3). The loop-count study re-measures tr(H) and the dominant
eigenvalue with every problem read out at a fixed K, to see how curvature scales with
looping (Gong et al. 2025, arXiv:2510.10089, argue looping biases training toward steep
valleys).

Global: writes `curvature.json` into the run folder; `report.py` reads it for the
manipulation check, tables and figures.
"""

import json
import os
import time

import torch

from looped_hessian import curvature
from looped_hessian.config import RunConfig
from looped_hessian.train import fixed_batch


def measure(model, cfg: RunConfig, logger, out_name: str = "curvature.json") -> dict:
    mc = cfg.measure
    t0 = time.time()
    batch = fixed_batch(mc.batch, cfg.train.max_train_len, seed=mc.seed)
    cs = mc.chunk_size
    out = {"loss": curvature.full_loss(model, batch, cs)}
    out["trace"] = curvature.hutchinson_trace(model, batch, mc.probes, seed=mc.seed + 1, chunk_size=cs)
    logger.info(f"tr(H) = {out['trace']['trace']:.4g} +- {out['trace']['se']:.2g} (SE, {mc.probes} probes)")
    out["dominant_eigenvalue"] = curvature.dominant_eigenvalue(model, batch, mc.power_iters, mc.power_tol,
                                                               seed=mc.seed + 2, chunk_size=cs)
    logger.info(f"dominant eigenvalue = {out['dominant_eigenvalue']:.4g}")
    out["fisher_trace"] = curvature.fisher_trace(model, batch, mc.fisher_samples, seed=mc.seed + 3, chunk_size=cs)
    logger.info(f"Fisher trace = {out['fisher_trace']['trace']:.4g} +- {out['fisher_trace']['se']:.2g}")
    out["adaptive_sharpness"] = [curvature.adaptive_sharpness(model, batch, rho, mc.sharpness_draws,
                                                              seed=mc.seed + 4, chunk_size=cs)
                                 for rho in mc.sharpness_rhos]
    out["weight_norms"] = curvature.weight_norms(model)
    out["loop_study"] = []
    for k in mc.loop_ks:
        # Same problems, every one read out after exactly k iterations.
        kb = (batch[0], batch[1], torch.full_like(batch[2], k))
        row = {"k": k, "loss": curvature.full_loss(model, kb, cs),
               "trace": curvature.hutchinson_trace(model, kb, mc.loop_probes, seed=mc.seed + 5, chunk_size=cs),
               "dominant_eigenvalue": curvature.dominant_eigenvalue(model, kb, mc.loop_power_iters, mc.power_tol,
                                                                    seed=mc.seed + 6, chunk_size=cs)}
        out["loop_study"].append(row)
        logger.info(f"loop study K={k}: loss={row['loss']:.4g} tr(H)={row['trace']['trace']:.4g} "
                    f"lambda_dom={row['dominant_eigenvalue']:.4g}")
    out["seconds"] = time.time() - t0
    with open(os.path.join(cfg.run_dir, out_name), "w") as f:
        json.dump(out, f, indent=2)
    return out
