"""Length-generalization evaluation with per-loop predictions and three stopping rules.

Local: for each test length, the model is run once for K = 1 .. n + extra_loops iterations
and the prediction after every iteration is scored (exact match over all n+2 answer
positions) together with a label-free confidence (mean log max-probability over the answer
positions). Three ways to pick the loop count per length (preregistration Sec. 3):
- oracle: the K with the highest accuracy for the length (uses labels). This is a per-length
  variant of the oracle of Kuo et al. 2026 (arXiv:2606.29983, Sec. 3.3), which picks the
  best K per example from {1..n+2}, so the two are not directly comparable;
- max_confidence: the K with the highest mean confidence over the length's test problems,
  a label-free batch variant of Fan et al. 2024's maximum-confidence rule (arXiv:2409.15647,
  Eq. 2, Appendix B);
- fixed_n: K = n (Fan et al.'s step rule T(n) = n).

Global: `evaluate_model` produces the per-length JSON that `report.py` turns into the
primary metric (mean oracle accuracy over OOD lengths), figures and tables.
"""

import torch

from looped_hessian import data


@torch.no_grad()
def loop_profile(model, x: torch.Tensor, y: torch.Tensor, max_loops: int, chunk_size: int,
                 device) -> dict:
    """Per-iteration exact-match (bool [K, N]) and confidence (float [K, N]) for one test set."""
    correct, confidence = [], []
    for i in range(0, x.shape[0], chunk_size):
        xc, yc = x[i : i + chunk_size].to(device), y[i : i + chunk_size].to(device)
        mask = yc != data.IGNORE
        c_rows, f_rows = [], []
        for _, logits in model.iterate_loops(xc, max_loops):
            logp = torch.log_softmax(logits.float(), dim=-1)
            # Exact match: every answer position predicted correctly (non-answer positions ignored).
            c_rows.append(((logp.argmax(-1) == yc) | ~mask).all(dim=1).cpu())
            # Label-free confidence: average over answer positions of log max-probability.
            maxlogp = logp.max(-1).values
            f_rows.append(((maxlogp * mask).sum(1) / mask.sum(1)).cpu())
        correct.append(torch.stack(c_rows))
        confidence.append(torch.stack(f_rows))
    return {"correct": torch.cat(correct, dim=1), "confidence": torch.cat(confidence, dim=1)}


def summarize_length(correct: torch.Tensor, confidence: torch.Tensor, n: int) -> dict:
    """Apply the three stopping rules to one length's loop profile."""
    acc_per_k = correct.float().mean(dim=1)

    def pick(k: int) -> dict:
        # k is 1-based; per-problem correctness is kept for paired (McNemar) comparisons.
        return {"k": k, "acc": acc_per_k[k - 1].item(), "correct": correct[k - 1].int().tolist()}

    return {
        "acc_per_k": acc_per_k.tolist(),
        "confidence_per_k": confidence.mean(dim=1).tolist(),
        "oracle": pick(int(acc_per_k.argmax()) + 1),
        "max_confidence": pick(int(confidence.mean(dim=1).argmax()) + 1),
        "fixed_n": pick(min(n, correct.shape[0])),
    }


def evaluate_model(model, lengths, n_per_length: int, seed_base: int, extra_loops: int,
                   chunk_size: int, device, logger=None) -> dict:
    """Evaluate every length in `lengths`; returns {length: summary}."""
    model.eval()
    out = {}
    for n in lengths:
        # Each length has its own seeded test set, disjoint in RNG stream from training data.
        x, y = data.make_eval_set(n, n_per_length, seed=seed_base + n)
        prof = loop_profile(model, x, y, n + extra_loops, chunk_size, device)
        out[n] = summarize_length(prof["correct"], prof["confidence"], n)
        if logger:
            s = out[n]
            logger.info(f"eval n={n}: oracle acc={s['oracle']['acc']:.3f} (K={s['oracle']['k']}), "
                        f"max-conf acc={s['max_confidence']['acc']:.3f} (K={s['max_confidence']['k']}), "
                        f"K=n acc={s['fixed_n']['acc']:.3f}")
    model.train()
    return out


@torch.no_grad()
def output_examples(model, n: int, count: int, seed: int, k: int, device) -> list[dict]:
    """Decoded qualitative examples (input, target, prediction) for one length and loop count."""
    model.eval()
    x, y = data.make_eval_set(n, count, seed=seed)
    logits = model(x.to(device), torch.full((count,), k)).cpu()
    model.train()
    mask = y != data.IGNORE
    rows = []
    for i in range(count):
        rows.append({
            "input": data.decode(x[i][: 2 * n + 2]),
            "target": data.decode(y[i][mask[i]]),
            "prediction": data.decode(logits[i].argmax(-1)[mask[i]]),
        })
    return rows
