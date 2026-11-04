# Summary of all experiments so far

## In plain language

A **looped transformer** solves a problem by running the same small neural-network block over and over. It is a bit like a person repeating one mental step: "add the next pair of bits, carry if needed". Because the block is reused, the model can in principle handle longer problems just by looping more times. Whether it actually does is the **length generalization** question.

**Training** finds weights that make the loss (the error on training problems) small. Many different weight settings can make the loss small:
- Some sit in a **sharp** valley of the loss landscape, where a tiny change in the weights makes the error shoot up.
- Others sit in a **flat** valley, where small changes barely matter.

An old idea says flat solutions tend to work better on new data. Measuring and penalizing **curvature** means pushing training toward flat valleys. Curvature here is the trace of the Hessian, tr(H): the sum of the second derivatives of the loss along every weight direction.

**What the experiment does.** It trains two otherwise identical looped transformers to add binary numbers of up to 19 bits, then tests both on 20- to 60-bit numbers (every 5th length). One is trained normally. The other pays an extra cost in its loss for curvature. Both start from the same weights and see the same training problems in the same order. Training status: see the headline below.

**What we can and cannot conclude.** With only one model of each kind, we can describe what happened. We cannot say it would happen again with different random seeds: looped models are known to vary a lot from seed to seed. The preregistered verdict is in the headline below.

<!-- BEGIN:headline -->
Stage 1 is in progress (finished arms: none).
<!-- END:headline -->

## Key terms

| Term | Meaning |
|---|---|
| Looped (weight-tied, recurrent-depth) transformer | A transformer whose block of layers is applied K times with shared weights, instead of stacking K different blocks |
| Input injection | Adding the input embedding again at every loop iteration: h_t = Block(h_{t−1} + e) |
| NoPE | No positional embeddings; position information comes only from the causal attention mask |
| Full-output prediction | The answer is predicted all at once at the placeholder positions after the last iteration, not one token at a time |
| In-distribution (ID) / out-of-distribution (OOD) | Test problems with lengths seen in training (1–19 bits) / longer than any training problem (20–60 bits) |
| Exact match | All n+1 sum bits and the end token are correct |
| Oracle loop count | For each test length, the loop count K with the best accuracy (uses labels; a per-length variant of Kuo et al. 2026's oracle, which picks the best K per example) |
| Max-confidence stopping | Picking K by the model's own confidence, without labels (Fan et al. 2024) |
| Hessian H | Matrix of second derivatives of the loss with respect to the weights; it describes the local curvature |
| tr(H), trace | Sum of the Hessian's diagonal, i.e. of its eigenvalues; a common "sharpness" summary |
| Hutchinson estimator | tr(H) ≈ vᵀHv for random ±1 vectors v; unbiased, with Hv computed by double backpropagation |
| Dominant eigenvalue | The eigenvalue of H with the largest magnitude: the curvature along the single sharpest direction |
| Fisher trace | Trace of the Fisher information = Gauss–Newton matrix; a positive semi-definite relative of tr(H) |
| Adaptive sharpness | Average loss increase when every weight is jiggled in proportion to its own size; unchanged by rescalings of the weights that leave the network's function unchanged |
| NSO | Noise Stability Optimization: averages gradients at θ+U and θ−U; in expectation, up to O(σ⁴) terms, this is the gradient of L + (σ²/2)·tr(H) (deferred to Stage 2) |
| SAM | Sharpness-Aware Minimization: minimizes the worst-case loss in a small ball around θ (deferred to Stage 2) |
| Wilson interval | Confidence interval for a proportion; here it covers test-set sampling only |
| McNemar test | Paired test of two models on the same test problems, using the problems only one of them got right |

## Networks in Stage 1 (training status in the headline above)

| Network | Architecture | Params | Training algorithm | Objective | Data / curriculum | Domain | ID acc. | OOD acc. (primary) |
|---|---|---|---|---|---|---|---|---|
| `stage1_baseline` | Looped GPT-2 block: 3 layers, d=256, 8 heads, NoPE, input injection; K ~ n±5 (≤ 20) loops | 2.37M | AdamW, cosine schedule, 30k steps, batch 64 | Cross-entropy on answer tokens | Binary addition, n ∈ [1,19], linear length curriculum | Algorithmic length generalization | see table | see table |
| `stage1_hutchinson` | Same | 2.37M | Same | Cross-entropy + λ·tr(H) (Hutchinson; λ = 5×10⁻⁵, applied every 8 steps ×8) | Same | Same | see table | see table |

<!-- BEGIN:results_table -->
Pending.
<!-- END:results_table -->
