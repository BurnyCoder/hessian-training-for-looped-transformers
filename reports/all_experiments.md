# Experiment index

| Experiment | Evidence and status | Record |
|---|---|---|
| Throughput and allocator benchmarks | Archived training-step timings on the original machine | [Timing records](../results/benchmark/) |
| Historical smoke test | Two short plumbing runs, retained separately from the pilot. The corrected replay uses actual stored counts. | [Archive](../results/audit-2026-11-26/) · [Reconstructed summary](../results/audited_smoke_summary.json) |
| Stage 1 paired pilot | One seed planned, ordinary and intermittent-penalty arms | [Experiment record](stage1_experiment.md) |
| Independent operator check | Scalar CPU check of native and composite LayerNorm | [Operator record](../results/audit-2026-11-26/layernorm_operator.json) |
| Stage 2 | Registered, deferred. Paired-seed superiority, practical bounds, curvature, variability and exploratory associations. | [Amendment](preregistration.md) |

<!-- BEGIN:headline -->
**Stage 1 accuracy comparison and final trace check are unevaluated.** The baseline history ends at logged step 6,000 of a 30,000-step budget and includes interim accuracy checks. No Hutchinson run is recorded.
<!-- END:headline -->

<!-- BEGIN:results_table -->
Final accuracy measurements pending.
<!-- END:results_table -->
