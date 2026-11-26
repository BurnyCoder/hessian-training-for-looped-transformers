# Research summary

The experiment asks whether changing how a network learns addition helps it answer longer problems. A looped transformer repeats one set of layers. The proposed treatment occasionally penalizes a measurement of how the training loss bends as its weights change. A lower signed trace could reflect less positive curvature, more negative curvature, or both. [Method definitions](../docs/methods.md)

<!-- BEGIN:headline -->
**Stage 1 accuracy comparison and final trace check are unevaluated.** The baseline history ends at logged step 6,000 of a 30,000-step budget and includes interim accuracy checks. No Hutchinson run is recorded.
<!-- END:headline -->

<!-- BEGIN:results_table -->
Final accuracy measurements pending.
<!-- END:results_table -->

**How to read the evidence.** A success requires all sum bits and the end token. For the primary score, the evaluator tries several iteration counts, selects one with the help of the correct answers at each operand width, then averages equally across widths. The fixed-depth rule can choose its iteration count before seeing the answers. Its intervals describe variation from drawing new test problems for the same fitted model. The number of test problems does not give the precision of a comparison across training seeds. [Protocol](preregistration.md) · [Selection bias](https://jmlr.org/papers/v11/cawley10a.html)

The evidence audit found a denominator error in the original smoke summary, identified previously seen examples in the planned final test stream, and reproduced a derivative discrepancy using a scalar tensor operation. These observations concern the measurement pipeline. The comparison between trained models still requires final records from both arms. [Independent evidence and numbers](reviews/2026-11-26-scientific-audit.md) · [Experiment record](stage1_experiment.md)
