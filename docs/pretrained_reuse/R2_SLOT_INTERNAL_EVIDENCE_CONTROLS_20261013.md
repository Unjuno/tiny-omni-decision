# Internal slot evidence controls: close this fitting path

Computed from preserved predictions, without model inference or training. Both
arms have identical 40 example records (20 complete initial/final video pairs).
The adjacent JSON records all input hashes, metrics and paired uncertainty.

| Head / input | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Original / normal | .425 | 2.914194 | .977441 | .465143 |
| Original / mismatched video | .475 | 4.401486 | .932683 | .434981 |
| Original / zero features | .300 | 1.103580 | .670065 | .058163 |
| Interaction / normal | .300 | 2.784886 | 1.141853 | .531306 |
| Interaction / mismatched video | .400 | 3.171957 | 1.046540 | .465892 |
| Interaction / zero features | .300 | 1.439974 | .901130 | .386824 |

Whole-video paired bootstrap, 10,000 replicates, seed17, 95% percentile intervals:
normal-minus-mismatch Accuracy is -.050 [-.250,.150] for the original head and
-.100 [-.250,.025] for the interaction head. Neither supports a resolved benefit
from correct video identity. NLL differences also have intervals including zero.
This does not prove the features contain no useful information.

NLL convention warning: stored aggregate metrics floor probabilities at 1e-12;
the existing bootstrap helper floors at 1e-15. The original head has extreme
probabilities, so its bootstrap NLL difference (-1.832680) differs from subtracting
the displayed aggregate NLLs (-1.487292). These are different clipped statistics;
do not equate them. The JSON explicitly preserves the bootstrap floor.

The zero-feature arm is an intervention on a trained model, not a separately
trained question-only baseline. Better NLL from that intervention illustrates
overconfidence; it does not establish a useful question-only product.

Decision: close additional fitting/capacity sweeps for these frozen slot features
and heads on these observed IDs. The previous aggregate-only comparison did not
establish evidence grounding, and this direct control does not repair that gap.
Do not use auxiliary easy questions to redefine success on temporal decisions.
Any subsequent route requires a prospectively fixed evidence/label contract and
a distinct intervention, preserving these results. No shared-Lens, compression,
product promotion or sealed evaluation is justified by these measurements.

## Matched NLL convention follow-up

The helper now accepts an explicit keyword-only `probability_floor`, preserving
the historical 1e-15 default. Regression tests check extreme probabilities,
aggregate equality, default compatibility and invalid floors (13 tests pass).
Recomputed both arms with 1e-12, the same convention as their aggregate metrics;
input file hashes are unchanged. See `R2_SLOT_INTERNAL_MATCHED_NLL_20261013.json`.
Original-head NLL difference is -1.487292 with interval [-4.887444,1.026715];
interaction-head difference remains -.387072 [-1.593878,.860630]. Point estimates
match stored aggregate subtraction within 1e-12. Both intervals still include zero.
The decision to close this fitting path is unchanged. No model was run or changed.

Reproduction: load `predictions.json` final normal/mismatch arrays and targets
from `examples.json` in each internal evaluation directory; call
`paired_video_bootstrap(normal, mismatch, targets, probability_floor=1e-12)`.
Compare its NLL difference with the normal-minus-mismatch NLL from `result.json`.
The previous results remain preserved under their explicitly different convention.
