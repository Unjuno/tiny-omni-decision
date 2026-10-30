# Training-only counting diagnostic

## Scope and verified calculation

No fitting, checkpoint selection, validation predictions, reporting images or audit payloads.
The existing cropped global checkpoint is frozen; it was previously selected using
development validation. This diagnostic uses only8,424 original training images and labels.
Source commit `c452965`; exact source/input hashes are in the companion JSON.

For fixed candidate embeddings, the bilinear scorer is affine in the observation:
`score_k(x) = (Wv*x+bv)^T*(Wc*ck+bc)/sqrt(h) = Ak*x+dk`.
Float64 unit tests verify this identity. On all8,424 actual training examples,
float32 affine versus original maximum logit difference is7.6294e-5 and maximum
probability difference8.7619e-6. Both pass predeclared tolerances; zero answers change.
This is not bitwise equivalence or a guarantee about every unseen input near a tie.

Increasing factor width alone does not introduce nonlinear functions of the image
features for these fixed options. It can still change rank constraints, optimization
and parameter sharing; this result does not establish a capacity ceiling.

## Exact counting performance on training data

Accuracy68.0199%, NLL0.726062, Brier0.411406, ECE0.017059. These reproduce the
original selected checkpoint training summary within floating point tolerance.
5,730 correct;2,694 incorrect. Overestimates2,001; underestimates693.
Mean absolute count error0.3551; mean signed(predicted minus true) error+0.1749.

Of the errors,2,409 differ by one,274 by two,10 by three and1 by four.
Adjacent errors remain errors: no relaxed metric replaces exact Accuracy.

| True count | Images | Exact Accuracy | NLL |
|---|---:|---:|---:|
| 1 | 832 | 1.000000 | 0.010540 |
| 2 | 829 | 0.921592 | 0.236687 |
| 3 | 841 | 0.828775 | 0.429851 |
| 4 | 804 | 0.753731 | 0.667632 |
| 5 | 837 | 0.463560 | 1.104278 |
| 6 | 888 | 0.546171 | 0.986039 |
| 7 | 855 | 0.603509 | 1.039983 |
| 8 | 834 | 0.400480 | 1.253605 |
| 9 | 855 | 0.389474 | 1.168545 |
| 10 | 849 | 0.912839 | 0.329092 |

## Confusion matrix: rows true, columns predicted1..10

| True | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 832 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 20 | 764 | 45 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 28 | 697 | 116 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 49 | 606 | 132 | 17 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 1 | 79 | 388 | 352 | 16 | 1 | 0 | 0 |
| 6 | 0 | 0 | 0 | 6 | 58 | 485 | 317 | 19 | 2 | 1 |
| 7 | 0 | 0 | 0 | 1 | 4 | 83 | 516 | 158 | 91 | 2 |
| 8 | 0 | 0 | 0 | 0 | 1 | 14 | 147 | 334 | 263 | 75 |
| 9 | 0 | 0 | 0 | 0 | 0 | 2 | 26 | 100 | 333 | 394 |
| 10 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 5 | 68 | 775 |

## Interpretation and next diagnostic

The predominant error is an overestimate on this training set, which is inconsistent
with explaining all errors simply as missing objects outside the crop. It does not
exclude cropping, occlusion, feature limitations or label ambiguity in some examples.
Class counts range804..888; do not infer subgroup causality from the accompanying
color/shape/texture breakdown because count composition and appearance can confound it.
Closed options1..10 constrain the signs at endpoints; strong performance at1 and10
does not prove general counting or extrapolation beyond this range.

Before increasing encoder size or calling the representation inadequate, distinguish
optimization/parameterization from nonlinear-readout need. A bounded training-only
fit of an explicitly affine classifier on the same frozen features is the next
appropriate diagnostic. Compare training loss/convergence, not new validation scores.
No such fit was run in this diagnostic; no new quality checkpoint is promoted.

## Artifacts and checks

`C:/CodexArtifacts/pretrained-reuse-r2/shared-count-training-diagnostic-01` contains
all8,424 training predictions, hashes, config, subgroup metrics and confusion matrix.
CPU analysis loop2.642s; no GPU training or paid compute. Original weights unchanged.
581 CPU tests pass,1 skipped; Ruff and manifest/catalog validation pass. Final CI
is checked after push. The general Omni95-98% objective remains unmet.
