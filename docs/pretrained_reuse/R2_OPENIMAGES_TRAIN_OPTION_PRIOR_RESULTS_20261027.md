# Four-choice train candidate option-only diagnostic

The [fixed protocol](R2_OPENIMAGES_TRAIN_OPTION_PRIOR_PROTOCOL_20261027.md)
constructed four choices for all 257 saved train-candidate images, retaining
their selected target and using only verified in-scope negatives. It generated
five deterministic folds containing 55, 50, 55, 41 and 56 examples. No image
pixels, validation labels, official test data or model weights were loaded.

| Train-candidate diagnostic | Accuracy | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|
| Option-only, four folds to predict the fifth | 0.4397 | 1.2469 | 0.6733 | 0.0887 |
| Uniform probabilities, deterministic first-position tie | 0.2490 | 1.3863 | 0.7500 | 0.0010 |

The uniform random-choice expected Accuracy is exactly 0.25; the table's
0.2490 is the observed deterministic tie-breaking accuracy. This is a small
development diagnostic, not a blinded model evaluation. The fixed option
construction considers all train-candidate rows before the folds are formed,
so its results do not establish independence of the task-construction process.
They do show that a simple option-only prior retains substantial predictive
signal after excluding out-of-scope choices. This particular four-choice task
should not be treated as a clean image-grounding benchmark.

Class roles remain asymmetric: Toy occurs 23 times as target versus 16 as a
negative; Animal 26 versus 20; Window 24 versus 143; Building 27 versus 129.
Only 19 source images offered more than three in-scope verified negatives, so
the prescribed greedy choice rule had limited freedom to balance roles. Do not
alter the existing evaluation or select rows after seeing the prior score.
Further work must independently construct stronger controls or obtain broader
label coverage; a reported model improvement must exceed option-only controls
and survive image ablation. Ownership, provenance and split gates remain open.

Local artifacts:
`C:/CodexArtifacts/pretrained-reuse-r2/openimages-train-option-prior-20261027-attempt-01/`.
Raw SHA-256 of `summary.json`:
`6db771811ea9655726fd07e6415010aa1c9ff823cc3f6d31b107b924cb826450`;
of `private-predictions.jsonl`:
`df80fe10fe4c41af2192afc9a4f8e0b9a4751f7d7df26fe2d49ebe7914751bba`.
The summary pins candidate, media-manifest, scope, protocol and script hashes.
