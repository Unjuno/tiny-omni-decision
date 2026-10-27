# Error confidence and multi-answer metric audit

This is a descriptive audit of the previously observed 2,593 official-train
OOF examples and the existing baseline checkpoints, not an independent test
or model-selection experiment. The script verifies the fixed run receipt,
history and prediction hashes and pairs the same records and targets across
epochs. No new model inference, training, or sealed audit was performed.

| Epoch | Errors | Errors with top confidence > 0.9 | Mean error confidence | Mean error NLL | Fraction of total NLL on errors |
|---:|---:|---:|---:|---:|---:|
| 4 | 1,228 | 20 | 0.444 | 2.292 | 81.8% |
| 8 | 1,114 | 137 | 0.590 | 3.044 | 89.4% |
| 32 | 1,139 | 560 | 0.815 | 7.838 | 98.7% |

From epoch 4 to 32, 356 previously incorrect predictions became correct,
while 267 previously correct predictions became incorrect; 872 were wrong at
both epochs. Therefore the overall accuracy gain does not mean that exactly
the same examples merely became more confident. Nevertheless, error
confidence rises sharply, consistent with the measured calibration decline.
These are descriptive counts on reused development folds and cannot establish
causality, generalization to new sources, or a universal training limit.

Of the 2,593 examples, 692 have no visible target, 799 have one, and 1,102
have two or more. The existing grouped NLL scores the total probability of the
set of correct candidates. Accuracy counts any correct candidate. The current
Brier score instead assigns equal target probability to every correct
candidate, and ECE compares the winning candidate's probability with whether
that candidate is correct. With multiple acceptable choices these quantities
answer different questions: a uniform target is an evaluation convention,
not evidence that each acceptable answer has equal probability, and the
winning candidate probability is not the probability mass of the full
acceptable set. Do not treat these metrics as interchangeable. A future
independent task evaluation should define its probabilistic target and
calibration event before comparing models; changing the metric does not
retroactively make this development set blind.

Output: `C:/CodexArtifacts/pretrained-reuse-r2/perception-source-confidence-audit-20261027-attempt-02/result.json`,
SHA-256 `11bb191cd8bb9f33adaddda274a330ab29107890088181636b71317bda826cab`.
