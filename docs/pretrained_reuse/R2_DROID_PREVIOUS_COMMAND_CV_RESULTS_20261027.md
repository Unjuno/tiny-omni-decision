# DROID previous-command baseline on fixed train folds

The [fixed protocol](R2_DROID_PREVIOUS_COMMAND_CV_PROTOCOL_20261027.md) reused
the same 275 train recordings and five outer folds as the prior temporal
comparison. For each held fold, a seven-by-seven transition table was fit only
on the other four folds, with one pseudocount per next-command class. Existing
sensor readout predictions were replayed on the same rows. No validation,
test, audit, encoder or future-action inputs were used.

| Input and predictor | Accuracy | Macro accuracy | NLL | Macro NLL | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|---:|
| Raw sensor-only head | 0.4036 | 0.1805 | 1.6425 | 2.2347 | 0.7568 | 0.0643 |
| Delta sensor-only head | 0.4182 | 0.1855 | 1.5712 | 2.1833 | 0.7239 | 0.0443 |
| Previous-command transition table | 0.9455 | 0.9002 | 0.3805 | 0.6080 | 0.1338 | 0.1319 |

The transition table uses **previous commanded action**, an extra input absent
from both sensor-only heads. It is a task baseline, not a controlled claim
that one architecture beats another. Of 275 episodes, 260 have the same next
and previous command: the transition table correctly predicts all 260. It
correctly predicts **0 of 15 changed-command cases**, where its NLL is 3.3368.
On those same 15 target-defined cases the raw and delta sensor heads correctly
predict 4 and 3, respectively; sample size is too small for a strong claim.
Target-defined stable/change subsets are descriptive, not an evaluation filter.

This strongly suggests that aggregate next-command Accuracy can reward simple
temporal persistence, while being unreliable when action actually changes.
The 94.55% result is not general decision quality, visual understanding, or
evidence that the four-modality target is met. Recordings came from a previously
inspected train cohort; folds do not prove new operators or sites. A useful next
task needs a prospectively specified source-independent evaluation with enough
change events and a comparison where all models receive the same history.

Private artifacts:
`C:/CodexArtifacts/pretrained-reuse-r2/droid-previous-command-cv-20261027-attempt-01/`.
Raw SHA-256 of `summary.json`:
`e20b3aceacb92977a700562a195aea6c7075f29ba385c5a5bd537662e0424e45`;
of `private-predictions.jsonl`:
`7a5809871bf3ad5c2f0219eab2eb1a638ab0ac57d2a2aeb06f4fa1aebd50999e`.
The output pins the existing previous-command data, outer predictions and folds,
metric implementation, protocol and script. No model weights were produced.
