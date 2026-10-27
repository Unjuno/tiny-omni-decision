# DROID prior-initialized nested probe: result

The [fixed procedure](R2_DROID_PRIOR_INIT_PROTOCOL_20261028.md) completed five
outer folds, using the same previously inspected 275 train recordings. Each
inner and outer fit estimated its own prior using only its fit labels. Both
the initialization and zero-update candidate differ from the prior experiment.

| Inputs | Accuracy | NLL | Brier | ECE-10 | Selected updates across five folds |
|---|---:|---:|---:|---:|---|
| Sensor | 36.36% | 1.6578 | 0.7658 | 0.0312 | 1, 1, 1, 1, 1 |
| Sensor + video | 37.09% | 1.6906 | 0.7750 | 0.0004 | 0, 0, 0, 0, 0 |
| Sensor + text | 37.45% | 1.7197 | 0.7838 | 0.0590 | 1, 0, 1, 0, 0 |
| Sensor + video + text | 37.09% | 1.6906 | 0.7750 | 0.0004 | 0, 0, 0, 0, 0 |

The zero-update predictions are the outer-training class priors, not evidence
of successful video use. Sensor-only yields lower NLL than the prior control
but slightly lower accuracy. Text's tiny aggregate accuracy increase is
accompanied by worse NLL; no multimodal complementarity is established. The
video-containing configurations' selected zero updates cannot demonstrate
video effects. Earlier development reuse, small classes and one update chosen
at the boundary limit interpretation. This does not show a fundamental
architecture or data ceiling.

Output:
`C:/CodexArtifacts/pretrained-reuse-r2/droid-prior-init-20261028-attempt-01/`;
`result.json` SHA-256
`0d79b769a6ab4558a7ea22592d19e873eaeff9bdca42d58a76e9b2d55edaf168`.
All fold scores, update choices and OOF predictions are saved. Two CPU tests,
including exact train-prior initialization and held-label isolation, and Ruff
passed. No validation, sealed audit, new encoder training or product promotion.
