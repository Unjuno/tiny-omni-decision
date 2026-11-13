# Scene-conditioned trajectory result — 2026-10-13

Fixed50-epoch run completed; not adopted. Same observed development set, not blind.
No legacy artifacts changed. Sourcecdbe1fe. Best epoch4 by fixed NLL; exact reload
passed. Head188,608 parameters; CPU wall51.167s; no GPU training/cloud charge.
99 unique train videos/198 questions;650 updates/9,900 presentations overall,
792 presentations at selected checkpoint. Both encoders and text features frozen.

| Control | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| normal | 0.389105 | 1.087580 | 0.660849 | 0.030877 |
| zero | 0.389105 | 1.098319 | 0.666471 | 0.054974 |
| mismatch | 0.389105 | 1.090220 | 0.662295 | 0.047540 |
| scene_mismatch | 0.389105 | 1.087270 | 0.660597 | 0.033397 |
| trajectory_mismatch | 0.389105 | 1.091164 | 0.662980 | 0.069919 |
| zero_scene | 0.389105 | 1.087006 | 0.660411 | 0.027080 |
| zero_trajectory | 0.389105 | 1.098319 | 0.666471 | 0.054974 |

Every control has0/127 changed-target pairs jointly correct. Scene removal and
scene mismatch do not harm these selected metrics. The trajectory-only reference
has NLL1.083386 versus1.087580 here. There is no observed useful improvement from
this pooled-scene addition; do not infer all viewpoint representations are useless.

| Epoch | Train Accuracy | Train NLL | Valid Accuracy | Valid NLL |
|---|---:|---:|---:|---:|
| 1 | 0.449495 | 1.064489 | 0.389105 | 1.090308 |
| 4 | 0.449495 | 1.057501 | 0.389105 | 1.087580 |
| 10 | 0.575758 | 0.938071 | 0.352140 | 1.227336 |
| 25 | 0.727273 | 0.529731 | 0.344358 | 1.789922 |
| 50 | 0.813131 | 0.428202 | 0.350195 | 2.489246 |

Later fitting worsens validation NLL. Close this fixed capacity/scene comparison
without a seed or hidden-size sweep. Paired uncertainty analysis is complete below.
All checkpoints, config/input hashes, histories and predictions are under
`C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-scene-trajectory-readout-01/`.
Protocol:R2_SCENE_TRAJECTORY_PROTOCOL_20261013.md. Prior cache endpoint difference
is retained explicitly. Product and multimodal-quality gates remain unmet.


## Paired uncertainty and final disposition

All saved artifact hashes, prediction IDs/targets, reconstructed metrics and the
50-epoch NLL selector were verified. Video-cluster paired bootstrap uses10,000
draws,seed17,percentile95 intervals. It does not correct selection bias from this
reused development set.

- Scene candidate minus trajectory-only NLL:[-0.005452,0.013696].
- Accuracy difference:[0,0]; both-correct difference:[0,0].
- Scene mismatch minus normal NLL:[-0.001633,0.001010].

No demonstrated gain or scene sensitivity supports adoption. Close this fixed
comparison; do not sweep seeds, capacity or calibration on these results.
This conclusion applies to the pooled-scene query-conditioning implementation,
not all visual context representations. General Omni semantic quality remains
unmet, and compression/shared-Lens work is not justified by this outcome.

Reproduction:`python -m scripts.report_trajectory_readout --run <scene-run> --reference <trajectory-run> --output <fresh-report>`.
Saved report:`C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-scene-trajectory-report-01.json`.
