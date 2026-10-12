# Frozen trajectory Decision results — 2026-10-13

Execution completed; candidate not adopted. This is reused development validation,
not a blind evaluation. The fixed protocol is in
[R2_TRAJECTORY_DECISION_PROTOCOL_20261013.md](R2_TRAJECTORY_DECISION_PROTOCOL_20261013.md).
No additional tuning, seed search, sealed evaluation or legacy artifact overwrite.

All356 clips (99 train /257 validation) passed feature SHA256, shape256×512,
finite-value, frame-index endpoint and cross-split prepared-content checks.
Training admission also verified pinned manifests, media identity and text routing.
Frozen TAPIR plus frozen MiniLM feed a164,032-parameter head; only this head was trained.
Total component parameters:48,626,183. This is a video task component, not a complete Omni model.

## Fixed validation results (514 questions)

| Control | Accuracy | NLL | Brier | ECE | Both conditions correct |
|---|---:|---:|---:|---:|---:|
| normal | 0.389105 | 1.083386 | 0.658297 | 0.059479 | 0.206226 |
| zero | 0.389105 | 1.098368 | 0.666504 | 0.055197 | 0.206226 |
| mismatch | 0.389105 | 1.095432 | 0.665522 | 0.068372 | 0.206226 |

Best is epoch11 by predeclared minimum validation NLL. Changed-target pairs:
0/127 jointly correct. Final-epoch Accuracy43.19% is not the selection rule;
its NLL1.544272 and ECE0.229773 deteriorate while training NLL falls.
There is no evidence here of useful conditional temporal decision quality.

## Learning curve

| Epoch | Train Accuracy | Train NLL | Validation Accuracy | Validation NLL |
|---|---:|---:|---:|---:|
| 1 | 0.449495 | 1.065245 | 0.389105 | 1.090392 |
| 5 | 0.449495 | 1.059195 | 0.389105 | 1.085806 |
| 11 | 0.449495 | 1.040250 | 0.389105 | 1.083386 |
| 25 | 0.722222 | 0.460522 | 0.365759 | 1.662021 |
| 50 | 0.742424 | 0.495379 | 0.431907 | 1.544272 |

## Paired uncertainty

Video-cluster bootstrap,10,000 draws,seed17,percentile95 intervals; both questions stay together. Checkpoint selection and repeated development-set use are not corrected by these intervals.

- Candidate minus videomaev2 attention: NLL[-0.01672709929510454, 0.004493775895588269], Accuracy[0.0, 0.0].
- Candidate minus videoprism attention: NLL[-0.019030643892743115, 0.007673635464150529], Accuracy[0.0, 0.0].
- Mismatch minus normal NLL:[-0.000557,0.024832]. Neither comparison establishes an improvement at this interval level.
- Exact predicted-label agreement with both prior attention heads gives zero Accuracy difference. Probability changes alone do not establish successful grounding.

## Execution and retained evidence

Feature extraction per-clip time sum:6152.293s; elapsed:6190.888s. Peak allocated GPU memory:10704988160bytes. RTX3080Laptop16GB, FP32/TF32off. CPU head training/evaluation:34.240s. Local compute only; no cloud charge.

Run650 updates /9,900 question presentations; selected checkpoint2,178 presentations.
Unique training198 questions /99 videos. Best/final/untrained checkpoints and predictions,
50-epoch history, config and hashes retained. Exact best reload passed.
Extraction source0e225e8; training sourceca71824. No encoder update or compression.

Artifacts under `C:/CodexArtifacts/pretrained-reuse-r2/`:
- `perception-occlusion-trajectory-features-01/`
- `perception-occlusion-trajectory-readout-01/`
- `perception-occlusion-trajectory-report-videomaev2-01.json`
- `perception-occlusion-trajectory-report-videoprism-01.json`

Reproduction: `python -m scripts.report_trajectory_readout --run <readout> --reference <prior-readout> --output <fresh-report.json>`.
Reporter validates artifact hashes, prediction IDs/targets, metric reproduction,
and the complete50-epoch selector before computing intervals.

This closes the fixed trajectory comparison without promoting it. It does not
establish a TAPIR or model-family ceiling. Next investigation should isolate
object identity/condition grounding using training-only evidence before another
quality run; simply extending this head's epochs is not supported. All broader
Omni quality/fusion/release gates remain open.
