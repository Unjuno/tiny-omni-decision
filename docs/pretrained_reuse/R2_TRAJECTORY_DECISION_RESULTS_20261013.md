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

## Train-only condition replay

Replayed the pinned best/final weights on198 training questions, with no fitting,
validation inference or checkpoint reselection. Initial/final text features are
numerically distinct (relative L2 range0.078768–0.082141); this does not prove
semantic adequacy.

| Checkpoint | Train Accuracy | Condition switches /99 | Changed-target both correct /53 |
|---|---:|---:|---:|
| Best epoch11 | 0.449495 | 0 | 0 |
| Final epoch50 | 0.742424 | 6 | 4 |

The final head fits all92 unchanged-target training questions but only55/106
changed-target questions. The contrast identifies conditional tracking/readout
as a remaining weakness even within train; it cannot distinguish insufficient
identity representation from insufficient readout or supervision. Do not infer
a universal encoder ceiling. Extra epochs are not justified by this overfitting
curve alone.

Runner:`scripts/diagnose_trajectory_train_condition.py`.
Artifact:`C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-trajectory-train-condition-01.json`.
Input hashes and checkpoint hashes verified; old artifacts preserved.

## Fixed train-only feature ablations

The same best/final checkpoints were replayed without fitting. Geometry is the
first128 channels; appearance is the remaining384. Misalignment rolls geometry
by one grid point within each clip, preserving each block's marginal values.
Zeroing is out-of-distribution, so these are dependence diagnostics, not clean
causal estimates or substitute validation quality.

| Checkpoint/control | Train Accuracy | Train NLL | Condition switches /99 |
|---|---:|---:|---:|
| attention/best/normal | 0.449495 | 1.040250 | 0 |
| attention/best/zero_geometry | 0.449495 | 1.094536 | 1 |
| attention/best/zero_appearance | 0.449495 | 1.053038 | 0 |
| attention/best/misaligned | 0.449495 | 1.040378 | 0 |
| attention/final/normal | 0.742424 | 0.495379 | 6 |
| attention/final/zero_geometry | 0.424242 | 1.353203 | 3 |
| attention/final/zero_appearance | 0.494949 | 1.084682 | 4 |
| attention/final/misaligned | 0.691919 | 0.722401 | 4 |

The final head depends on both blocks to fit train, and breaking their association
also worsens train NLL. This contradicts a claim that geometry is completely
ignored. It does not prove correct hidden-object identity tracking: changed-target
fit and development generalization remain weak. Best-checkpoint labels barely
respond, consistent with its previously observed lack of condition switches.
No additional validation inference or checkpoint selection was performed.

Runner:`scripts/diagnose_trajectory_train_ablation.py`.
Artifact:`C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-trajectory-train-ablation-01.json`.
Next representation work must test hidden-object identity evidence, since initial
appearance plus point motion does not explicitly encode what becomes visible
later. This is a hypothesis to investigate, not a demonstrated root cause.

## Initial-query timing check

Pinned train annotations contain99 marked occluder tracks;98 have an unmasked
frame0 box and all99 have a pre-cut observation. Therefore a blanket explanation
that the relevant covering objects enter after the initial query frame is not
supported. These are covering-object tracks, not an oracle label identifying the
hidden object's pixels. Missing annotation is not proof of invisibility, and
frame0 coverage does not establish identity tracking through occlusion.

Read-only timing inventory retained at
`C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-initial-query-coverage-01.json`.
The inventory records every train track's label, marked-occluder flag and first
unmasked annotated frame; annotation SHA256 is checked against the existing pin.
No new extraction/training is justified solely by the late-entry hypothesis.

## Exact-frame train tracking diagnostic

Read the existing train feature cache only. For each marked covering-object track,
require an unmasked frame0 annotation. Among grid queries inside that box, choose
the nearest to its center (oracle initialization). Compare saved xy only where a
nonzero sampled frame exactly matches an unmasked annotation; no box interpolation.
All feature SHA256 values and the pinned train annotation archive were checked.

Of99 train videos,71 have eligible later exact-frame comparisons, totaling108
points. The selected track falls inside the covering-object box at72/108 points.
Mean normalized center error is0.105969 versus0.149384 for retaining the initial
point. Using one mean error difference per video,48/71 improve; mean difference
is-0.040782. Video bootstrap with replacement,10,000 draws,NumPy default_rng17,
percentile95 interval:[-0.066221,-0.016773].

This confirms some usable motion information in the actual retained features.
It neither proves continuous identity retention between sparse observations nor
solves which covering object hides the target. Oracle initialization, varying
numbers of observations and incomplete temporal coverage limit this diagnostic.
Do not treat72/108 as Decision Accuracy or apply it to all356 videos.

Artifact:`C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-grid-track-exact-train-01.json`.
Contains per-video selected grid index, exact frames, tracked/static errors,
box-membership result, source hashes and explicit missing-coverage records.
No model inference, training, checkpoint selection or validation use was needed.
The fixed trajectory experiment therefore supports motion reuse but still fails
the conditional-semantic gate. Future work must explicitly connect target identity,
covering-object association and condition-dependent readout.

CI for documentation commit0fad303: both workflows completed successfully.

## Grounding/orientation feasibility check

Visually inspected fixed train video10103 at frames0,90,180,270,360,458,
all before the reveal cut. There are two covering glasses, consistent with the
upstream count question; the three ordinal answer options also come from the
upstream questions. Do not remove the third option or simplify the benchmark.
Contact sheet:`C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-train-visual-inspection-01/video_10103.png`.

Upstream grounded questions link the hidden object, its covering object and the
set of similar covering objects. An annotation-only diagnostic on all99 train
videos ranked the marked covering object's box-center x among that set at the
first/last common unmasked annotated pre-cut frame. Comparing zero-based ordinal
ranks with canonical question targets gives101/198 matches for screen-left order,
and134/198 for screen-right order. Both orientations were reported; neither is
adopted as a learned or product rule. The latter67.68% is not model Accuracy.

Even oracle identity plus a simple screen-x rank is therefore insufficient as a
surrogate task. Person-relative viewpoint, temporal endpoint mismatch, overlap
and object changes require review before using this supervision. These results
do not establish annotation errors. Do not feed grounded answer IDs into runtime
inputs or use them to select evaluation trajectories.

Full diagnostic rows, exact comparison frames, object counts and both ranks:
`C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-oracle-rank-train-01.json`.
No fitting, model inference, validation or sealed evaluation was performed.

## Visual review of oracle-rank mismatches

Diagnostic-only review selected the first three train records disagreeing with
screen-right order:10212,10387,10886. This is intentionally failure-selected and
must not estimate prevalence or quality. Contact sheet preserves frames0 and the
previously compared final common annotation (1050/480/630 respectively):
`C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-train-visual-inspection-02/frames.png`.

The views differ:10212 is an oblique table view,10387 has a row extending in depth
at the start and a markedly different screen orientation at the end,10886 shows
the covering glasses from the interacting person's side. Thus neither one global
screen-left nor screen-right transform is an adequate definition of the explicit
person-relative question. This observation does not verify every target or explain
all mismatches. Initial/ending game state can also differ from first/last available
annotation; the oracle-rank calculation remains a surrogate.

Across all99 rows, screen-left matched49 initial/52 final answers; screen-right
matched66 initial/68 final. Preserve both, not a per-example best orientation.
Do not train an ordinal head using these surrogate ranks. A future approach must
retain scene/person reference information and target-cover association rather
than assuming static camera coordinates. No extra quality experiment is launched
from these three visual observations alone.

## Prospective scene-feature reuse admission

Before any combined readout fit, compared existing VideoMAEv2 scene and TAPIR
trajectory caches. Their manifest pins and356 video-ID sets agree, but strict
temporal-endpoint equivalence failed on33 clips: every affected clip has
cut_frame=-1 and the scene plan ends one frame earlier than the trajectory cache.
No training was launched and no old feature was modified. This is a preprocessing
extent difference, not evidence of leakage or the cause of the quality failure.
Do not describe the two encoders as receiving identical frames.

Failure inventory:
`C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-scene-trajectory-admission-failed-01.json`.
Trace the original duration/end-index calculation before admitting a combined
experiment; explicitly retain or resolve this difference in a new protocol.

### Endpoint mismatch resolved as an explicit policy difference

`decode_window` in `scripts/prepare_av_correspondence_windows.py` uses the common
AV support ending no later than the last video PTS, then an exclusive end.
`plan_windows` preserves that exclusion. The trajectory extractor instead uses
all decoded frames when cut_frame=-1. For witness10212, last timestamps are
34.97249541590265 and35.00583430571762 seconds: scene ends at index1049, trajectory
at1050. This explains the observed one-frame endpoint difference.

A full second admission verified all356 IDs/splits/media bindings, both feature
files per video by SHA256 and finite values, matching manifest pins, and allowed
only zero delta or delta1 with cut=-1.33 cases have the latter. Original files
remain untouched. Report:
`C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-scene-trajectory-admission-02.json`.

Caches may be reused in a prospective combined-feature comparison with this
explicit difference fixed across its arms. They must not be called identical
preprocessing or evidence isolating encoder architecture alone. No additional
extraction is required to test whether the existing scene representation adds
useful information; usefulness remains unproven.
