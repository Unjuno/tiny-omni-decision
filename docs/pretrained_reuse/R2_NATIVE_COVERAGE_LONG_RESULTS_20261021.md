# R2 native unique-coverage long-budget results — 2026-10-21

## Decision

The preregistered fresh-coverage arm completed all 32,768 optimizer updates and
passed exact best/final checkpoint replay. The run is an engineering comparison,
not a product candidate. More unique data improved the final macro Accuracy
relative to the matched repeat control, but the selected checkpoint was slightly
worse on the fixed macro-NLL selector. The Video result is uncertain: validation
Accuracy rose only modestly, video-cluster intervals include zero, and Video NLL
and ECE became severely worse late in training. Do not promote this readout or
call the Video quality target achieved.

## Question and fixed conditions

The question was whether a 32,768-update fresh-coverage run changes the result
when compute, initial weights, validation IDs, model, LR schedule and evaluation
rules match the existing repeat-control run. This isolates sampler coverage
more cleanly than the earlier 2,048-step fresh arm, whose cosine horizon was
different and is therefore descriptive only.

- Branch: `codex/r2-native-coverage-long-20261021`
- Source commit used by the completed runs: `cb68d786e15d94152960e20db7081de11b94fa24`
- Config: `configs/pretrained_reuse/native_unique_coverage_long_v1.json`
- Effective config SHA-256: `7a6a026d82a121187ef3f252846c7c1b4101f141ca2790e880147b78445a2cae`
- Seed 17; fresh original native readout initialization; no warm start or resume
- Same frozen original-crop feature store and 24,346 validation IDs/order as the
  repeat control; validation ID hash:
  `db7d0f1d09319a48a7725ed36ccb515cdcc85a409f019400cf0062be38768f30`
- Same 32,768-step AdamW schedule in both arms: 8 microbatches/update, LR
  `0.001`, 62 warmup updates, full-horizon cosine, weight decay `0.01`, clip
  norm `1.0`. The LR-plan SHA is identical in both runs:
  `14234584342acb62df7d3987fdc835ec7beec3b04f5070ae864a9eac0c6becc9`.
- Validation points: 0, 512, 1,024, 2,048, 4,096, 8,192, 12,288, 16,384,
  24,576 and 32,768. Performance early stopping was disabled.
- 889,345 trainable readout parameters; 196,254,874 total pre-quantization
  parameters. No encoder or feature updates.
- Sealed audit was not loaded. No data, artifact or checkpoint was uploaded to
  Git; model outputs remain in the local artifact directory.

The matched repeat control is `native-long-budget-fit-01`, result hash
`645bedbe73ed508fad68c4f636a41df6c8630cd996641c44ce2a87d31550250b`, source
commit `8756a09f582554c0321ed5ece71846d6b19469cd`. Its best step was 8,192;
the new arm also selected step 8,192. The fixed hierarchical macro-NLL selector
was not changed.

## Execution and artifacts

The CPU-only preflight passed against 1,525 pinned files, including six source
files verified from their receipt-pinned Git blobs. Corpus counts were 100,580
train and 24,346 validation rows. The fresh-pool hash was
`a9b8bb1ce7019d0cd392333a7b855fddf5c1ad286802a9702af92ec81f027f16`; the
sample-order hash was
`3968c4fb4668548071556515cf126b832e7d65a304d6e8ecf5ac0c2cffd57995`. The
preflight recorded the full source/task bucket counts, depletion and LR curve
outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\native-coverage-long-preflight-04`.
Its status was `PASS_NATIVE_COVERAGE_LONG_CPU_PREFLIGHT`.

The separate 32-update smoke passed in 53.7 seconds (result SHA-256
`06d3b25d01b79df89af74bccb1c84041afffe59dd5a95f60afd61cb6be9a0421`). The
main fit ran from 2026-10-21 03:36:36 to 04:20:52 UTC and completed normally at
step 32,768. It used 1,878.9 seconds of optimizer time and 730.8 seconds of
evaluation time (2,655.9 seconds total). Peak allocated VRAM was 199.5 MiB,
peak reserved VRAM 312.0 MiB, and peak process working set 3.49 GiB. The
best/final reload replay passed in 192.9 seconds with exact saved-score parity
(`replay_max_abs=0`). Hardware/software: RTX 3080 Laptop 16 GiB, Windows,
Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, NumPy 2.4.6, float32. Cloud
compute and cloud cost: none. The run used the existing HotUGPU guard, which
waited for Orbitune's cooperative idle-yield and then held a 3,200 MiB
reservation below the supervisor's 14,000 MiB free-memory launch threshold.

Fit result SHA-256:
`c93da50addd7e92760cfe1878b126c2fcf8953612d8afd2d106d86d9cec779bf`.
Selected adapter/readout (`best.pt`, step 8,192) SHA-256:
`0ae36cd35e4b6cc187a067af8a9bd24968956cd10a927aa068f000472609de93`.
Final weights (`final.pt`, step 32,768) SHA-256:
`f46a754a28dace2efcde5c43d04e03c689d0260c3caf3313d8d0d7bf35cea59e`.
Learning-curve SHA-256:
`3680b03def5c0dc122abee8001e4a33d5f139d148574cd484041468d924bf244`.
The replay result is under
`C:\CodexArtifacts\pretrained-reuse-r2\native-coverage-long-fresh-replay-01`.

## Sampling actually consumed

Both arms consumed 262,144 examples (32,768 × 8), with 65,536 presentations per
modality. The repeat pool contained 6,656 unique questions, 3,114 unique media
assets and 3,382 groups. The fresh pool contained 12,800 unique questions,
5,385 unique media assets and 5,461 groups. It added 2,048 unique Text, 2,048
Image and 2,048 Audio questions; it added **zero** Video questions. Both pools
were fully visited by step 2,048.

| Measurement | Repeat control | Fresh coverage |
|---|---:|---:|
| Unique questions in full run | 6,656 | 12,800 |
| Unique questions by selected step 8,192 | 6,656 | 12,800 |
| Consumed examples by selected step 8,192 | 65,536 | 65,536 |
| Repeated presentations by selected step | 58,880 | 52,736 |
| Consumed examples in full 32,768-step run | 262,144 | 262,144 |
| Repeated presentations in full run | 255,488 | 249,344 |
| Unique media assets in full pool | 3,114 | 5,385 |

Fresh unique-question composition was Text/Image/Audio/Video = 4,096 / 4,096 /
4,096 / 512. The full-run source presentations were Text condition-relevance
21,856, Text programmatic 21,824, Text original 5,952, Text curriculum 15,904,
Clevr-4 65,536, Lombard 32,768, STARSS 32,768 and CLEVRER 65,536. CLEVRER
presentations were evenly split between static descriptive and temporal
descriptive tasks (32,768 each). There were 103 sampler buckets; 8 exhausted
before the full budget. The video source/task mix did not include explanatory,
predictive or counterfactual tasks.

At the selected step, all 12,800 unique questions had already been consumed;
the selected checkpoint was not selected before fresh examples entered the
training stream. At the full-run endpoint the same 12,800-question pool had been
presented 262,144 times in total, including 249,344 repeats.

## Learning curve

The table reports macro metrics (equal modality/source-task macro aggregation)
and the minimum modality-macro Accuracy. It is separate from the
sample-weighted overall metrics in `result.json`.

| Step | Macro Accuracy | Macro NLL | Macro Brier | Macro ECE | Minimum modality-macro Accuracy |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.2959 | 1.7171 | 0.7460 | 0.0762 | 0.1436 |
| 1,024 | 0.2924 | 1.7147 | 0.7474 | 0.0887 | 0.1499 |
| 2,048 | 0.3157 | 1.7376 | 0.7541 | 0.1156 | 0.1918 |
| 4,096 | 0.3376 | 1.6709 | 0.7313 | 0.0980 | 0.1906 |
| 8,192 | 0.3638 | 1.6591 | 0.7215 | 0.1322 | 0.1945 |
| 12,288 | 0.3985 | 1.7830 | 0.7259 | 0.1612 | 0.2005 |
| 16,384 | 0.4243 | 2.1362 | 0.7366 | 0.1863 | 0.2130 |
| 24,576 | 0.4667 | 3.3639 | 0.7418 | 0.2378 | 0.2298 |
| 32,768 | 0.4877 | 4.1994 | 0.7310 | 0.2458 | 0.2383 |

Accuracy continued to rise while macro NLL/ECE deteriorated sharply after
8,192. The fixed 6,656-question original-pool train probe at 32,768 had macro
Accuracy 0.7427 and macro NLL 0.6735 versus validation macro Accuracy 0.4877
and NLL 4.1994. The separately evaluated full 12,800-question train union at
the endpoint had macro Accuracy 0.7389 and macro NLL 0.6842. The latter is an
in-sample endpoint diagnostic, not a selected validation metric.

The run emitted explicit warnings at steps 2,048, 4,096, 12,288, 16,384,
24,576 and 32,768: train-probe NLL fell while validation macro NLL rose, or the
aggregate Accuracy gain coincided with weakest-modality degradation. No warning
caused early stopping.

## Same-validation comparison

All entries use the same ordered 24,346 validation IDs and preprocessing.
`Repeat` is the existing 32,768-step repeat arm; `Fresh` is this run. The first
row is the selected checkpoint in both arms; the second is the fixed final-step
comparison.

| Step | Arm/checkpoint | Macro Accuracy | Macro NLL | Macro Brier | Macro ECE | Weighted Accuracy | Weighted NLL |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8,192 | Repeat selected | 0.3742 | 1.6302 | 0.7057 | 0.1261 | 0.4170 | 1.5496 |
| 8,192 | Fresh selected | 0.3638 | 1.6591 | 0.7215 | 0.1322 | 0.3925 | 1.5530 |
| 32,768 | Repeat final | 0.4570 | 4.8481 | 0.8155 | 0.3116 | 0.4744 | 2.4270 |
| 32,768 | Fresh final | 0.4877 | 4.1994 | 0.7310 | 0.2458 | 0.5031 | 1.7695 |

The selected fresh checkpoint is slightly below the repeat control on all four
macro metrics. At the fixed final step, Fresh is higher in Accuracy and lower
in NLL/Brier/ECE, but both final checkpoints show severe calibration/loss
degradation compared with their selected checkpoints. This endpoint comparison
does not override the validation selector.

### Per-modality validation metrics

Each cell is `Accuracy / NLL / Brier / ECE`. Counts: Audio 11,166; Image 2,048;
Text 10,732; Video 400.

**Selected step 8,192**

| Modality | Repeat | Fresh |
|---|---|---|
| Audio | 0.6246 / 1.0706 / 0.4655 / 0.0769 | 0.5665 / 1.1114 / 0.5033 / 0.0759 |
| Image | 0.3540 / 1.7989 / 0.7353 / 0.0766 | 0.3608 / 1.7797 / 0.7282 / 0.0701 |
| Text | 0.2160 / 2.0022 / 0.8378 / 0.0802 | 0.2191 / 1.9667 / 0.8262 / 0.0799 |
| Video | 0.3375 / 1.5017 / 0.7556 / 0.1898 | 0.3475 / 1.6174 / 0.7945 / 0.2448 |

**Fixed endpoint step 32,768**

| Modality | Repeat | Fresh |
|---|---|---|
| Audio | 0.6882 / 1.3634 / 0.4402 / 0.1576 | 0.7297 / 0.9694 / 0.3479 / 0.0828 |
| Image | 0.5469 / 1.9488 / 0.6687 / 0.2286 | 0.6055 / 1.2536 / 0.5289 / 0.1319 |
| Text | 0.2419 / 3.2376 / 0.9549 / 0.2650 | 0.2521 / 2.3127 / 0.8783 / 0.1908 |
| Video | 0.3725 / 12.8189 / 1.1739 / 0.5727 | 0.3850 / 12.1739 / 1.1496 / 0.5551 |

Selected/final source-level `Accuracy`, `NLL`, `Brier` and `ECE`, including all
Text sub-sources, are preserved in `selected_metrics`/`final_metrics` in the
hash-pinned local run receipt. No source was omitted based on its result.

### CLEVRER question-type metrics

Each type has 200 validation questions. The new arm's Accuracy/NLL was:

| Step | Static descriptive Accuracy / NLL | Temporal descriptive Accuracy / NLL |
|---:|---:|---:|
| 512 | 0.330 / 1.397 | 0.365 / 1.331 |
| 1,024 | 0.325 / 1.360 | 0.305 / 1.375 |
| 2,048 | 0.300 / 1.451 | 0.360 / 1.458 |
| 4,096 | 0.350 / 1.442 | 0.345 / 1.452 |
| 8,192 | 0.375 / 1.534 | 0.320 / 1.701 |
| 12,288 | 0.390 / 2.065 | 0.350 / 2.620 |
| 16,384 | 0.390 / 3.391 | 0.375 / 4.268 |
| 24,576 | 0.355 / 8.275 | 0.380 / 9.285 |
| 32,768 | 0.395 / 11.159 | 0.375 / 13.189 |

At the endpoint, the repeat control was 0.410 / 12.404 on static descriptive
and 0.335 / 13.234 on temporal descriptive. Both arms became severely
overconfident on Video.

### Video scene-cluster uncertainty

Saved scores were paired by the exact validation row and resampled by the 200
CLEVRER video/scene groups (400 questions, two per scene). The analysis used
20,000 paired bootstrap resamples, seed 1701, percentile 95% intervals. Deltas
are first minus second. This is a development-validation diagnostic; repeated
validation and checkpoint selection mean it is not blind confirmatory evidence.

| Comparison | Video Accuracy delta (95% CI) | Video NLL delta (95% CI) |
|---|---|---|
| Fresh − Repeat, step 8,192 | +0.0100 (−0.0450, +0.0625) | +0.1157 (+0.0096, +0.2266) |
| Fresh − Repeat, step 32,768 | +0.0125 (−0.0450, +0.0700) | −0.6450 (−2.6022, +1.2835) |
| Fresh step 2,048 − Fresh step 512 | −0.0175 (−0.0625, +0.0275) | +0.0904 (+0.0381, +0.1454) |
| Fresh step 8,192 − Fresh step 512 | 0.0000 (−0.0500, +0.0500) | +0.2533 (+0.1596, +0.3536) |
| Fresh step 32,768 − Fresh step 512 | +0.0375 (−0.0225, +0.0975) | +10.8098 (+9.2957, +12.3994) |

The raw bootstrap output is saved beside the run as `video-paired-bootstrap.json`
(SHA-256 `1f75cf0224e157d03d76f09d5b4a0060067ee3330621375110a8ec9ec6ded5db`);
its exact resample seed and prediction source are recorded there. The video
Accuracy gains are uncertain, while the late-run NLL deterioration versus the
same run's 512-step checkpoint is clear in this resampling diagnostic.

## Interpretation and next gate

- **Learning-amount hypothesis: uncertain, not cleanly supported for Video.**
  Macro Accuracy rose through the long run, but the fixed selector chose 8,192
  and late validation macro NLL/ECE worsened sharply. Video Accuracy's
  scene-paired interval for 32,768 versus 512 includes zero, with a large NLL
  penalty. This does not prove the model family has reached a ceiling.
- **Coverage finding:** additional unique Text/Image/Audio data changed the
  final results, but did not improve the selected macro-NLL checkpoint over the
  matched repeat baseline. Video received no new unique questions. Its 512
  training questions reached 100% endpoint training Accuracy while held-out
  CLEVRER remained 38.5% Accuracy; this points to video data/task coverage and
  memorization as a likely bottleneck for this run, not evidence that more
  repeats are useful.
- **Calibration/overfit:** training macro NLL kept falling while validation
  macro NLL and ECE rose after the selected checkpoint. The strongest late
  warning is Video, whose endpoint NLL exceeded 11 on both descriptive types.
- **Uncertainty:** one seed; validation influenced checkpoint selection; only
  200 independent video scenes; two CLEVRER question types; no explanatory,
  predictive or counterfactual training examples. The intervals diagnose
  paired variation but do not remove these limits.
- **Next:** preserve this completed arm and the repeat baseline. Do not extend
  this same run, add seeds, tune on sealed audit or promote either endpoint.
  If video work resumes, first admit more unique, scene-disjoint, authorized
  Video supervision with distinct question types, then preregister a new
  matched run and selector. Keep the four-modality 95–98% objective and
  200M pre-quantization cap unchanged.

## Orbitune handoff and repository verification

Before using the RTX 3080, the active Orbitune run was backed up to
`C:\CodexArtifacts\pretrained-reuse-r2\orbitune-protected-before-hotgpu-20261021\model.pt`
(SHA-256 `9f5e1ced520bac3ca0e24c8962223160603d0a0de75976fec953a3c67a1d84c9`).
The cooperative yield completed at step 343,623 with `reason=idle_yield` and a
final checkpoint write. The post-yield `model.pt` was 117,880,492 bytes with
SHA-256 `baa3991c72412897f978334cb7d81b783adb1fc4f2b282624fdaac61a9e31a52`.
After the HotUGPU reservation was released, the existing idle supervisor
automatically resumed Orbitune. At the post-run read-only check, PID 17812 was
active at step 343,750; its live checkpoint was not rehashed to avoid racing
its writes. No process was force-killed. The research commits did not alter the
Orbitune source or run.

Ruff passed; CPU tests passed (2,319 passed, 11 skipped); the long-run smoke,
full run and replay passed. CI for source commit `cb68d786e15d94152960e20db7081de11b94fa24`
passed as run 39539292594. This report is research documentation only.
