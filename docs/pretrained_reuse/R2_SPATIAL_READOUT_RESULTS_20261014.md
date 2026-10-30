# R2: time-varying spatial appearance comparison — executed

## Outcome

Both fixed50-epoch arms completed650 updates. Execution PASS; **no observed
original-QA Accuracy gain, no useful conditional/temporal reasoning established,
no adoption or budget extension**. All99train/257observed-development videos and
198/514 original questions remain. Best/final outputs match their original epochs
bitwise, including a separate fresh process. The full Omni95–98% goal is unmet.

[Preregistered protocol](R2_SPATIAL_READOUT_PROTOCOL_20261014.md),
[complete spatial cache and replay](R2_SPATIAL_CONTEXT_RESULTS_20261014.md),
[audited results/hashes](R2_SPATIAL_READOUT_RESULTS_20261014.json),
[all100 learning-curve rows](R2_SPATIAL_READOUT_CURVES_20261014.csv).

## Controlled design

Same232,385-parameter readout, initial weights and seed17 sampler, batch8 videos,
AdamW0.001/weight_decay0.01,50epochs, auxiliary role weight1. Frozen TAPIR motion/
initial-point features and MiniLM supplied-option features are unchanged. Spatial
context32×256×387 is projected to64dimensions and read by16 latent queries;
point/candidate attention then returns logits over the original three options.
Both questions reuse a single encoded state. Only appearance values differ:
`initial_broadcast` repeats frame0; `temporal_appearance` uses actual32 snapshots.
Identical source-time/x/y tags and original point motion remain in both arms.
No boxes/role IDs/answers enter forward; source role labels supervise loss only.
Auxiliary coverage98train/253dev does not reduce original QA coverage99/257.

Selection is earliest minimum original-question dev NLL from stable log-softmax.
All attempted epochs remain. This is already observed development used for
checkpoint selection, not a blind audit or proven participant-disjoint sample.
No option-count reduction, repeated seed search, data replacement or encoder fit.
The previous139,521-parameter role-supervised readout reached40.27% on these IDs;
that contextual comparison changes architecture and cannot isolate one factor.

## Selected original QA — all514 questions

| Arm | Best epoch/update | Accuracy | NLL | Brier | ECE15 | Both questions correct |
|---|---|---:|---:|---:|---:|---:|
| initial_broadcast | 6/78 | .389105 | 1.081485 | .655647 | .010603 | .206226 |
| temporal_appearance | 6/78 | .389105 | 1.081710 | .655848 | .011678 | .206226 |

Both predict the second canonical option on every question: argmax counts
`[0,514,0]`, no initial/final prediction differences, **0/127 changed-target pairs
both correct**. Low ECE for nearly constant class-frequency predictions is not
evidence of high-quality decisions. Initial/final Accuracy is40.08%/37.74% in both.

Actual-time minus broadcast,10,000 paired video-ID resamples, seed17,
percentile95% intervals; retain the two questions from each video together:

| Metric | Difference | 95% interval |
|---|---:|---|
| Accuracy | 0 | [0,0] |
| NLL | +.000226 | [-.000139,+.000588] |
| Brier | +.000201 | [-.000044,+.000443] |

Accuracy interval degenerates because both selected classifiers have identical
argmax predictions. It does not prove the absence of possible future improvements.
NLL/Brier intervals include zero. No ECE interval was computed; selection bias and
participant/script dependence remain. This is not an independent final-test claim.

## Fixed controls

All controls preserve every original question/option/target; none selects weights.
Zero removes point/context state; mismatch cyclically substitutes only another
video's spatial context, leaving point motion unchanged. Initial broadcasts frame0;
reverse reorders appearance while position tags stay fixed.

| Arm/control | Accuracy | NLL | Brier | ECE15 |
|---|---:|---:|---:|---:|
| broadcast/normal | .389105 | 1.081485 | .655647 | .010603 |
| broadcast/zero | .389105 | 1.082484 | .656509 | .014623 |
| broadcast/mismatch | .389105 | 1.081904 | .655943 | .014454 |
| actual/normal | .389105 | 1.081710 | .655848 | .011678 |
| actual/zero | .389105 | 1.083440 | .657220 | .018210 |
| actual/mismatch | .389105 | 1.081779 | .655902 | .012027 |
| actual/initial | .389105 | 1.081607 | .655790 | .012072 |
| actual/reverse | .389105 | 1.081710 | .655848 | .011678 |

All controls change0/514 argmax predictions. Actual-time reversal alters NLL by
only6.61e-9. Probabilities do depend weakly on state, but useful scene-specific or
temporal decision contribution is not demonstrated. Supplemental cover top1
point-in-bbox is58/253(22.92%) vs57/253(22.53%). This is partial auxiliary
localization; grid background Accuracy does not meet the product quality target.

## Learning curves and gap

| Arm/epoch | Train Accuracy | Dev Accuracy | Train NLL | Dev NLL |
|---|---:|---:|---:|---:|
| broadcast/1 | .449495 | .389105 | 1.076312 | 1.092864 |
| broadcast/6 best | .449495 | .389105 | 1.062120 | 1.081485 |
| broadcast/10 | .449495 | .389105 | 1.046742 | 1.086363 |
| broadcast/20 | .742424 | .367704 | .467764 | 1.615538 |
| broadcast/30 | .792929 | .324903 | .418343 | 1.697417 |
| broadcast/40 | .878788 | .367704 | .276351 | 2.124296 |
| broadcast/50 final | .974747 | .334630 | .082188 | 2.844171 |
| actual/1 | .449495 | .389105 | 1.076199 | 1.092985 |
| actual/6 best | .449495 | .389105 | 1.062387 | 1.081710 |
| actual/10 | .449495 | .389105 | 1.048849 | 1.085689 |
| actual/20 | .737374 | .361868 | .507685 | 1.664668 |
| actual/30 | .797980 | .330739 | .388313 | 1.963644 |
| actual/40 | .904040 | .311284 | .205077 | 2.858799 |
| actual/50 final | .959596 | .357977 | .139763 | 2.676397 |

Final dev ECE .483556/.439647 and Brier1.058060/1.024174 degrade sharply.
Training exceeds95% while development remains poor: clear overfitting in both
bounded runs. These are fixed-corpus metrics, not rolling minibatch gaps.
Do not extend this same readout/data repetition to force an aspirational score.

## Actual compute and accounting

- Source commit `fa360f1aae214ff87bf567eb3fe71aa6bf778b9c`.
- Command: `python -m scripts.train_spatial_trajectory_readout --output C:/CodexArtifacts/pretrained-reuse-r2/spatial-readout-full-01`.
- UTC start/end `2026-10-14T01:35:03.841431+00:00` / `2026-10-14T01:44:40.264605+00:00`.
- Full body576.415863seconds. Arm bodies199.539339/346.083892seconds;
  synchronized train windows50.560907/76.861617seconds and evaluation windows
  131.741932/242.022460seconds. These include input I/O/CPU work, not GPU busy time.
  Part of execution overlaps lightweight CPU tests/review/publication preparation;
  timing is not an uncontended architecture benchmark.
- RTX3080Laptop16GB, Python3.11.9, Torch2.6.0+cu124/CUDA12.4, FP32, TF32off,
  deterministic algorithms, CPU4threads. Existing packages unchanged. No encoders
  load for fitting; only immutable feature caches and this readout.
  Environment also has NumPy2.4.6, PyAV18.1.0, Transformers5.6.2 and PEFT0.21.2;
  these versions were queried after execution, without an upgrade.
- Peak CUDA allocation396,385,280bytes; reservation497,025,024bytes;
  Windows peak process working set2,182,733,824bytes. Disk remained above8GiB.
  Local only, cloud cost0. No new full-base/encoder/projector optimization.
- Each arm consumed4,950video/9,900question presentations across99unique videos/
  198unique questions (4,851/9,702 repeated presentations). Selected6epoch point
  consumed594video/1,188question presentations; do not mix it with full-run counts.
- Each readout has232,385 registered/trainable parameters and939,250weight-file
  bytes. Whole frozen Video+Text+readout component:48,694,536 parameters, counting
  full31,072,327TAPIR and17,389,824MiniLM. Raw pinned encoder+readout weight files
  total194,857,212bytes. This is not a built200MB package or complete four-modality
  resident model; tokenizer/runtime assets and other modalities remain unmeasured here.

Train-only smoke `spatial-readout-smoke-01` completed two updates per arm in a
separate process,37.937889seconds body,462,603,264byte peak allocation. Smoke
includes startup/concurrent CPU test contention; no validation quality selection.

## Audit, protection and artifacts

All100 epochs, nested initial/final/changed metrics and auxiliary membership
metrics were recomputed from saved original predictions;116 arm artifact hashes,
exact original labels/ordered IDs, sampling, selected controls and earliest-minimum
selector were checked. Missing-label masks remain98/253 and all original QA rows.
The first publication byte check rejected Windows CRLF versus Git LF differences.
Numeric content was identical; original publication bytes were archived separately,
and publication JSON now uses explicit UTF-8 LF with a regression test. Training,
config and checkpoint bytes were not changed by this publication correction.
Separate fresh-process replay passed allfour best/final checkpoints on full198train/
514dev predictions, cover probabilities and log-softmax outputs bitwise. It took
46.483787seconds and overlapped CPU publication audit; no training occurred.

- Full run: `C:/CodexArtifacts/pretrained-reuse-r2/spatial-readout-full-01/`.
- Effective config SHA: `ad98fbc9f31866f16d2268a46c160dd43951a3217bbfb49502dda7f9e6ecb954`.
- Full result SHA: `cdb3b521519af36d3394fcd6e5a620f1dd0312da193223d71371075704ab019b`.
- Curves SHA: `d538860e8a6fd81d7106a0d6c0346b87922c4d5d97df0d4367d9e59f1466d385`.
- Fresh replay: `C:/CodexArtifacts/pretrained-reuse-r2/spatial-readout-fresh-reload-01.json`,
  SHA `a28898f20398ed7b3b1a7de64d9644abc722faa82c100ceae5560a71618993ca`.
- Best weights SHA broadcast: `3eeaba1895223e82e8bf33481f98b0e49373461bd2fe68b9e7b819f570865509`;
  actual: `071445b8f1c0ba892ef83f2d2178e6f924e0344e330f0cfff6296d4105e78bad`.
- Best/final optimizer/sampler/Torch/CUDA RNG states are saved, but a strict
  uninterrupted-equivalent resume has not been implemented/tested or claimed.

Code review corrected missing paired intervals, incomplete final resource limits,
missing process-memory accounting and publication label/nested-metric gates before
their dependent full execution/publication. Regression tests showed those failures
then passed. All smoke/extraction/comparison attempts remain in independent outputs;
no failed learning attempt was hidden. No legacy teacher/data/config/artifact or
active unrelated process was changed. No sealed audit, quantization, cloud,
product promotion or main merge. Fresh CPU verification:1044passed/1skipped in
15.71seconds; Ruff passes, all manifests/four catalogs/policy audit pass.
Implementation-head Actions [38496555042](https://github.com/Unjuno/tiny-omni-decision/actions/runs/38496555042)
and [38496551653](https://github.com/Unjuno/tiny-omni-decision/actions/runs/38496551653)
passed on `fa360f1`. Final publication-head CI is checked after push and recorded
in the local publication status and user report. No GPU/model tests enter normal CI.

## Next direction

Close the global spatial-context pooling/readout comparison. More snapshots were
available, yet selected decisions stayed at a canonical-option prior and ignored
the condition change. Data scarcity, low-level TAPIR appearance, missing explicit
object/cover association and condition/readout representation remain plausible
bottlenecks; this study does not uniquely identify one or prove a model ceiling.
Inspect condition/state binding on training-only examples before another fit,
and continue public licensed task selection for genuinely complementary native
inputs. R2 quality, conditional/OOD/joint benefit and eventual package gates stay open.
