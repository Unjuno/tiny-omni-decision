# Conditional audio comparison completed: mixed, not adopted

Full fixed development corpus was acquired and tested before learning. Existing
Clotho/FSD/STARSS/model artifacts are retained. No sealed audit or official AQA
validation/test questions were opened. These are development results only.

## Executed integrity and data coverage

- 856 licensed audio sources,564 uploaders; train578/387, calibration138/96,
  validation140/81 sources/uploaders. Questions2312/550/560 respectively.
- Attempt01 ended onHTTP504 with205 files. Attempt02 reused216 already verified
  files, transferred1,026,806,109bytes and finishedPASS in3721.37s with24 transient
  HTTP errors handled. Attempt01 remains unchanged.
- All856 files rehashed and decoded to finite/nonzero audio; no exact decoded
  duplicates. Total19271.256seconds of audio; decode audit17.04s CPU.
- All180,004 cross-split whole-template pairs screened at2kHz: zero .99/undefined
  flags, maximum .544492. Runtime130.71s, prepared cache2,411,883,392bytes.
  This does not guarantee absence of edits, mixtures or shorter reused excerpts.
- Frozen EfficientAT strict-load/hash/source checks passed; all856x384 features
  finite and bitwise-repeatable.143.95s CPU excluding model load. FeatureSHA
  `fe0b7adafb4841a107015a9f7ac014490bccec0f61d23b49f56fc9d6e18fc0b4`.
- Frozen manifests, options, majority labels and all annotation disagreements
  preserved. Pretraining contamination and global other-corpus reuse remain
  unverified; this is not a new blind benchmark.

## Fixed learning and calibration

All primary conditions use the same49,409-parameter770→64→1 candidate scorer,
seed17, AdamW .001/weight decay .001, batch128,24epochs, same train query order.
Select earliest minimum raw validation NLL, then use calibration-only17-point
temperature grid. Each consumes55,488 presentations of2,312 unique questions
from578 unique underlying train sources. No architecture, ratio, seed or budget
search followed the results. Best/final24 weights are separate.

| Selected condition | Best epoch | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train-frequency prior | — | .566071 | .684406 | .491284 | .002700 |
| Question only | 8 | .648214 | .623079 | .433142 | .028331 |
| Audio only, criterion removed | 3 | .566071 | .686626 | .493431 | .026661 |
| Audio + question | 7 | .669643 | .624661 | .434001 | .025451 |
| Joint weights, source-deranged audio | fixed7 | .623214 | .651505 | .459546 | .036699 |
| Joint weights, audio set to train mean | fixed7 | .614286 | .642392 | .451031 | .049634 |

The audio-only arm hides the question in a task whose answers depend on that
question. Its56.61% is a control, not an overall environmental-audio recognition
score. Source-deranged diagnostics retain original paired targets; they are
sensitivity to incorrect evidence, not labels for the replacement sound.

## Interpretation and uncertainty

Joint exceeds question-only Accuracy by2.14points, but NLL/Brier are slightly
worse. Paired uploader bootstrap (81groups,10000 draws,seed17, percentile95%):
Accuracy difference[-.51,+5.00]points; NLL difference[-.010595,+.013268]. Thus
an improvement over question-only is not established. This model uses audio:
aligned-minus-deranged Accuracy4.64points, interval[1.61,7.64]; NLL difference
-.026844, interval[-.041954,-.012988]. That diagnostic does not prove robust
reasoning or justify product adoption. The same validation selected checkpoints;
intervals are descriptive and not selection-adjusted or blind.

All3 selected models and both joint diagnostics independently reloaded and
reproduced all560 probabilities bitwise. All source/question IDs matched.

## Learning curves and fixed-example gaps

All72 condition/epoch points are in the adjacent CSV and original full histories.
Rolling training CE is not a fixed-example gap. Separately replayed the complete
fixed train/validation examples, without calibration or weight changes:

| Condition/checkpoint | Train Accuracy | Val Accuracy | Train NLL | Val NLL |
| --- | ---: | ---: | ---: | ---: |
| Question best8 | .733997 | .648214 | .542790 | .627310 |
| Question final24 | .826125 | .655357 | .414296 | .670354 |
| Audio best3 | .570502 | .566071 | .677538 | .685995 |
| Audio final24 | .616349 | .558929 | .641252 | .702727 |
| Joint best7 | .774221 | .669643 | .501496 | .632454 |
| Joint final24 | .936419 | .655357 | .221414 | .770890 |

Joint fixed Accuracy gap grows10.46→28.11points while train NLL falls and
validation NLL rises. Additional epochs at this setup are not justified.
Possible contributors include limited unique conditional data,41% validation
annotation disagreement, source shift, pooled representation and audio/text
alignment. Their relative causes and any model-family ceiling remain unknown.
No capacity increase or new learning run follows automatically.

## Reproduction, artifacts and limits

Learning sourcec3d2d2840fe4713bc6c136f40a34ebd4dcfd55ed;
`python -m scripts.train_clotho_audio_comparison`. Question-only source/history
retained separately. Exact script/helper hashes, model revisions, corpus/feature
hashes and protocol are saved. The adjacent JSON includes all selected/final
weight, normalization, protocol, probability and ID hashes plus independent audits.

Python3.11.9, Torch2.6.0+cu124, NumPy2.4.6, Transformers5.6.2, PyAV18.1.0,
torchaudio2.6.0+cu124; reused process-local dependencies, no environment upgrade.
Audio/joint head learning+selection+calibration .701/.657s CPU; joint diagnostic
.0047s. No GPU learning or paid/cloud compute. Peak process RAM was not measured;
the prepared-cache byte count is not a peak-RAM claim. Joint-only inference model
inventory19,400,472 parameters (audio1,961,239 + text17,389,824 + head49,409),
not an all-modality release inventory or proof of the95–98% goal.

Artifacts under `C:/CodexArtifacts/pretrained-reuse-r2/`:

- `clotho-aqa-development-media-02/`, `clotho-aqa-decoded-audit-01/`
- `clotho-aqa-cross-split-waveform-screen-01/`, `clotho-aqa-audio-features-01/`
- `clotho-aqa-question-control-01/`, `clotho-aqa-audio-control-01/`,
  `clotho-aqa-joint-control-01/`: best/final, protocols, histories, probabilities
- `clotho-aqa-comparison-audit-01/`, `clotho-aqa-fixed-gap-01/`: independent checks
- `clotho-comparison-parity-01/`, `clotho-control-implementation-parity-01/`:
  zero-audio software replays, not extra quality candidates or seed searches

Exact executed audit scripts remain in the parent artifact directory. Keep the
joint model as a reproducible experiment; do not promote, compress or call it
a completed Omni Teacher. The full project remains open. Next improvement must
address generalization using a controlled representation/alignment or data test,
not extend epochs or redefine the benchmark around unanimous/easy questions.

Verification: repository-wide Ruff passed;793 CPU tests passed,1skipped; all25
model/dataset/catalog manifest audit commands passed. The scorer class was
confirmed to load from this worktree's `src/tiny_omni_decision/decision.py`.
