# Perception Test source screen — 2026-10-21

## Decision

**Admit Perception Test only for a bounded exploratory development probe. Do not
admit it as independent product validation or as proof of general audiovisual
fusion.** The official release provides useful video questions and time-localized
sound segments with a visibility label, licensed annotations/media, and separate
train/validation ID lists. Its task and split metadata merit a small feature/data
pipeline trial. Before any quality-selected fit, the media timing and per-source
integrity must pass on a preregistered sample, and controls must compare audio-only,
video-only, and joint input on identical video-grouped events.

The test split was not downloaded or opened. Two frozen encoders were later run
on the four integrity-smoke pairs using CPU only; this was a feature-path smoke,
not model fitting and not a quality evaluation. No metric was generated.

## Why this source is relevant

The official annotations expose a compact binary audiovisual decision: for a
time-localized sound segment, determine whether its source is visible. The released
schema gives the segment timestamps/frame IDs and `is_visible`; the audio/video
bytes remain the actual model inputs. The released label/parent-object metadata
must not be passed to the scorer. This is more directly relevant to complementary
audio/video use than generic AV co-occurrence.

The official multiple-choice video QA task is a separate three-option text/video
candidate: it includes descriptive, explanatory, predictive, and counterfactual
reasoning types. Its media includes audio, but that alone does not establish that
each answer requires audio. Keep the QA task separate from the visibility task and
do not count either as standalone text or audio competence.

## Pinned source and integrity screen

- Official repository revision: `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`.
- The repository declares software Apache-2.0 and all other released materials
  CC-BY-4.0. This source disposition is scoped to exploratory research with
  attribution; it is not a blanket redistribution decision for a final model.
- Official `mc_question` annotations: 1,955 train videos / 7,392 questions and
  5,260 validation videos / 19,140 questions. Every question has exactly three
  choices and a valid target. The official ID lists exactly match annotation
  video IDs; train/validation ID overlap is zero.
- Official `sound_localisation` annotations: 2,149 train videos / 25,757 segments
  and 5,359 validation videos / 35,625 segments. The official ID lists exactly
  match annotation video IDs; train/validation ID overlap is zero. The sound and
  MCQA video groups overlap within each split, as expected, but no train ID
  appears in validation.
- Every inspected sound segment has a two-point increasing timestamp and binary
  `is_visible`. Metadata/video/audio archive IDs agree for all requested
  annotation IDs. Remote archive directories were inspected with bounded HTTPS
  byte ranges, pinned object size and ETag, and exact `Content-Range` checks; no
  full archive was downloaded.
- A deterministic timestamp-boundary audit found 247/25,757 train segments and
  133/35,625 validation segments whose annotation end exceeds the audio duration;
  all frame ranges remained in bounds. These events must be excluded by this
  media-availability rule, not selected or removed based on model outcomes.
- Requiring a 250ms margin before and after the annotated event, and checking
  both audio and video duration, leaves 4,159 not-visible / 18,530 visible train
  events across 1,407 / 2,045 distinct videos, and 1,927 / 31,909 validation
  events across 1,023 / 5,184 distinct videos. A group-disjoint balanced cohort
  can therefore contain at most 1,023 validation videos per class. This filter
  does not resolve annotation quality or media near-duplicates.
- A fixed seed-17 exploratory cohort now contains 128 events per class for each
  of train and validation, with exactly one event per video and zero train/valid
  video overlap. The immutable selection manifest is stored under
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-test-mcqa-screen-20261021`;
  SHA-256 is `dd8c9de6d8970eb1105cb81551e386fca563eb174b8bcda5b09dcc8a16e567be`.
  Pinned ZIP central directories predicted 10,026,549,995 compressed bytes and
  10,642,102,013 expanded bytes for the selected 512 video/audio pairs. The
  separate acquisition completed: all 1,024 selected files matched expected
  lengths and ZIP CRCs, actual uncompressed bytes exactly matched the estimate,
  and train/validation raw video and audio SHA-256 overlap were both zero. The
  13,901,192,764 range bytes were transferred in 2,204.3 seconds. Receipt:
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-v1\media\acquisition_manifest.json`,
  SHA-256 `f29bcfee3465164bf2165240ad2b76821a86d043656b2b7b76a71ee8dadc7641`.
  The 512 event crops have not yet been fully decoded/encoded or evaluated.
- Annotation/archive directory hashes and selected-media hashes are recorded in
  `manifests/pretrained_reuse/perception-test-sound-visibility-20261021.json` and
  the local artifact receipt under
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-test-mcqa-screen-20261021`.
  Only four selected audio/video pairs (two train and two validation videos) were
  extracted for decoder/sync integrity. They are not a quality sample. No test
  media or labels were read.

Both frozen encoders subsequently produced finite embeddings over the same
centered audio/video interval: EfficientAT emitted 384 values and VideoPrism
emitted 768 values, with eight video frames sampled across the interval. The
joint backbone count is 116,326,423 pre-quantization parameters before any
readout. The pinned unquantized VideoPrism checkpoint is 457,475,296 bytes;
the raw 4-bit payload for both encoders would be about 58.2MB before quantization
metadata and any Decision readout. The complete <=200MB package target is
therefore still unverified, not disproven. These four embeddings are not a
quality sample.

A separate one-example EfficientAT CUDA smoke was then started through the
HotUGPU guard. The guard waited while the Orbitune compute PID remained active,
reserved 2,471MiB only after cooperative yield, and released the reservation
after the child completed. The RTX 3080 Laptop forward took 0.156s with
44.6MB peak allocated; this is a runtime check, not training or a quality result.

The complete fixed cohort's EfficientAT embeddings then passed on all 512
examples (256 train / 256 validation), using the HotUGPU guard for the CUDA
child. Mean encoder forward time was 0.082s/example; peak allocated VRAM was
62.2MB. EfficientAT state hashes before and after were identical. Raw audio,
resampled window PCM, and embedding hashes had zero train/validation overlap.
The format audit found 118 44.1kHz mono, 1 44.1kHz stereo, 63 48kHz mono, and
330 48kHz stereo 32-bit PCM files; the maximum absolute metadata-duration
difference was 35 microseconds, and none of the fixed windows exceeded the
actual audio duration. The first broad extraction attempt exposed an overly
narrow 48kHz-stereo-only decoder assumption after saving 63 valid rows; it
stopped without training or changing model weights. The decoder was expanded to
the four observed PCM layouts, and those rows were hash-verified before resuming.
The final run is recorded at
`C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-v1\features_efficientat\result.json`.

## Important data limitations

Visibility is highly imbalanced and shifts between the official partitions:

| Split | Not visible | Visible | Majority-class accuracy |
| --- | ---: | ---: | ---: |
| Train | 6,587 (25.6%) | 19,170 (74.4%) | 74.4% |
| Validation | 3,090 (8.7%) | 32,535 (91.3%) | 91.3% |

Consequently, raw validation accuracy is insufficient. Any later fit must report
per-class recall, balanced accuracy, NLL, Brier, ECE, and video-grouped uncertainty,
plus the train-only majority baseline. It must not tune a threshold or sample the
validation set to obtain a target score.

The sound-label vocabulary also differs (17 labels in train, 10 in validation),
which is another reason not to use the released sound-class text as an input or
claim label-shift robustness from this source. The proposed visibility task should
use only the synchronized raw audio/video window and a fixed yes/no question;
the upstream event name, parent object IDs, and `is_visible` field are label-side
metadata only. Segments from the same video must stay in one split. A predeclared
evaluation cohort should cap repeated events per video so a few videos cannot
dominate uncertainty estimates.

ID disjointness does not establish raw-media or normalized-content disjointness.
The four-file smoke verifies archive extraction and timestamps only; it does not
establish that the annotation boundary matches the audible/visible event on every
sample. Before training, inspect synchronized decoded windows from a frozen,
video-grouped sample and audit near-duplicate media hashes. If that gate fails,
stop this candidate without fitting.

## Completed input/readout gates and remaining limits

1. Fixed a deterministic, balanced train/validation cohort with one event per
   video and 250ms media context around the annotated interval.
2. Acquired all selected raw media; archive integrity, exact media identities,
   synchronized crops, and train/validation media/feature disjointness passed.
3. Extracted all 512 examples using frozen EfficientAT and VideoPrism.
4. Fit the single fixed audio-only, video-only, and joint readouts on train only;
   evaluated the same 256 validation video/event IDs once and stored per-example
   JSON probabilities plus paired bootstrap comparisons.
5. Kept official test and sealed audit untouched. This narrow development result
   is not a blind estimate and is not a product-quality gate.

The main unresolved issue is generalization: every tiny readout fit achieved
100% train accuracy, while validation balanced accuracy was only 56–65%. The
joint readout shows some development evidence that both streams affect outputs,
but accuracy, calibration, and shortcut robustness remain inadequate. The next
work should diagnose task/data semantics and obtain independent task coverage;
do not expand this fitted readout or tune it against these validation IDs.

## Source links

- [Official Perception Test repository at the pinned revision](https://github.com/google-deepmind/perception_test/tree/3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf)
- [Official dataset README and license statement](https://github.com/google-deepmind/perception_test/blob/3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf/README.md)
- [Official release downloads](https://github.com/google-deepmind/perception_test/blob/3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf/README.md#download-the-data-and-annotations)


## Fixed matched readout result (2026-10-21)

A single preregistered CPU readout fit used the 256 train examples and evaluated
all 256 fixed validation examples (128/class). Each readout is one affine binary
logit head, train-only feature standardization (std floor 1e-6), zero initialization,
full-batch LBFGS, fixed 250 maximum iterations, L2=0.01. No validation checkpoint
selection, threshold adjustment, calibration, seed search, or sealed/test evaluation
was performed. The validation cohort is deliberately class-balanced; prior-weighted
values below are importance-weighted estimates from this small class-conditional
cohort, not a representative validation sample.

| Frozen features + readout | Head params | Train Acc / NLL | Validation Acc / balanced Acc | Recall not-visible / visible | Val NLL | Val Brier | Val ECE | Mean confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientAT audio | 385 | 100.0% / 0.115 | 56.25% / 56.25% | 60.16% / 52.34% | 1.199 | 0.325 | 0.274 | 0.837 |
| VideoPrism video | 769 | 100.0% / 0.035 | 57.81% / 57.81% | 49.22% / 66.41% | 1.404 | 0.337 | 0.293 | 0.863 |
| Joint audio+video | 1,153 | 100.0% / 0.018 | 64.84% / 64.84% | 63.28% / 66.41% | 1.088 | 0.284 | 0.256 | 0.879 |

The 100% train accuracy versus 56–65% validation accuracy is a pronounced
small-sample overfit warning. With importance weighting to the official
validation visibility prior (91.33% visible), joint estimated Accuracy is 66.14%,
NLL 1.068, Brier 0.274, ECE 0.246; these estimates are not observed full-split
metrics. Joint-minus-audio paired bootstrap on 256 unique validation videos
(2,000 resamples, seed 17) gives +8.5pp Accuracy (95% percentile interval
+1.6 to +16.0pp); NLL delta -0.107 (interval -0.339 to +0.124) and Brier delta
-0.040 (interval -0.091 to +0.010), so calibration/loss evidence is inconclusive.
Joint-minus-video is +7.0pp Accuracy (interval +1.2 to +12.9pp), NLL delta
-0.317 (interval -0.507 to -0.128), Brier delta -0.054 (interval -0.094 to
-0.013). These are development-only paired diagnostics after a readout was fit
on train, not blind inference.

Input interventions changed predictions: replacing audio with its train mean
flipped 19.9% of joint predictions; replacing video flipped 32.4%; donor-swapping
audio or video between validation videos flipped 27.3% / 43.8%. The joint model
therefore reacts to both feature streams, but these interventions do not prove
semantic complementarity or rule out shortcuts. The task is one narrow binary
source-visibility decision and cannot support general AV, text, image, sensor,
JSON-schema, quantized-package, or product-quality claims. Accuracy is far below
the 95–98% aspiration. Do not expand this readout or promote it as a product
model. Keep the feature and prediction receipts for audit; no GPU training was
run.

The result JSON's `train_final_ce` field contains PyTorch LBFGS's returned initial
zero-logit closure value, not final fitted train CE. Final fitted train metrics
are computed separately from predictions and are reported above; this logging
limitation is preserved explicitly in the manifest.
