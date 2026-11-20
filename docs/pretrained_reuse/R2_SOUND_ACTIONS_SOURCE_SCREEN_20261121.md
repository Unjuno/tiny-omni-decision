# SoundActions v2 source and task screen — 2026-11-21

## Decision

`365 Sound Actions` v2 is admitted for a bounded local development experiment.
Its official Zenodo release provides an explicit CC BY 4.0 grant, a small
aligned audio/video corpus, and human consensus labels for a narrow temporal
sound-shape Decision. This is a task-specific R2 candidate, not a general
audio/video benchmark or product-quality claim.

The dataset is small, has one recordist, and reports only moderate inter-rater
agreement. It cannot measure new-person generalization or establish the
four-modality accuracy target. The sealed partition remains unavailable to
normal feature extraction, fitting, validation, and model selection.

## Pinned public source

- Zenodo record: [365 Sound Actions v2](https://zenodo.org/records/23971067),
  DOI `10.5281/zenodo.23971067`, published 2026-11-05.
- The exact `LICENSE.txt` grants CC BY 4.0, including commercial reuse with
  attribution, a license link, and notice of changes. Its downloaded size and
  publisher MD5 match the version-2 API listing.
- The official README states 365 mono 48 kHz WAV files, 365 paired H.264 MP4s,
  and 365 stills; video retains roughly 60 fps in v2. The audio and video
  archives are about 326 MB and 1.9 GB, respectively. No media was downloaded
  during this metadata screen.
- Each recording was made by one recordist across 2022. Three anonymous raters
  independently supplied labels; no annotator personal information was
  collected. The release does not establish multi-person recording
  generalization. Some activities can comprise several operations rather than
  one atomic action.
- The current v2 summary reports `impulsive=126`, `iterative=125`,
  `sustained=83`, and `unknown=31`. `Unknown` means no majority; the reported
  free-marginal multirater kappa is `0.46`. Only majority-labeled rows will be
  supervised/scored. No label will be silently corrected or discarded based on
  model predictions.
- The source paper already reports five-fold experiments on these labels. In
  its reported all-adapter condition, audio-only validation reached about
  `0.66` accuracy, combined audio/video about `0.60`, and video-only about
  `0.48`; the paper describes overfitting and modality imbalance. Our study
  therefore tests a compact frozen-encoder decision path with stricter
  sample separation, and does not assume paired input will win. See the
  [open-access paper](https://doi.org/10.5334/tismir.223).

## Frozen task and split

Question: `Which description best fits the sound's temporal shape?`

Supplied choices: `impulsive`, `sustained`, `iterative`. Probabilities are
returned against stable option IDs; the majority `perception_type` is the
target. Media only is exposed to the candidate model. Title, action, object,
material, environment, source URL and label metadata are excluded from model
features.

The split was frozen before downloading media or parsing additional per-record
labels. IDs 1–35 are forced to train because the official overview preview
exposed those rows before the split freeze. Remaining IDs are assigned by the
seed-17 SHA-256 rule in
[`365-sound-actions-split-v1-20261121.json`](../../manifests/pretrained_reuse/365-sound-actions-split-v1-20261121.json):
266 train, 53 validation, and 46 sealed-audit recording IDs. Audio, video and
still assets stay together by recording ID. The publisher checksum inventory
lists 1,095 media files for the 365 IDs and has zero exact per-modality byte
hash collisions between recording IDs. This checks exact byte identity only;
it does not prove no semantically similar actions or near-duplicate scenes.

The one-recordist design means this split supports held-out-recording evidence
only. It cannot support a person-independent claim. Existing published
cross-validation results are prior exposure of this task in the field; our
held-out partition is a one-time project evaluation, not a blind benchmark.

## Planned comparison

Use the pinned, frozen `EfficientAT DyMN-0.4` audio encoder and the already
audited `VideoPrism-B` video encoder, then fit an equal-budget compact
candidate-conditioned readout. Compare audio-only, video-only, and paired
audio/video conditions on identical train/validation recording IDs. Keep
validation for the predeclared checkpoint rule. After architecture, epoch
rule, and controls are frozen, evaluate the sealed recordings once.

Required controls include option permutation, audio/video identity mismatch,
missing-audio and missing-video inference, and majority-label coverage. Report
Accuracy, NLL, Brier, ECE, classwise counts, and paired recording-level
uncertainty. A paired gain must exceed both unimodal controls on the same
recordings before claiming useful fusion. The dataset's five-option-free
three-class task is not evidence for unrestricted user-supplied options.

The current frozen pretraining overlap with these source recordings is
unknown. The model comparisons will be described as transfer results under
that caveat.

## Acquisition record

Metadata-only artifacts are in
`C:\CodexArtifacts\pretrained-reuse-r2\soundactions-v2-20261121-attempt-01`.
The official record JSON, README, overview, label archive, statistics,
license, citation file and publisher SHA-256 list were downloaded and checked
against the exact record's API size and MD5 fields. The only opened per-record
overview rows were 1–35, and they are confined to train. No model, media,
validation metric, sealed-audit label, or sealed-audit signal has been read.

No media extraction or training has started yet. Before that, verify the local
media extraction plan, feature-cache memory, and end-to-end frozen feature
smoke in a separate process; keep outputs outside Git. A pre-quantization
VideoPrism+EfficientAT pair is expected to exceed the provisional 200 MB
package target, so this experiment can establish task quality only, not
deployment-size compliance.

## Primary sources

- [Pinned Zenodo v2 record](https://zenodo.org/records/23971067)
- [CC BY 4.0 license text](https://creativecommons.org/licenses/by/4.0/legalcode)
- [SoundActions study](https://doi.org/10.5334/tismir.223)
