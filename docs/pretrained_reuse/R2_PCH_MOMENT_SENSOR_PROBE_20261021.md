# R2 PCH wearable sensor probe (2026-10-21)

## Outcome and scope

This was a narrow development experiment for one optional sensor modality. It
tests whether a frozen, pretrained time-series representation is useful for an
18-way wearable activity decision. It does not test text, image, audio, video,
sensor fusion, JSON generation, quantization, or the full product target. The
held-out participant partition was not evaluated, and this result does not
close R2-M-2 or justify product promotion.

The frozen MOMENT representation plus a small linear readout reached 60.32%
validation Accuracy and 59.72% macro Accuracy. A fixed handcrafted-statistics
representation with the same readout procedure reached 26.98% and 24.17%.
The paired participant-cluster bootstrap favored MOMENT on this fixed
development split, but the absolute score is far below the aspirational
95–98% goal.

## Data and split

Source: [PCH Data 2017](https://doi.org/10.21251/b9566edc-c68c-4eee-b67e-c0228d0d3833),
anonymized wrist accelerometer recordings with 18 activity labels. The official
dataset release lists CC BY 4.0. The downloaded archive is 98,272,925 bytes,
SHA-256 `7814f871c834725d0108192cc9265b1abd2db8ead96815d82c828202f1e55ba9`.

Participants were assigned by a seed-17 hash split. To prevent absent classes
from making the 18-way validation metrics undefined, a deterministic
participant-level coverage pass exchanged fit participants into validation
until every class had at least two validation participants. The 28 participant
IDs initially assigned to heldout were left unchanged. Coverage swaps used
fit/validation filenames only; heldout labels were not parsed. Fit and
validation share no participant, source-record SHA-256, or normalized-window
signal SHA-256.

| Partition | Participants | Recordings | Status |
| --- | ---: | ---: | --- |
| Fit | 92 | 276 | Used to train the linear readouts |
| Development validation | 21 | 63 | Used for early stopping and comparison |
| Heldout | 28 | Not opened by this run | No metrics produced |

| Split | Participant-ID set SHA-256 |
| --- | --- |
| Fit | `ced7c047eff00d101aa8975aea41845685d87d327752e1de15b3e84c46691b6a` |
| Validation | `e1925e75b43736e305567fc4bc42b952c9ad42cdb6771118afe04b64ea42f373` |
| Heldout | `4dc5a3e76d3fcb19ef2cff19a18c92db88269a077ca6b180866e99c4be2bd901` |

The validation sample-ID order SHA-256 is
`32ba9659a2b6cec52812ddb4ac0bd60fab073f39ba92901d04c056102d12d3f9`.
Validation labels are development data, not a blind audit. All six
food-preparation validation recordings come from only two participants, so
those class-specific scores have low precision. The archive's CRC check was
run during acquisition before the split; that read archive bytes but did not
extract heldout labels or calculate heldout metrics. The preflight and model
run subsequently skipped heldout members before label parsing or file reads.

## Input processing and model

The input is the calibrated 3-axis accelerometer only. Timestamps are resampled
to 51.2 Hz and converted to nonoverlapping 128-sample (2.5-second) windows.
Windows never cross a timestamp reset or a gap greater than three times the
median interval. Sparse missing axis values are interpolated only when the
per-axis missing rate is at most 1%; three values across the fit and validation
recordings were interpolated. Incomplete trailing windows are dropped. The run
processed 16,141 windows in 253 inference batches of 64.

The frozen encoder was `AutonLab/MOMENT-1-small`, embedding configuration,
revision `411e288267f82cce86296dbe4d6c8bc533cc162f`; its source checkout was
`38f7310ad594100747ca2a8357e9c7ca7d323e0e`. Encoder weights were FP32 and
SHA-256 `785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877`.
Window embeddings were averaged per recording. The encoder remained frozen;
only a 9,234-parameter linear readout was trained. Total model plus readout was
35,346,642 parameters.

The reference readout used per-window mean, standard deviation, min, max, RMS,
and five quantiles for each axis and acceleration magnitude. Both readouts used
the same CPU AdamW trainer, seed 17, batch size 64, maximum 300 epochs,
patience 25, and best validation NLL selection (`min_delta=1e-5`). Neither
condition used class-weighted loss. Model selection used validation only.

## Validation results

| Readout | Accuracy | Macro Accuracy | NLL | Brier | ECE (15 bins) | Mean confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen MOMENT + linear | 0.6032 | 0.5972 | 1.0860 | 0.5627 | 0.1504 | 0.7243 |
| Fixed statistics + same linear trainer | 0.2698 | 0.2417 | 36.1216 | 0.9621 | 0.2534 | 0.2243 |

Per-class MOMENT validation Accuracy (record count):

| Activity | Accuracy (n) | Activity | Accuracy (n) |
| --- | ---: | --- | ---: |
| Arm curl | 0.75 (4) | Bounce | 0.75 (4) |
| Brushing teeth | 0.50 (2) | Catch | 0.50 (4) |
| Chopping vegetables | 0.50 (2) | Deadlift | 0.25 (4) |
| Hair grooming | 0.50 (2) | Ironing | 0.25 (4) |
| Jogging | 0.80 (5) | Lateral arm raise | 1.00 (4) |
| Mixing food | 0.50 (2) | Pass | 0.50 (4) |
| Sieving flour | 0.50 (2) | Stepping | 0.40 (5) |
| Walking | 0.80 (5) | Washing dishes | 0.25 (4) |
| Washing hands | 1.00 (2) | Washing windows | 1.00 (4) |

Paired participant-cluster bootstrap: 2,000 replicates, seed 17, percentile
95% intervals, resampling the 21 validation participants with replacement.
Differences are MOMENT minus fixed-statistics readout:

- Accuracy difference: `+0.3333`, 95% interval `[+0.1587, +0.4921]`.
- NLL difference: `-35.0356`, 95% interval `[-5.0952, -1.0954]`.

The large NLL gap reflects severe overconfidence in the fixed-statistics
baseline; it is not a comparable indication of a 35-point quality gain. Both
conditions are single-seed, one-split results. The validation cohort is small,
and selection and reporting both use that development split.

## Compute and size

- GPU: NVIDIA GeForce RTX 3080 Laptop GPU, capability 8.6.
- Runtime: Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, cuDNN 9.1.
- HotUGPU waited for the occupied GPU to yield, reserved 2,471 MiB, then
  launched the child with 13,743 MiB free. No process was killed or signaled.
- Preprocessing: 7.41 s; frozen feature extraction: 8.30 s; both CPU readouts:
  0.32 s; full run: 22.09 s.
- Peak CUDA allocated/reserved: 261,141,504 / 308,281,344 bytes.
- Encoder optimizer updates: 0; cloud cost: $0.
- Encoder checkpoint size: 151,615,328 bytes FP32; quantized package size was
  not measured. This one-modality result is not evidence that a full multimodal
  model meets either the 100–200M parameter or package-size target.

Two expected runtime notices were observed: the Windows allocator does not
support `expandable_segments`, and MOMENT warns that its classification head
was not pretrained. The experiment uses the frozen embedding output and trains
its own linear classifier. No numerical or memory error occurred.

## Interpretation and next work

The experiment supports further evaluation of frozen time-series embeddings
for this activity task. It does not show 95–98% performance, robust transfer,
multimodal complementarity, sensor/audio/video fusion, or general decision
quality. Pretraining overlap is unknown: PCH Data 2017 was described in a 2018
paper even though this dataset release is dated 2026. Do not call these records
unseen by MOMENT pretraining.

The fit/validation result is not a product selection. Next, preserve this
checkpoint and validation record, then prospectively evaluate a distinct
participant-heldout cohort before considering any fusion work. Keep the
heldout partition untouched until its protocol is frozen. The 95–98%
multimodal target, the base model, and all other modality requirements remain
unchanged.

## Artifacts

External run directory:
`C:\CodexArtifacts\pretrained-reuse-r2\pch-moment-activity-probe-attempt-01\`.

| Artifact | SHA-256 |
| --- | --- |
| Exact run script | `561fc05c3053ce0ecad655978edeca4f09471473d42990b0525d008fdc2bd0e1` |
| Preflight | `d7fe349a132caf77caffb811a1f4dbebd22205568d9fd8ccf492a31e386fd959` |
| Result JSON | `e14255295125421708bdcd82edeaa20364b8698b46ac09f279b57da19ed92633` |
| MOMENT best readout | `1abec7fd55857f569b8c4ce3990477aabd02c694adb529682d6f66ff3bd1a895` |
| Feature cache | `b928fa2bb08f7a23db2f5e240b99bb92bc609ac5b84e25db401bb8a564b85857` |

The external directory also contains both readout histories, validation
predictions with per-example probabilities, the fixed-statistics checkpoint,
the exact command, and `artifact-hashes.json`. Model weights and the PCH media
archive are not committed to Git.
