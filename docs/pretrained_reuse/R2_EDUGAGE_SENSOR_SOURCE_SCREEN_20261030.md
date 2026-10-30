# EduGage optional-sensor source screen — 2026-10-30

## Disposition

**Not admitted as the R2-M synchronized video/sensor task.** EduGage may be a
narrow sensor-only ordinal Decision candidate, but it cannot establish whether
video and sensors jointly improve decisions because its released participant
records do not include the viewed lecture videos. It does not replace the
four-modality Path A objective.

## Publisher evidence

The authors' [public repository](https://github.com/ZikangLeng/EduGage) reports:

- 16 learners, 64 lecture sessions, 699 self-ratings and roughly 12 hours of
  synchronized sensor records.
- Each target is a participant's 1–5 response to how difficult it was to pay
  attention during the preceding minute. Inputs are the 44 seconds of sensor
  data before the probe.
- The released participant files are an engagement log and available wearable
  / gaze streams (EEG, IMU, PPG, ECG, EDA, heart rate, eye tracking and related
  signals). The README does not list synchronized lecture-video files among
  the released participant data; lecture videos are identified as MIT Open
  Learning material.
- The authors state the 3.05 GB dataset is CC BY 4.0 and that participants
  consented to release of de-identified data. This is the publisher's statement;
  this screen downloaded no records and independently checked no media.
- Its reported benchmark uses four fixed participant-grouped folds and reports
  published results for fifteen methods. Those scores are already public and
  cannot serve as a blind test for this project.

## Fit and limits

The ordinal five-option target is technically compatible with calibrated
Decision probabilities, and the synchronized time-series streams could support
a narrow temporal sensor study. However, the task measures a subjective
self-rating rather than an externally verified world state. It has only 16
independent learner groups, and the source page reports persistence of ratings
and strong sensor-free peer-window baselines. Any future experiment would need
to compare against participant / session history, fix participant folds before
selection, and evaluate modality-specific sensor ablations. It would remain a
sensor-only auxiliary result unless the original videos become part of an
explicitly licensed and pinned input corpus.

## Action

No dataset bytes were fetched, no model was fitted, and no existing project
artifact was changed. The repository page provides enough evidence to reject
EduGage as the current paired video+sensor source; no storage increase is
needed for this disposition. Keep the main R2-M gate open and continue seeking a
rights-cleared, adequately grouped source whose actual inputs include both
modalities.

Primary source: [EduGage authors' repository](https://github.com/ZikangLeng/EduGage),
read 2026-10-30.
