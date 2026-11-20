# R2 sensor temporal experiment protocol — UCI HAR

## Purpose and limits

This is a small, sensor-only M-1 experiment for the causal temporal Decision
path. It compares a fixed temporal LSTM against a trainable static summary
readout on the same participant-disjoint task. It does not evaluate image,
audio, video, sensor fusion, or the full Omni product, and it cannot close R2
or establish the project’s 95–98% product target.

UCI HAR is used here only with models trained from scratch. MOMENT or any other
pretrained sensor model is not loaded, so the unresolved UCR/UEA overlap that
blocked using this source for MOMENT selection does not apply to this probe.
The official repository identifies the dataset as CC BY 4.0 and reports 30
participants, six activities, 50 Hz sampling, and 128-sample windows with 50%
overlap. Cite the dataset as Reyes-Ortiz et al., DOI
[`10.24432/C54S4K`](https://doi.org/10.24432/C54S4K); see the [official UCI
record](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).
The experiment uses only six preprocessed body-acceleration and gyroscope
channels, labels, and participant IDs. It does not use the companion video.

## Frozen protocol

- Source: official UCI HAR archive, dataset ID 240, CC BY 4.0; archive SHA-256
  is recorded in the external run artifacts and final report.
- Official participant split: use only official `train` rows during fitting;
  the official `test` rows are not parsed until after `candidate-freeze.json`
  exists and its config/checkpoint hashes verify.
- Development split: sort the official train participant IDs, shuffle with
  Python `Random(seed=17)`, reserve the first four participants for validation,
  and use the remaining participants for training. No window from one
  participant crosses this split.
- Input: six ordered channels `body_acc_{x,y,z}` and `body_gyro_{x,y,z}`;
  128 samples per window at 50 Hz. Per-channel mean and standard deviation are
  fitted only on the training-participant windows.
- Typed decision: one six-option enum over the fixed UCI activity names; the
  target is the activity label for the full window. The model receives no
  participant ID, sample ID, free-form question, target, or option label text.
- Arms: (1) static mean/std/min/max summaries with the shared typed scorer;
  (2) the causal 64-hidden-unit LSTM stem with the same option schema and
  128-hidden-unit readout. Both are initialized from scratch with seed 17.
- Initial training: 12 epochs, batch size 128, AdamW, learning rate 0.001, weight decay
  0.01, one CPU thread, deterministic Torch algorithms. Select the earliest
  epoch with minimum validation NLL. Preserve best and terminal weights and
  all epoch metrics plus predictions for the selected validation checkpoint.
- Metrics: sample Accuracy, macro recall over represented activities, NLL,
  multiclass Brier, equal-width 10-bin ECE, mean confidence, and per-activity
  support/recall. The test comparison additionally uses a paired
  participant-cluster bootstrap (2,000 draws, seed 17).
- Public-test policy: freeze both selected checkpoints and their effective
  configs before parsing the official test signal/labels. Evaluate both arms
  once using the same preprocessing, then preserve every prediction. The
  official public test is a known development reference, not a blind global
  audit; it is not used for model selection, tuning, or retraining.
- A separate, validation-only 100-epoch follow-up is registered in
  [`R2_UCI_HAR_SENSOR_LONG_BUDGET_PROTOCOL_20261120.md`](R2_UCI_HAR_SENSOR_LONG_BUDGET_PROTOCOL_20261120.md).
  It does not reopen the public test.
- Compute and output: CPU only, local machine, no cloud. Dataset and model
  outputs remain outside Git under a new `C:\CodexArtifacts` directory. Existing
  corpora, checkpoints, branches, and artifacts are read-only.

## Interpretation rule

This comparison asks whether the temporal architecture improves this narrow
participant-held-out activity task over simple order-insensitive summaries.
If temporal quality does not improve, report that result without declaring an
architecture or modality ceiling. Even a high score remains sensor-only,
dataset-specific development evidence and cannot be combined with unrelated
benchmark accuracies as an Omni score.
