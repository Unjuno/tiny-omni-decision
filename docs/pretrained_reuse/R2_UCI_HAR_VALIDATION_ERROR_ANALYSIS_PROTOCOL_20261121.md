# UCI HAR selected-validation error analysis — frozen protocol

## Scope and limits

The 100-epoch validation-only static/LSTM run is complete. This is a one-time
descriptive analysis of its already-exposed, checkpoint-selection validation
set. It addresses the pre-recorded stationary-class collapse and checks
whether the static/LSTM difference is consistent across the four validation
participants. It does not reselect either checkpoint or reopen the public
test.

## Frozen inputs

- Run directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\uci-har-sensor-v1\attempt-02-long100`.
- Use only `static-best-validation-predictions.jsonl` and
  `lstm-best-validation-predictions.jsonl`, after checking their byte sizes and
  SHA-256 against `artifact-index.json`.
- Use the existing best-checkpoint selection in `candidate-freeze.json`:
  static epoch 84 and LSTM epoch 15. No weight file is loaded.
- Verify sample IDs, targets and subject IDs match exactly before comparison.
- Validation contains 1,324 overlapping-window records from four held-out
  participants. These records were already used for selection and are not
  independent windows or a blind estimate.

## Fixed analysis

1. Report overall and per-subject Accuracy, macro Accuracy, NLL, multiclass
   Brier, and the existing fixed-bin ECE definition.
2. Report each activity's support, recall and precision, plus the complete
   6×6 confusion matrix for each selected checkpoint.
3. Tabulate paired static/LSTM correctness by subject × target class and show
   the per-subject metric differences. Do not use a window-level bootstrap as
   if overlapping windows were independent. With only four validation
   participants, treat subject-level results as descriptive rather than
   population inference.
4. Preserve all 1,324 saved prediction rows, their order and probabilities;
   write a compact analysis JSON and a concise report. The predictions and
   checkpoints stay outside Git; no signals are loaded.

## Prohibited actions

Do not load or score official test records; do not load the UCI archive; do not
train, calibrate, choose another epoch, or change the model/data/split. Do not
interpret this validation analysis as a new quality estimate. Do not use these
results to claim that LSTMs generally fail or help sensor tasks.
