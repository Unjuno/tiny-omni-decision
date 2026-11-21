# R2-M time-aware gated fusion primitive (2026-11-20)

## Scope

Added `src/tiny_omni_decision/time_aware_gated_fusion.py` as an opt-in readout primitive for timestamped frozen features. It is intended to support audio+sensor and future video+sensor routes without loading or unfreezing any encoder. It returns candidate-aligned hidden states; a separate typed Decision scorer must turn those states into option probabilities.

The block takes an explicit stream per configured modality, each with `[batch,time,features]` values, millisecond timestamps, valid-token masks and a separate per-example presence mask. Relative event time is encoded against the latest valid token in that example. Candidate queries cross-attend to the concatenated token memory, then each modality contributes a separately candidate-conditioned sigmoid-gated residual. Missing modalities are masked and cannot contribute a residual. Invalid padded values are zeroed before projections. A total 8,192-token bound fails closed.

Checkpoint extra state pins modality names and dimensions, query/hidden/head sizes, time scale, and time/gate policy. A mismatched or missing configuration is rejected before parameter mutation. Constructor defaults have **141,986 trainable parameters** for an audio width of 1 and sensor width of 3, query width 5, fusion width 96 and four heads. This count excludes all upstream encoders and any Decision scorer.

## Verification

Fourteen focused CPU tests and Ruff pass. Tests cover candidate permutation, padding invariance, present-zero vs absent masks, timing sensitivity, gate ablation, invalid timestamp/mask rejection, gradients and checkpoint-policy integrity. The exact-padding test permits a `5e-7` floating reduction tolerance because adding masked sequence padding changes CPU attention reduction order.

These tests establish API and numerical mechanics only. The block is randomly initialized and **untrained**; no task, accuracy, fusion gain, calibration, inference cost or product admission is established. The implementation does not synchronize clocks or certify alignment. The selected source candidate is still pending archive verification and data-format audit.

## Next experiment gate

After the frozen Zenodo audio/vibration source is downloaded and verified, first freeze participant/trial splits and check whether within-trial paired windows are valid. Compare the same train/validation IDs under:

1. audio-only and vibration-only match baselines;
2. concatenated masked mean-pool MLP control;
3. this time-aware candidate-conditioned gated fusion.

Freeze all candidate definitions, pairing rules and validation selection before fitting. Keep same-activity/different-participant hard negatives where the source supports them. Run shuffled-pair, time-offset, missing-stream, and contradiction controls using fixed weights. Stop before model fitting if trial pairing leaks identity or if grouped splits do not support the task. Do not use sealed audit rows for iteration.
