# RoboMNIST temporal readout protocol — 2026-11-05

Prior TRAIN-only cross-validation found overfitting in the pooled readout, while
lowering its cosine multiplier reduced overconfidence but harmed Accuracy. This
fixed comparison tests whether preserving each encoder's eight ordered features
helps AV correspondence, before product or broad-task claims.

- Same 780 original TRAIN groups, 390 pair units, 1,560 rows; identical seed17
  five grouped folds. No original validation feature arrays or audit opened.
- Two fresh model arms per fold, initialized from the same seed and same parameter
  shapes: separate one-layer audio/video LSTMs, hidden width16. Ordered uses each
  stream's original token order. Shuffled uses a deterministic per-source,
  per-modality permutation, reused consistently for every sample containing that
  source. Both use identical model capacity, optimizer and training schedule.
- Keep the original 400 updates and report 1/25/50/100/200/400. Fit normalization
  per fold from fit groups only. Report OOF per-example predictions and all
  Accuracy/NLL/Brier/ECE curves; no winner selection, calibration, or tuning.
- Audio's two-second windows and video's frame sampling have no shared clock.
  Each modality gets its own LSTM; no aligned cross-attention or token-to-token
  synchrony is assumed. Per-token VideoPrism embeddings are already contextual.
- Model has76,032 trainable parameters versus73,728 in the baseline (~3.1%
  additional parameters). Combined encoders plus readout are116,402,455 parameters.
  These are encoder/readout counts only, not a full four-modality product bundle.
- This correspondence task does not establish semantic action quality, room/scene
  generalization, a product task gate, or general Omni reasoning. All results are
  internal development and must be interpreted with the preceding overfit curves.
