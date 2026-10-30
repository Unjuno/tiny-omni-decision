# EgoViz ordered cross-modal GRU readout protocol (2026-10-30)

## Question

Does one small recurrent readout over the already-frozen VideoPrism and
MOMENT features improve temporal video-to-IMU window matching over the
fixed linear temporal scorer, without changing the corpus or encoders?

The previous fixed-feature controls showed that reversing either stream or
shifting sensor patches changes the selected scorer's decisions. That supports
testing one sequence-aware head; it does not establish that a GRU will help.

## Frozen conditions

- Source branch/commit: `codex/r2-egoviz-train-coverage-20261030` at
  `135dac6cd6d3b936af2d8dca3df4b20d256c1e14`.
- Train/development corpus and features: the merged 773-example / 48-session
  training corpus and the exact same 307-example / 20-session development
  selector from the coverage comparison. Verify every input hash against
  `merge-report.json` before execution.
- Model inputs: existing 8 × 768 VideoPrism features and existing six-channel,
  16-patch × 512 MOMENT features. Encoders remain frozen and are not loaded.
- Candidate: project video and sensor tokens independently to 64 dimensions;
  mean MOMENT channels and pair adjacent sensor patches to align 16 patches to
  8 video positions; form `[video, sensor, video*sensor, abs(video-sensor)]`
  tokens; process each 8-step candidate sequence with one shared, unidirectional
  GRU (hidden size 64); score each candidate from its final state. No candidate
  position embedding is used, so scoring is option-permutation equivariant.
- Seed 17; AdamW, LR `0.003`, weight decay `0.01`, batch size 32, four CPU
  threads; cross-entropy; no dropout, scheduler, calibration, or data changes.
- Exactly 5,632 optimizer updates. No performance-based early stop. Save/evaluate
  at updates 352, 704, 1,408, 2,816, 4,224 and 5,632. The step schedule and
  random-permutation generator match the expanded linear comparison. Final
  checkpoint is the sole fixed endpoint; no best-by-validation selection is
  made for this candidate.
- Compare to the already-saved linear candidate at the same six update counts,
  with paired bootstrap over the 20 fixed development sessions (2,000 draws,
  seed 17). This development set has been repeatedly observed; inference is
  exploratory and cannot qualify a product or substitute for a blind audit.
- CPU-only execution; no GPU, cloud, new media, sealed audit, or output reuse.
  Output root is `C:\CodexArtifacts\egoviz-gru-temporal-budget-20261030-attempt-01`.
  Keep at least 3,000,000,000 bytes free throughout.

## Falsifiable interpretation

The temporal-head hypothesis receives limited support only if the fixed
5,632-update endpoint improves both development Accuracy and NLL over the
linear head and the 1,408 / 2,816 / 4,224 trajectory does not show a short-lived
spike followed by regression. Any mixed result is uncertain. Failure to
improve on this development set does not establish a model-family limit.

All performance comparisons are post-hoc on a development set previously used
for selection. Do not change architecture, hyperparameters, or task based on
these metrics and do not evaluate the sealed audit.
