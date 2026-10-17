# AV gate cost and input diagnostic

## Answer to the parameter-cost question

The learned audio/video gate has 61,634 parameters: 0.031% of the
197,098,333-parameter pre-quantized system. Its parameter count is not a
meaningful size risk. The functional result is the concern: in the completed
E0G1 fit both gate outputs fell below 0.01 for every valid candidate by update
512 and stayed near zero. The joint score change therefore does not show that
the gate learned to choose complementary audio and video evidence. The base
readout still sees the inputs, and the residual path also sees the query.

The proposed full fixed-input check is not a training run. It would score the
same 8,875 frozen validation questions through five pre-registered input views
for E0G0 and E0G1, verify natural predictions against the saved checkpoint
references, and report only target-free prediction sensitivity for donor
replacements. It is useful for separating input dependence from query-only
effects, but the partial run below is not sufficient to draw that conclusion.

## Execution and current status

- Source branch: `codex/r2-av-postnorm-stability`.
- Scoring implementation: commits `e3c3239` through `81134b7`; latest source
  includes a fail-fast GPU process check. CPU tests: 27 passed; Ruff passed.
- Fixed inventory: 8,875 questions, five views, seed 17, 3,693 unique donors,
  maximum donor reuse 14. Inventory result SHA-256:
  `f12690080072e7cb0c60e0528394367114afc296f8c9cbe3d281425937687c62`.
- Inputs and selected checkpoints were pinned to the existing matched audit.
  No training update, checkpoint selection, sealed-audit access, or cloud use
  occurred.
- E0G0 attempt 04 emitted 500 of 44,375 planned forwards (the first 100
  questions across their five views) before it was stopped. Its partial output
  is retained at
  `C:\CodexArtifacts\pretrained-reuse-r2\greatest-hits-postnorm-replacements-E0G0-04`.
  No aggregate sensitivity summary was produced, so these rows are not used to
  estimate a metric.
- The scorer process allocated 116,764,160 bytes of CUDA memory. At the same
  time, the Orbitune training process occupied about 15,821 MiB of the 16,384
  MiB device. Its automatic yield did not occur during this inference run.
  We left those processes untouched and stopped our scorer.
- E0G0 attempt 06 failed closed before loading features or scoring after its
  GPU preflight found external compute PID `27384`. Attempts 01–03 also failed
  before any forward due to runner setup defects; each has a separate retained
  incomplete receipt. Attempt 05 was stopped before feature audit and emitted
  no predictions.

## Interpretation and next step

There is no reason to reduce model size to solve the gate issue: the gate is
already a negligible part of the parameter budget. The evidence supports
keeping the gate disabled in the current design unless a completed frozen
input-sensitivity analysis demonstrates a useful effect. It does not support
claiming that audio/video inputs are ignored by the whole model.

The E0G0/E0G1 five-view comparison remains incomplete. Continue only when the
existing local GPU is free; then run the committed scorer with a new attempt
number and do not reuse any incomplete output directory. Do not start another
fit just to justify the gate's size, and do not change the frozen inventory,
checkpoint, corpus, or validation labels.
