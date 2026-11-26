# R2 M1 modality contract hardening — 2026-11-04

## Finding

The 2026-11-04 screen-routing correction exposed duplicated runtime modality
allowlists: the public `Observation` schema, feature cache, temporal session
state, and text-free M1 adapter could drift independently. A newly supported
modality could therefore validate at request construction and still fail later
in the feature/reuse path.

## Change

`observations.py` now derives one immutable `OBSERVATION_MODALITIES` set from
the `Modality` Literal and the `NATIVE_DECISION_MODALITIES` set by excluding
text. The feature cache and temporal state use the shared observation set; the
single-stream M1 adapter uses the shared non-text set. No scoring, feature
representation, task, checkpoint, or preprocessing behavior changed.

A regression test iterates every declared modality through feature-key
construction and insertion into temporal state. This guards the contract from
drifting again while keeping text valid for cached temporal context and
excluding it from the text-free M1 route.

## Verification and limits

- Focused cache/temporal/native-decision suite: **77 passed**.
- Ruff passed on all changed Python files; `git diff --check` passed.
- CPU-only; no model, media, validation, sealed-audit data, GPU, or cloud compute
  was used.
- This is contract consistency evidence only. It does not measure screen
  accuracy, cross-modal benefit, calibration, or product readiness.

## Next R2 gate

The current source/task matrix still has no task admitted for another quality
fit. Define a concrete end-user task battery and deployment limits, then admit
only a rights-cleared source with independently grouped train/calibration/dev
data and task-matched controls. Existing exposed validation sets remain
diagnostics and must not be retuned.
