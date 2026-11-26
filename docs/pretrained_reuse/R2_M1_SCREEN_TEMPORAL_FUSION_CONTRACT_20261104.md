# R2 M1 screen feature integration correction — 2026-11-04

## Finding and change

The R2 contract and `Observation` schema include `screen`, and the single-modality screen adapter had CPU smoke coverage. The durable `ObservationFeatureKey`, `SessionTemporalFeatureState`, and `decide_temporal_native_fusion` allowlists nevertheless omitted `screen`, so a screen feature could not be stored in the reusable cache or routed through time-aware fusion. This contradicted the declared optional screen-stream contract.

Added `screen` to those three allowlists and added a regression that creates hash-pinned screen/video feature keys, inserts both streams into the bounded session state, and returns typed-option probabilities from the temporal fusion path. The test checks interface routing only; it does not claim screen perception quality, correct action selection, or a learned cross-modal benefit.

## Verification

- The regression failed before the change at `ObservationFeatureKey` with `unsupported observation modality: screen`.
- After the change, the relevant CPU suite passed: **76 passed** across temporal decision/state, screen/audio/video smoke, feature cache, observation schema/alignment, and aligned-feature tests.
- Ruff passed for the three changed modules and the test; `git diff --check` passed.
- No model weights, media, validation predictions, sealed audit, GPU or cloud compute were used. Existing feature artifacts and checkpoint files remain unchanged.

## Remaining gate

This closes a code-contract gap only. The screen-to-action M1 quality task and real screen/video fusion task remain unmeasured. R2-M still needs a fresh, rights-cleared, independently grouped task where both streams are necessary; exposed validations remain excluded from tuning.
