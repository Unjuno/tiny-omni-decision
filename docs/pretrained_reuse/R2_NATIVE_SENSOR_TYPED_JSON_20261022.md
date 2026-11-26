# Native sensor typed Decision JSON adapter

## Change

Added a small adapter from the existing HARTH sensor-window runtime to the
caller-ID Decision JSON contract. The adapter accepts caller-provided option
IDs and text, validates that their order matches the sensor result, and
preserves the existing center-label `temporal_scope` fields.

Selected results carry the 12-option probability distribution. An incomplete
sensor window carries `status: "abstain"`, a reason, no selected option, no
confidence, and `null` option probabilities. It does not fabricate a uniform
distribution or convert missing input into a low-confidence prediction.

The generic JSON validator now accepts an explicitly selected option when
multiple options share the maximum probability. This preserves deterministic
task-specific tie-breaks, such as the sensor runtime's semantic activity-code
tie-break, while still rejecting a selected option below the maximum.

## Verification and limits

- CPU tests cover selected and abstaining sensor records, JSON serialization,
  option order, temporal scope, and maximum-probability ties.
- This is a serialization/runtime-contract change. It adds no model parameters,
  performs no training or new quality evaluation, and makes no quality or
  calibration claim.
- The optional sensor route remains research-only. It does not satisfy the
  synchronized multimodal quality gate or the four required modality targets.
- No GPU work was started: the local RTX 3080 remained occupied by the
  unrelated Orbitune process during this change.
