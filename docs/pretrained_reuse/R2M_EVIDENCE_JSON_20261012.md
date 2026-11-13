# Evidence-gated JSON boundary

`evidence_decision.py` adds an inference callback gate for a fixed complete
classification taxonomy. Missing required modalities produce JSON status
`abstain`, reason `insufficient_evidence`, null selected ID and null probabilities.
The callback is not invoked. Availability uses strict booleans: numerical zero
is not a substitute for missingness. Options are validated even on abstention.

The aligned wrapper binds each required modality to a distinct exact source ID.
Existing causal alignment selects the latest valid sample at/before the query
within the caller's explicit maximum age. Different clocks are rejected by the
alignment contract. The scorer receives selected indices/ages and must verify
and load the matching payloads. No confidence threshold is fitted here.

This is metadata/control integration, not an evaluated streaming model. The
VIDIMU episode-pooled readouts are NOT made causal by this wrapper, and their
uncertain cross-device offsets remain unresolved. No payload verification,
learned missing-modality robustness, calibration improvement or production
readiness is claimed. Binding a real time-indexed scorer remains required.

Validation: nine new gate tests plus nine existing alignment tests passed.
Whole local suite:544 passed,1 skipped; whole-repository Ruff passed. No training,
checkpoint modification, audit-set access or additional neural parameters.
