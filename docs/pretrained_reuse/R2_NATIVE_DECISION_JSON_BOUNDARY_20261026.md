# Typed text-free Decision JSON boundary (2026-10-26)

## Implemented

`native_decision_json` accepts the versioned `ObservationBatch` contract and
one probability per supplied typed option. Its JSON preserves option IDs and
typed attributes, reports present modalities, and returns the selected option
and confidence. It does not require an option sentence, prompt, transcript, or
language-generation model. Probabilities are checked for alignment,
normalization, finiteness, and range. Explicit abstention emits null option
probabilities instead of a fabricated distribution.

The serializer is an output-boundary primitive, not an encoder, trained model,
calibrator, or quality result. Feature-level `decide_native` adapters now
connect the fixed-taxonomy audio and image readouts to this output. They reject
mismatched modalities, text criteria, duplicate class IDs, and invalid option
schemas; incomplete observation masks abstain. They consume frozen feature
tensors supplied by the caller and do not load raw media or authenticate that
a feature tensor came from the declared payload. Video, sensor, and screen
readouts are not yet migrated. R2 M-0/M-1 remain open until the remaining paths
and feature-provenance handoff are covered and each path is measured on its task.

## Verification

- New JSON tests were first run before implementation and failed because the
  serializer module did not exist.
- After implementation: `tests/test_native_decision_json.py` and
  `tests/test_native_observations.py`: **27 passed**.
- New audio/image adapter tests were first run and failed because those
  adapters did not exist. A tie-case test caught and prevented the serializer
  from changing the readouts' existing class-based tie break.
- Adapter and related readout tests: **38 passed**.
- Full CPU suite after adapter integration: **2,509 passed, 11 skipped**.
- `ruff check .`: passed.
- CI-equivalent model/dataset manifest, catalog, and example audit commands:
  passed.

No media, checkpoint, corpus, model weights, GPU, or sealed audit was used.
