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
calibrator, or quality result. `decide_native` adapters connect the audio,
image, video, and sensor readouts to this output. Audio and image consume frozen
feature tensors supplied by the caller. Video also consumes eight ordered
frozen frame features and explicit typed program tokens; it rejects natural
language criteria. Sensor loads and verifies its referenced payload directly.
The audio/image/video adapters do not authenticate that a supplied feature
tensor came from the declared payload. Incomplete observation masks abstain;
modality, criterion, vocabulary, option, and timestamp mismatches fail closed.
The screen path is not implemented. R2 M-0/M-1 remain open until screen
coverage, feature-provenance handoff, and task-level metrics are addressed.

Follow-up 2026-10-27: the explicit `decide_bound_native` path now verifies actual
media bytes, the full trusted cache key, independent feature-payload hash,
source/clock/timestamps, shape and finiteness before calling these readouts.
The existing tensor-level methods remain low-level operations. A 24-request
real-checkpoint replay preserves their JSON exactly; the trusted historical
extraction and broader quality limits remain. See the
[feature-binding report](R2_NATIVE_FEATURE_BINDING_20261027.md).

## Verification

- New JSON tests were first run before implementation and failed because the
  serializer module did not exist.
- After implementation: `tests/test_native_decision_json.py` and
  `tests/test_native_observations.py`: **27 passed**.
- New audio/image adapter tests were first run and failed because those
  adapters did not exist. A tie-case test caught and prevented the serializer
  from changing the readouts' existing class-based tie break.
- Audio/image adapter and related readout tests: **38 passed**.
- Video adapter tests failed before its implementation; targeted audio/image/
  video/sensor regression suite: **28 passed**.
- Full CPU suite after integration: **2,512 passed, 11 skipped**.
- `ruff check .`: passed.
- CI-equivalent model/dataset manifest, catalog, and example audit commands:
  passed.

No media, checkpoint, corpus, model weights, GPU, or sealed audit was used.
