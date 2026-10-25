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
calibrator, or quality result. Existing modality-specific runners have not all
been migrated to it; R2 M-0/M-1 remain open until the request contract is wired
through the supported text-free modality paths and measured on their respective
tasks.

## Verification

- New JSON tests were first run before implementation and failed because the
  serializer module did not exist.
- After implementation: `tests/test_native_decision_json.py` and
  `tests/test_native_observations.py`: **27 passed**.
- Full CPU suite: **2,501 passed, 11 skipped**.
- `ruff check .`: passed.
- CI-equivalent model/dataset manifest, catalog, and example audit commands:
  passed.

No media, checkpoint, corpus, model weights, GPU, or sealed audit was used.
