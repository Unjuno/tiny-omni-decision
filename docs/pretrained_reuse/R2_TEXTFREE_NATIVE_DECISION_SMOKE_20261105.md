# Text-free native Decision wiring smoke (2026-11-05)

**Status: interface smoke passed; modality quality remains unmeasured.** This
implements two explicitly open R2-M engineering items. It does not train a
model, evaluate real-world quality, or close M1/M2.

## What was exercised

- Added `src/tiny_omni_decision/native_decision.py`, an inference adapter from
  hash-verified, timestamp-aligned cached features to typed Decision JSON for
  one non-text observation. It accepts only explicitly named numeric option
  attributes as model features. Option IDs, display text, request IDs, source
  IDs and cache paths are validation/serialization metadata and are never
  passed to the scorer.
- Five M1 fixtures route synthetic cached features for **audio, video, image,
  screen and sensor** separately. Each has one observation, no criterion and no
  text observation. The smoke scorer has 41 parameters; it is randomly
  initialized and is not a quality model.
- The M1 invariance test changes the cache filename, source ID, option IDs,
  display strings and request ID while holding features and typed option
  attributes fixed. The option probability vector is unchanged and the selected
  ID still follows the selected option position. A clean Python subprocess
  imports the runner without importing `transformers`.
- Added one synchronized synthetic M2 Video+Sensor fixture with four equally
  represented binary factor combinations and an XOR label `[0,1,1,0]`. A
  lookup using either modality alone obtains 50% on this truth table; both
  factors are needed to determine the label. Four aligned timestamps, shared
  clock metadata, cache hashes, explicit masks, numeric options and the
  existing `FusionDecisionScorer` path are exercised. The tiny random fusion
  scorer has 74 parameters. It is used only to check finite normalized output,
  not to measure accuracy or learn the XOR rule.
- The fixed task contract requires both Video and Sensor evidence. Removing
  Sensor returns an `abstain` record without invoking the scorer.

## Verification

Focused tests: **9 passed**. Full repository validation: `ruff check .` passed,
the new Python files passed Ruff format checking, and `pytest -q` reported
**2,681 passed, 11 skipped**.

- Implementation commit: `9a035f5a1dbfc5978f676b070bb4d6db0e3aa16c`.
- Runner SHA-256: `49d9cf7e2e5e4f38b14130c2a1d0f5c6f5459d645eea81921f599efa6359b597`.
- Test SHA-256: `f1b856321d4cbcf8b4b3d009bfe035a063c5881b443baab56e7ac7d6fc1f1107`.
- Runtime: local Python 3.11 / PyTorch 2.6.0+cu124 build, CPU execution only.
  No media, pretrained checkpoint, GPU or paid compute was used.

## Limits and next gate

This proves that the metadata/cache/typed-option/JSON path can execute without
a mandatory text encoder and can fail closed when required sensor evidence is
missing. It does not show that any pretrained feature identifies a real event,
that the fusion scorer uses both learned signals, or that quality improves over
unimodal models. The four-row synthetic truth table is a wiring fixture, not a
benchmark or a substitute for an admitted synchronized dataset.

The R2-M quality gate still needs rights-cleared, source-compatible and
independently grouped real M1 tasks, plus synchronized paired M2 examples with
matched unimodal, replacement, timing and missing-input controls. No existing
source screened so far clears that full gate. Keep M2 quality `NOT TESTABLE`
until such data pass the source, label and split audits.
