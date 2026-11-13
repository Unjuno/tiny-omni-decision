# Native M1 and synchronized M2 interface smoke (2026-11-02)

## Result

Added CPU-only contract coverage for the existing text-free cache-to-Decision
interfaces. The single-modality runner is in
`src/tiny_omni_decision/native_feature_decision.py`; the synchronized fusion
runner is in the same module and uses `gather_aligned_features` plus
`FusionDecisionScorer`. The change here closes a test-coverage gap; it does not
introduce a new model architecture or fit any weights.

`tests/test_native_fusion_smoke_matrix.py` passed all 8 tests. Five parameterized
M1 cases route hash-pinned synthetic cached features for audio, image, screen,
video and sensor through typed numeric options and return normalized
option-ID probabilities. A fresh Python subprocess imports the text-free
scorer while `transformers`, `sentence_transformers` and the text runtime are
blocked.

The synchronized synthetic M2 fixture checks timestamp-matched audio/video
cache gathering, missing-stream abstention before inference, stale-stream
abstention, shuffled audio feature positions, an explicitly offset audio stream,
conflicting feature values reaching the fusion scorer, and option permutation
invariance after mapping probabilities back by opaque option ID. Cache names
are neutral (`payload-01`, `stream-a`, `stream-b`), options contain numeric
attributes, and the fixtures do not load target or label content.

## Scope limits

These are interface/mechanics tests, not a learned quality evaluation. The
synthetic features have no real sensor/video/audio semantics, the random
readout is not trained or calibrated, and the contradiction case only verifies
that both streams reach the model. It does not establish that a model resolves
contradictions correctly. No real dataset, model checkpoint, feature archive,
validation set or sealed audit was opened. No GPU or cloud compute was used.

R2 M-0 interface smoke is now covered for these paths. R2 M-1 remains open for
task-admitted, measured text-free modality Decision quality. R2 M-2 remains
open for a rights-approved, independently grouped real task requiring multiple
inputs, matched unimodal baselines, calibrated quality and fixed intervention
results. This smoke does not admit a source or justify training.

## Verification

- `pytest -q tests/test_native_fusion_smoke_matrix.py`: 8 passed.
- `ruff check tests/test_native_fusion_smoke_matrix.py`: passed.
- GPU status was read-only; Orbitune PID 2128 remained live, so no GPU work was
  started.
