# R2 AVATAR frozen-feature runtime check — 2026-11-02

## Scope

This work adds an inference wrapper around the already selected 73,857-parameter
AVATAR concatenation readout. It checks reload integrity and the supplied-option
JSON contract on fixed TRAIN examples. It does not change training, the selected
checkpoint, feature caches, source corpus, validation selector, or any sealed
audit. It is not a new quality evaluation.

The task is the narrow corrected AVATAR category-match adaptation: an audio
query and four candidate video clips. The scorer consumes one 384-D audio vector
and one 768-D vector per candidate. Candidate strings are carried through to
the response schema but are not model inputs; this readout does not understand
natural-language questions or arbitrary option semantics.

## Runtime and checks

Added `AvatarAVDecisionRuntime` in
`src/tiny_omni_decision/avatar_fusion_runtime.py`. It applies the train-derived
normalization, loads the MLP state strictly after checking checkpoint and
normalization SHA-256 values, freezes all parameters, scores each supplied
option, and returns `selected_option_id`, confidence, and normalized
probabilities in the shared Decision JSON schema. It abstains when either
required modality is absent. The verification CLI is
`scripts/verify_avatar_fusion_runtime.py`; it requires expected hashes for all
four files and refuses to overwrite an existing output directory.

The readout has 73,857 parameters. The pinned encoders associated with its
feature cache have 1,961,239 EfficientAT parameters and 114,365,184 VideoPrism
parameters, for 116,400,280 parameters across this AV route before
quantization. This count excludes software/runtime overhead and does not imply
that quantized deployment size or latency has been measured.

The CPU verification used a separate process and these fixed inputs:

- 8 examples selected from the corrected AVATAR TRAIN split;
- 8 distinct query-audio source groups and 32 distinct candidate-video source
  groups;
- only the previously frozen feature cache, with exact cache hashes checked;
- the selected joint checkpoint and train-derived normalization, both
  hash-checked;
- no validation or sealed-audit examples or labels.

The result is in the local artifact directory
`C:\CodexArtifacts\pretrained-reuse-r2\avatar-runtime-integration-20261102-attempt-01\integration-cli-01`.
Selected input IDs, feature vectors, and manifests remain outside Git. The
integration report records the hashes.

| Check | Result |
|---|---:|
| Hash-checked fresh-process load | PASS |
| Readout parameters | 73,857 |
| Supplied-option JSON probabilities finite and sum to 1 | PASS |
| Option reordering maximum probability delta | `1.1921e-7` |
| Missing audio abstains | PASS |
| Missing video abstains | PASS |
| Fixed TRAIN examples scored | 8 |
| TRAIN examples predicted correctly | 8/8 (not a quality metric) |
| Runtime inference on CPU, including reordering checks | 0.0165 s |
| Encoder calls during this wrapper check | 0 |

The cached vectors were previously extracted from raw media by the pinned
EfficientAT and VideoPrism pipelines; this check begins at those encoder
outputs and does not freshly decode or re-encode raw media. Consequently it
verifies the feature-to-probability runtime boundary, not the complete raw-media
deployment package. The 8/8 TRAIN result is expected to be optimistic and must
not be used to claim accuracy, calibration, or generalization.

## Interpretation and remaining work

This is a small executable route showing how a compact learned readout can
produce normalized probability JSON over supplied choices. It does not
establish calibrated probabilities: no calibration quality was evaluated here.
The AVATAR validation result recorded in the separate task report remains the
only development-quality evidence for this narrow task; it is not re-run here.
The readout is not a four-modality or general-purpose Decision Teacher and does
not yet accept text or image evidence.

Next work should admit a real, non-leaking task where the model must combine
multiple pretrained representations and the input question/options affect the
answer. Then validate a complete raw-media-to-JSON path on permitted TRAIN
examples, freeze a task battery and untouched audit before any quality tuning,
and measure end-to-end footprint/latency on the intended device. No model
promotion, quantization, encoder fine-tuning, or sealed evaluation occurred in
this check.

## Verification

`python -m ruff check src/tiny_omni_decision/avatar_fusion_runtime.py scripts/verify_avatar_fusion_runtime.py tests/test_avatar_fusion_runtime.py`

`python -m pytest tests/test_avatar_fusion_runtime.py -q` — 5 passed.
