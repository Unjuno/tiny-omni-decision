# Native conditional candidate: frozen Path A integration

## Intent and authorization

Implement the next step in `NEXT_STRATEGY_20261016.md`: one explicit candidate
that accepts real text/image/audio/video and supplied conditions/candidates,
within200M total prequantization parameters. User authorizes autonomous bounded
local implementation and experiments (ROADMAP product contract2026-10-12).
This architectural integration uses the existing worktree and preserves every
legacy runtime/artifact. It does not promote rejected semantic endpoints.

## Decision

Retain the pinned VideoPrism114.365M (one shared image/video tree), original
DeBERTa70.831M, Whisper **encoder only**8.208M and EfficientAT1.961M. Add a
192-wide query-conditioned attention readout. This reuses original pretrained
features; its new connection weights start at seed17 and are **untrained**.
The architecture and random initialization are a functional research baseline,
not a replacement for any qualified existing task model.

Alternatives: full Whisper with this language/visual allocation exceeds200M;
replacing the language model now would mix language quality and audio changes.
A new speech-specific CTC decoder may be studied later if native supervision
shows the shared readout cannot retain speech information. Do not begin it now.

## Interfaces

`NativeTokenBlock(kind, values, positions, valid)` carries tokens [T,D], explicit
normalized time/row/column coordinates [T,3], and bool validity [T]. Kinds:
text384, image768, video768, speech384, event_audio384. Audio/image/video do not
require a text observation. The condition and candidate descriptions are text;
DeBERTa encodes each pair without truncation, producing [K,384] masked means.

Each native observation may be encoded once and reused for new queries. Raw
tokens remain separate from readout projection, so training cannot cache stale
projected features. Every token block is checked for shape/dtype/device/finiteness;
invalid tokens are removed before projection. Empty/all-invalid observations
fail. At most8192 total tokens and2..40 candidates. No silent truncation.

One projection per kind, learned modality vectors and a3->192 coordinate
projection produce memory. A192-wide four-head cross-attention reads memory with
each candidate-condition vector. A shared scalar MLP scores each candidate using
query, attended value, product and absolute difference. Candidate softmax T1 is
serialized as JSON with user IDs. IDs never enter the model. No class-vocabulary
restriction or generated text. Probabilities are **uncalibrated** until measured.

Temporal coordinates retain order information; this does not demonstrate temporal
reasoning. Different sources retain their own timelines; no cross-clock temporal
relation or synchronization is inferred from this integration. Joint quality and
alignment remain separate gates. Position metadata must come from preprocessing,
never labels or correct-option identities.

## Actual integration smoke, fixed before execution

Use first existing TRAIN Clevr-4 image, the already-verified TRAIN Lombard
`s11_l_bbab2n` audio/video assets, and a literal text observation. Source files and
model/preprocessor/tokenizer/source revisions are hashed. Do not load validation,
reserved calibration, transfer payloads or sealed audit. No download or training.
Retain existing8-frame VideoPrism profile and17-cell image profile; speech retains
valid20ms tokens; EfficientAT supplies one complete-clip feature.

Run the same initialized readout on each modality, two fixed question forms,
original/reversed candidate order, repeated observation reuse, and a separately
loaded saved readout. Record raw inputs/features/queries/scores/probabilities and
hashes, exact live parameters, stored buffers, FP32 tensor bytes, model revisions,
time/RAM/VRAM. Execute a fresh-process replay with the same pinned inputs and
saved readout. CPU unit tests exercise multi-input blocks, padding, order,
time-coordinate effects, gradient reach, malformed inputs and checkpoint reload.

PASS means functional wiring, <=200M actual resident parameters, finite scores,
frozen weight hashes, stable option association and replay tolerance<=1e-6. It
does **not** mean useful Accuracy or R2 completion. No accuracy gate can pass on
this unlabeled smoke. Keep the head/frozen weights independently identifiable.
Use separate new output directories; no existing file overwrite. Local CPU/GPU
only, no dependency upgrade. Stop on nonfinite values, hash mismatch, resource
failure. Do not tune initialization, rendering or hyperparameters on smoke scores.

## Next required work

After integration, freeze one native training/validation task contract covering
conditional text, image attributes/count/relations, speech/environmental audio,
video order/state, and genuinely joint decisions. Use independent source groups,
hard negatives, modality-removal/mismatch/time controls and retention. Supervise
only the new readout initially. Native task quality, OOD, all-task95–98%, package,
latency, calibration and final audit remain unmet until measured.
