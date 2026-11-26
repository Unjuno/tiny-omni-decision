# ViLT answer coverage and conditioned feature path

Executed after the pinned runtime audit. Source `a5d3a33`; no training or quality
evaluation. Original model/data files unchanged; no validation image or sealed
audit used. Checkpoint revision/hashes are inherited from the runtime audit.

## Closed-head coverage gate

Exact string membership in the3129-answer head covers31 of40 existing taxonomy
option slots. These options are missing:

- color: cyan
- texture: emojis, zigzag, chessboard
- shape: cube, sphere, torus, teapot, gear
- count: all1–10 covered

No synonyms were fitted, no options removed, no examples dropped, no unsupported
option probabilities fabricated. This rejects the unchanged closed head as a
direct scorer for the existing full Clevr-4 option contract. It does not establish
what concepts the underlying joint encoder represents.

## Conditioned encoder path

The first two training images in the pinned8424-image manifest were each paired
with four fixed questions (exact strings in companion JSON). No labels were
loaded. The whole-image published processor produced four384×512 images per
forward, with padded text length9; individual questions have8/9/8/8 tokens.
No crop or truncation. This differs from the DINO224×224 resize and would be
an overall model/preprocessing comparison, not an isolated fusion ablation.

Frozen `model.vilt(...).pooler_output` returns finite `[2,4,768]` FP32 values.
The saved array reloads exactly. Feature bytes/file hash and both media hashes
are recorded. Standard classifier is loaded but not executed in this path.

- Loaded parameters117,588,537; feature encoder111,595,008.
- GPU forward0.299407s first image/four questions,0.029576s second.
- Peak allocated VRAM531,295,232bytes; local cost$0.
- Projected8424train+512validation cache:109,805,568 raw FP32 bytes.
- Projection from one warm batch:264.3s forward time, excluding decode/I/O and
  throughput variation. **Full extraction has not been run.**

This is a question-dependent representation: a new question requires another
joint forward. It cannot be advertised as one reusable image cache for arbitrary
future questions. Four fixed taxonomy features also do not establish general
instruction following, video reasoning, or longer state conditioning.

## Next bounded comparison

Compare a frozen ViLT conditioned representation plus the existing147,712-parameter
bilinear candidate head and frozen MiniLM candidate embeddings against the prior
DINO result. Keep all8424/512 images, all four10-option taxonomies, seed17,
8epochs, effective batch64, AdamW1e-3/weight-decay1e-4 and earliest minimum
validation NLL. Freeze questions before extraction, preserve all unsupported
closed-head options in the new candidate head, and report per-taxonomy metrics,
rotated-image control, parameters and real extraction cost. No synonym search.
New output paths and a separate execution smoke are required.

This candidate is **not yet trained or evaluated**. Its image/text head stack
would use135,126,073 parameters with the unused VQA head still resident;
adding Whisper-tiny would total172,886,713, without an established video path.
Do not claim an omni product from this arithmetic. No additional VideoPrism
tower fits the200M limit alongside this stack.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r1/vilt-feature-audit-01`.
All95% modality requirements remain unmet/unproven.
