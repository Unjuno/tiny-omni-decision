# Shared spatial image comparison: fixed protocol

One new frozen representation: VideoPrism spatial intermediate output on single
RGB image frames, fixed8-image batches on CPU. Same pinned checkpoint114365184
parameters, source and existing resize-short-side288/center-crop preprocessing.
Keep16x16 spatial grid, average4x4 blocks to16 cells; prepend global spatial
mean as a summary (not a CLS token). Float32 cache17x768/image. No old cache
parity claimed, since the earlier single/eight-frame parity gate failed.

All8424 existing training images, same512 selector images, same IDs/order and
media hashes. No1595 reporting images or sealed payloads. Require disjoint
image IDs and SHA256 across train/validation, preserve row manifests. ~467MB
float32 cache; fresh output only. No encoder fitting, no augmentation.

Then compare global-versus-spatial representations with matched bilinear
scorers: observation13056 (summary768 +16cells768/4), candidate384 from pinned
MiniLM, interaction width128. Global control repeats cell mean in all16 slots.
Each scorer1,720,576 parameters; image stack133,475,584 with VideoPrism+MiniLM.
These heads are experimental alternatives, not both deployed. This differs
from DINO in encoder, preprocessing and feature width; not a pure backbone
causal comparison. DINO-only is smaller; shared VideoPrism potentially removes
the separate DINO tower when a VideoPrism video path is already resident.

Two fresh seed17 readouts,24epochs each, same initialization and question order,
batch64 questions, AdamW lr0.001/wd0.0001, earliest minimum validation NLL.
Keep best/final weights, all learning curves, per-taxonomy metrics and matched/
rotated-image predictions. Check gradient finiteness and checkpoint reload.
No extra seeds, thresholds, preprocessing variants or epoch extensions.
Use local GPU for the readout only if free; first run a separate small smoke.

Primary comparison: new spatial versus its global control. Compare full512-image
results by aligned IDs against saved DINO and pooled-VideoPrism references,
report paired-image uncertainty and all taxonomy scores. No test-set tuning.
This is reused development validation, not proof of broad image understanding,
native no-text operation or all-modality95%. Do not adopt based only on size or
latency; shared complete Omni runtime remains a separate gate.
