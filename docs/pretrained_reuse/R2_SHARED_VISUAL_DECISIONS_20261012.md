# Shared raw visual worker to decision JSON

## Executed integration

One VideoPrism model/tree in a JAX CPU worker now accepts a bounded typed batch
of pinned raw-media requests. The PyTorch host retains the existing image affine
head, MiniLM and video token-attention head. It receives checksummed FP32 tensors
and emits deterministic custom-option JSON without retraining or reloading the
visual encoder between image/video profiles.

Executed sequence: eight original cropped CLEVR-4 validation PNGs, one fixed
CLEVRER development clip (scene10005), then the same eight images. Image decisions
use the fixed lambda0.01 research head; video decisions use the original attention
checkpoint and first eight existing questions from that scene.

| Request | Distinct decisions | Max probability difference | Answer changes |
|---|---:|---:|---:|
| images | 32 | 1.7763568394e-15 | 0 |
| video | 8 | 4.17232513428e-07 | 0 |
| images-repeat | 32 | 1.7763568394e-15 | 0 |

The final image batch repeats the same32 decisions; do not count it as new
quality evidence. Image features repeat exactly. Image cached features and video
pooled-feature reference pass 1e-5 absolute gates. Probability tolerance was
fixed at1e-4 with zero permitted answer changes. Reversing supplied option order
preserves the selected custom ID. Full request/feature/prediction hashes are saved.

## Representation correction

The existing default video profile returns8 spatially averaged temporal vectors.
The attention readout requires2048 space-time tokens. An explicit video-only
`representation="tokens"` path now preserves those tokens and their order.
The old default behavior remains covered by tests. Supplying tokens mode for
images fails closed. A typed request contract bounds image batches to8 assets
and video batches to1 clip; every local asset must match its SHA-256.

The shared request contract uses standard-library dataclasses, because the
existing JAX environment has no Pydantic. Both environments execute the same
validation code without installing or upgrading packages.

## Measured resource scope

This functional slice contains 132,210,472 parameters before quantization:
VideoPrism114,365,184, MiniLM17,389,824, affine image61,480 and video393,984.
This is a subset of the prior195.42M resident collection: speech, event audio,
sensor and fusion heads were not wired by this replay. Do not use this smaller
number to claim the complete Omni model is deployed.

| Request | Preprocess + encode + pipe send seconds |
|---|---:|
| images | 1.304415 |
| video | 1.754679 |
| images-repeat | 1.000784 |

Visual worker peak working set: 1,083,531,264bytes.
This excludes host peak memory. Timings exclude model loading/host scoring and
can benefit from persistent compilation caches; not cold-start or end-to-end
latency claims. CPU only, no GPU training or paid compute.

## Limitations and next gate

This is saved-output parity, not new accuracy or multimodal-fusion evidence.
Video quality remains at the previously reported weak baseline. Binary video
choices are conditioned on text; image outputs are fixed ten-class taxonomies.
Custom IDs do not establish arbitrary semantic option handling. The worker uses
whole-clip uniform8-frame sampling, so it is not a causal online observation path.
No screen, speech, event-audio, sensor or joint decision is included in this run.
The next integration should preserve actual preprocessing and observation timing
when connecting other existing encoders/heads; size feasibility cannot substitute
for the required per-modality and joint quality gates.

## Reproduction

Command: `python -m scripts.replay_shared_visual_decisions`.
Artifacts: `C:\CodexArtifacts\pretrained-reuse-r2\shared-visual-decision-replay-01`. Run source `0cfb0a3f77c37c3610e1327bd7980980e1855a36`; subsequent runner change
was formatting only. Source commit and recorded SHA identify the executed file.
All raw assets, heads and encoder weights were read-only; no audit access or
checkpoint selection. Parent/worker exit cleanly after the bounded request batch.
Adjacent JSON contains exact decisions, provenance and worker telemetry.

## Verification

Ruff passed; 587 CPU tests passed, 1 skipped. Model/dataset/catalog validation
passed. Target-commit GitHub Actions is checked after push.
