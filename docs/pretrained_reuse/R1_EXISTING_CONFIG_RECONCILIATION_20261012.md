# Existing checkpoint/config reconciliation after T5 audit

Five actual local checkpoints loaded on CPU with their original task classes,
FP32, local-files-only, frozen. No training/inference, network model download,
environment upgrade or existing artifact modification. This checks load/config
integrity and parameter accounting, not current quality or all possible model fields.

| Model | Loaded parameters | Prior count matches | Missing/unexpected keys |
|---|---:|---|---|
| dinov2 | 22,056,576 | yes | none |
| minilm | 17,389,824 | yes | none |
| crossencoder | 22,713,601 | yes | none |
| smollm2 | 134,515,008 | yes | none |
| whisper | 37,760,640 | yes | none |

Compared explicitly published structural/activation/dropout/tie/position fields
before and after actual loading. DINO, MiniLM, Cross-Encoder and Whisper match.
SmolLM2 moves top-level rope_theta100000 into the effective rope_scaling mapping
with rope_type default; the numerical base100000 is preserved. This is schema
migration, not evidence of changed rotary frequency. No tensor load warnings.

The T5 meta-construction mismatch is therefore not reproduced in these five
actual existing checkpoint loads. Do not discard earlier quality results on that
basis. This does not establish compatibility for every architecture, VideoPrism
(separate JAX path), or ViLT (already documented legacy-buffer handling).

## Consequence for the next quality experiment

Existing text/video failures are not explained by a demonstrated wrong parameter
count or missing weight in these paths. Retain the pinned environments and models;
focus further experiments on conditional reasoning/data/representation, not
unmotivated library upgrades or claiming that a larger text model fits200M.

Measured sequential model load/count time: 0.346s; no GPU work, cloud$0.
Exact paths, config/weight hashes, loader diagnostics and versions are in JSON.
Artifact: `C:/CodexArtifacts/pretrained-reuse-r1/existing-config-reconciliation-01`.
Four-modality95–98% remains unmet. This audit provides no new quality score.
