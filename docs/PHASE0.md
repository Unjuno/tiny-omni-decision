# Phase 0 — Locked Design Decisions

## High-precision teacher target

`google/gemma-4-E2B-it-qat-q4_0-unquantized`

Phase 1 pinned the upstream repository and processor to immutable revision
`6befbaca7398925921802abd1f277b495b78b738`. See
`manifests/base-model.example.yaml` for the API-reported parameter count,
architecture, license metadata, and file identifiers. The model card identifies
Apache-2.0 and links Google DeepMind's Gemma 4 terms; comply with both when
redistributing derivative artifacts. The unquantized QAT checkpoint contains
BF16 weights and is not itself a 4-bit runtime checkpoint.

The project uses the QAT-family checkpoint to build and preserve a high-precision multimodal Decision Teacher. The teacher is not required to be the final deployment architecture.

## Deployment student candidate

EmbeddingGemma 2 is the preferred student candidate for the compact deployment path. Its exact upstream model ID, immutable revision, processor, license metadata, module graph, and supported runtime must be pinned before implementation.

Use the native embedding/pooling/readout path first. A custom classifier is a fallback only if the native path misses the held-out decision-quality gate. For that fallback, inspect the actual module graph before removing or bypassing terminal pooling/readout components or any final attention/projection block; do not assume a cut point from architecture names alone.

## What the base is expected to provide

- aligned text representations
- image understanding
- native audio understanding
- video through the model's supported multimodal path
- a trainable decoder suitable for lightweight decision adaptation

## What the project adds

1. typed decision behavior;
2. calibrated probability outputs;
3. a compact EmbeddingGemma 2 decision-student path, with a classifier fallback only when required;
4. extreme ternary compression of the selected deployment student;
5. a single final probability-recovery adapter;
6. reproducible edge/mobile packaging.

## Precision policy

| Component | v0.1 precision policy |
|---|---|
| Large student backbone/decoder linear weights | ternary candidate |
| Vision/audio encoders | preserve high precision initially |
| Multimodal projector | preserve high precision initially |
| Norms | preserve |
| Pooling/readout or fallback classifier | preserve |
| Recovery adapter | BF16/FP16 trainable |

This is a starting policy, not a claim that every listed component is sensitive.

## Teacher, student, and recovery lifecycle

The Decision LoRA is used to create the high-precision Gemma 4 Decision Teacher. Freeze that Teacher as the external quality reference.

Next, validate an EmbeddingGemma 2 decision student on the same held-out splits. Prefer its native embedding/pooling/readout path. Use the structural classifier fallback only if required by measured decision quality.

Ternary conversion operates on the selected high-precision student, not on the Teacher by default. Quantization damage is measured against the pre-quantization student while the frozen Teacher remains the external decision-quality reference.

After ternary compression, train one Recovery adapter only if needed. Keep the recovery target at the option-score/probability level; full-vocabulary logits remain optional diagnostics rather than a required artifact.

## Hard-gate status

The pinned base/processor, license record, frozen train/selection/evaluation
corpus, file hashes, and pairwise zero-overlap checks are complete for the first
local high-precision Decision Teacher. EmbeddingGemma 2 student integration and architecture-compatible ternary runtime
support remain open and must be verified before compression.

- [x] pin exact upstream model and processor revision
- [x] record upstream license/notice requirements
- [x] record base and corpus file hashes
- [x] freeze non-overlapping train/validation/evaluation manifests
- [ ] pin and inspect the exact EmbeddingGemma 2 student revision
- [ ] verify architecture-compatible ternary runtime support for the selected student

## Local hardware assumption

Development machine:
- dedicated GPU memory: 16 GB
- system RAM: 32 GB

Local work:
- schema/data development
- low-memory smoke training
- quantization tooling
- evaluation subsets

Rented GPU:
- optional future scale-up after the local 16 GB training path and budget are reviewed
- Recovery training when needed

## Research boundary

This is a product-oriented compression project, not a benchmark survey. Comparisons are added only when they answer an engineering decision.
