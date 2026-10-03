# Phase 0 — Locked Design Decisions

## Primary target

`google/gemma-4-E2B-it-qat-q4_0-unquantized`

Phase 1 pinned the upstream repository and processor to immutable revision
`6befbaca7398925921802abd1f277b495b78b738`. See
`manifests/base-model.example.yaml` for the API-reported parameter count,
architecture, license metadata, and file identifiers. The model card identifies
Apache-2.0 and links Google DeepMind's Gemma 4 terms; comply with both when
redistributing derivative artifacts. The unquantized QAT checkpoint contains
BF16 weights and is not itself a 4-bit runtime checkpoint.

The project uses the QAT-family checkpoint because the objective is to reuse an already aligned multimodal representation and move quickly toward a low-bit deployment model. The project does not require the base to be a frontier reasoning model.

## What the base is expected to provide

- aligned text representations
- image understanding
- native audio understanding
- video through the model's supported multimodal path
- a trainable decoder suitable for lightweight decision adaptation

## What the project adds

1. typed decision behavior;
2. calibrated probability outputs;
3. extreme decoder-focused ternary compression of the task-adapted model;
4. a single final probability-recovery adapter;
5. reproducible edge/mobile packaging.

## Precision policy

| Component | v0.1 precision policy |
|---|---|
| Large decoder linear weights | ternary candidate |
| Vision/audio encoders | preserve high precision initially |
| Multimodal projector | preserve high precision initially |
| Norms | preserve |
| Decision readout | preserve |
| Recovery LoRA | BF16/FP16 trainable |

This is a starting policy, not a claim that every listed component is sensitive.

## Adapter lifecycle

The Decision LoRA is used to create the high-precision Decision Teacher. Before ternary conversion, its update is merged into the task-adapted quantization source. The final runtime therefore does not need stacked Decision + Recovery adapters.

After ternary compression, train one Recovery LoRA against the high-precision teacher. The default teacher cache stores only option logits/probabilities at the decision position. Full-vocabulary logits are optional diagnostics, not a required training artifact.

## Hard gates

Before durable training:

- [x] pin exact upstream revision
- [x] record upstream license/notice requirements
- pin processor/tokenizer revision
- record file hashes where practical
- verify architecture-compatible ternary runtime support
- freeze non-overlapping train/eval manifests

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
- durable decision training
- Recovery training when needed

## Research boundary

This is a product-oriented compression project, not a benchmark survey. Comparisons are added only when they answer an engineering decision.
