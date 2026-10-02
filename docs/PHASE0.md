# Phase 0 — Locked Design Decisions

## Primary target

`google/gemma-4-E2B-it-qat-q4_0-unquantized`

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
3. extreme decoder-focused ternary compression;
4. a separate probability-recovery adapter;
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

## Hard gates

Before durable training:

- pin exact upstream revision
- record upstream license/notice requirements
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
