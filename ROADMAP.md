# Roadmap

## Goal

Build a compact multimodal decision model that accepts text, image, audio, and video and returns calibrated probabilities over typed options.

The v0.1 pipeline is:

```text
Gemma 4 E2B QAT alignment backbone
        ↓
Decision adaptation / probability distillation
        ↓
Strong low-bit decision model
        ↓
Decoder-focused ternary compression
        ↓
Calibration-preserving Recovery LoRA
        ↓
Mobile / edge package
```

The project does not depend on preserving long-form generation.

## Phase 0 — Freeze the design

**Status: implementation-ready with hard gates.**

Primary target:
- `google/gemma-4-E2B-it-qat-q4_0-unquantized`

Design decisions:
- E2B is used for pretrained multimodal alignment.
- Text/image/audio/video decision quality matters more than long-context reasoning.
- Decision tuning may use labels and/or soft probability targets from stronger decision teachers.
- No custom decision head is required by default; option-token/readout approaches remain preferred when they preserve runtime portability.
- Ternary compression focuses on large decoder linear weights first.
- Modality encoders, projectors, norms, and readout components remain higher precision until proven safe.

Hard gates before durable training:
- [ ] pin exact base-model revision/hash
- [ ] record upstream license/notice requirements
- [ ] freeze redistribution-safe training data manifests
- [ ] verify architecture-compatible ternary implementation/runtime
- [ ] define held-out paired benchmark splits

## Phase 1 — Reproducible text decision LoRA smoke

- [x] Python package skeleton, config layout, and validation CLI
- [x] immutable upstream model and processor revision pinned in manifest
- [x] license/attribution metadata recorded; weights excluded from Git
- [x] CPU tests for schema, option labels, reordering, Brier, probabilities, and manifest
- [x] local preflight command reports optional ML versions and CUDA device VRAM
- [x] actual-model inspection CLI (requires download and optional ML dependencies)
- [x] config-only architecture inspection verifies decoder attention paths separately from modality encoders
- [x] vocabulary-logit text decision path (single-token labels validated at runtime)
- [x] synthetic CE + Brier LoRA smoke CLI with save/reload and metadata
- [x] load pinned weights and revalidate architecture module/LoRA targets against checkpoint tensors
- [x] pass forward/backward/save/reload smoke on RTX 3080 Laptop GPU (16 GB VRAM)
- [ ] confirm CI is green on GitHub

Exit criterion: a fresh clone can validate manifests and run CPU-only CI, and the pinned model completes the documented 16 GB GPU smoke with base frozen and adapter trainable. Local CPU checks, loaded checkpoint inspection, and the GPU smoke pass. The GitHub Actions run for the PR is queued; Phase 1 remains open until remote CI is green.

## Phase 2 — Dataset integration

Normalize public decision data into one schema.

Initial sources to investigate and license-filter:
- typed-decision / Open-Jev style text data
- OneJev-style multimodal decision data
- audio multiple-choice / classification data
- video multiple-choice / classification data

Required per sample/source metadata:
- source repository/dataset
- revision
- split
- modality
- license
- media reference
- contamination/eval status

Do not commit redistributable media blindly.

## Phase 3 — Local text decision experiment

Local machine target: 16 GB dedicated VRAM / 32 GB RAM.

Goals:
- load the E2B QAT family in a memory-efficient training mode
- run 100–1000 text decision samples
- forward → loss → backward → checkpoint → reload
- confirm option-order shuffling
- confirm finite CE/Brier losses

No rented GPU time is spent until this passes.

## Phase 4 — Multimodal decision adaptation

Add modalities in this order:
1. text
2. image
3. audio
4. video

Initial policy:
- freeze modality encoders
- freeze projector unless evidence requires adaptation
- train decoder LoRA
- use a portable decision readout
- randomize option ordering

Initial objective:
- cross entropy
- Brier loss
- optional probability-distillation KL
- optional small coherence penalty

## Phase 5 — Decision training on rented GPU

Use rented GPU only after local smoke tests pass.

Budget target: **≤ US$15 total** for the first complete experiment.

Record:
- GPU model
- hourly price at run time
- wall time
- processed samples/tokens
- checkpoint hashes
- exact environment

Evaluate:
- Accuracy
- ECE
- Brier
- NLL
- latency by modality

## Phase 6 — Extreme ternary compression

First target:
- decoder attention linear weights
- decoder MLP linear weights

Preserve initially:
- image/audio encoders
- multimodal projector
- norms
- embeddings/readout when sensitivity requires it
- decision-specific small components

Representation target:
- group-wise ternary values `{-s, 0, +s}`
- measure actual whole-model BPW and bytes; never infer them from a format name

Do not assume `TQ1_0` or another runtime format supports the architecture until verified.

## Phase 7 — Quantization damage measurement

Paired evaluation on exactly the same held-out samples:

- decision reference
- ternary base

Metrics:
- top-1 preservation
- Accuracy
- ECE
- Brier
- NLL
- KL / JS to reference probabilities
- probability rank preservation
- latency
- peak RAM / VRAM

## Phase 8 — Recovery adapter

Freeze the ternary base.

Train a separate Recovery LoRA against compact teacher signals:

```text
sample_id
options
teacher_probabilities
```

Primary objective:
- KL(reference || ternary + recovery)

Optional:
- CE
- Brier

No full-vocabulary logit cache is required.

## Phase 9 — Runtime and mobile prototype

Target outputs:
- portable model package
- reproducible conversion
- PC inference
- Apple Silicon / Android / iOS feasibility notes
- direct packed low-bit kernels only if required after v0.1

## v0.1 non-goals

- full-parameter fine-tuning
- reinforcement learning
- preserving long chain-of-thought generation
- quantizing every modality encoder to ternary
- custom mobile ternary kernels before model quality is proven
- large quantizer comparison ladders
- claiming speedup from lower bit-width without a kernel benchmark
