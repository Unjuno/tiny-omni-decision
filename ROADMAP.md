# Roadmap

## Goal

Build a compact multimodal decision model that accepts text, image, audio, and video and returns calibrated probabilities over typed options.

The v0.1 pipeline is:

```text
Gemma 4 E2B QAT alignment backbone
        ↓
Decision / Omni LoRA training
        ↓
High-precision Decision Teacher
        ├── cache option logits at the decision position
        └── merge Decision LoRA into task-adapted quantization source
                    ↓
          Decoder-focused ternary compression
                    ↓
          Quantization damage measurement
                    ↓
          single Recovery LoRA
                    ↓
          Final Tiny Omni Decision model
```

The final runtime is intended to use a task-adapted ternary base plus one Recovery LoRA. The Decision LoRA is a teacher/training artifact and is merged before ternary conversion rather than stacked as a second runtime adapter.

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
- [x] pin exact base-model revision/hash
- [x] record upstream license/notice requirements
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
- [x] confirm CI is green on GitHub (PR #1 Actions run #9 passed install, lint, tests, and manifest validation)

Exit criterion: a fresh clone can validate manifests and run CPU-only CI, and the pinned model completes the documented 16 GB GPU smoke with base frozen and adapter trainable. Local CPU checks, loaded checkpoint inspection, the 16 GB GPU smoke, and GitHub Actions on PR #1 all pass. Phase 1 implementation gates are complete; dataset licensing, held-out evaluation splits, and ternary runtime compatibility remain gates for later phases.

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

**Phase 2 implementation status:** schema, fail-closed source and component license policy,
pinned candidate manifests, text/image/audio/video metadata adapters, deterministic option
shuffling, streaming normalization, and source/content split checks are implemented. CPU
fixtures exercise unknown and non-commercial licenses, media metadata, malformed targets,
and cross-split duplicates. No dataset training or full-media download was run.

Candidate inventory and verified rights decisions are in `docs/DATASETS.md`; separate
training and evaluation candidate catalogs are in `manifests/`. Small normalized samples
from Typed Decisions Synth, Open-Jev train/test, MVBench, and Clevr-4's official train/val
annotation splits passed the disjointness check. Clevr-4's nominal 10k archive contains
10,531 annotation rows (8,424 train / 2,107 val); each image produces four ten-class
decisions, and media bytes are not copied during normalization.
Open gates before durable training remain full-corpus split/output verification, further
OneJev component rights review, and ternary runtime compatibility. Phase 2 GitHub Actions
validation is pending.

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

Quantization source:
- the completed high-precision Decision Teacher
- merge the Decision LoRA into a task-adapted checkpoint before ternary conversion
- do not carry a separate Decision LoRA into the final runtime stack

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

## Phase 8 — Teacher option-logit cache and Recovery adapter

Freeze the high-precision Decision Teacher and the task-adapted ternary base.

For each training decision, cache only the teacher signal needed for the decision task at the single readout position.

Canonical cache record:

```text
sample_id
option_token_ids
teacher_option_logits
teacher_option_probabilities
target
teacher_temperature / normalization metadata
```

Default storage policy:
- raw FP16/BF16 **option logits** are the canonical teacher signal
- option probabilities may be stored as a derived convenience field
- full-vocabulary logits are **not required** and are disabled by default
- optional full-vocabulary capture is reserved for diagnostics or experiments that explicitly test whether preserving non-option language distribution helps

This keeps the cache small and aligned with the product objective: calibrated decisions over supplied options.

Recovery training uses a **single final LoRA** on the frozen task-adapted ternary base. It is not stacked on top of an unmerged Decision LoRA.

Primary objective:
- KL over teacher vs student **option distributions**

Auxiliary objectives:
- CE on the labeled option
- Brier loss

Optional experiment only:
- low-weight full-vocabulary KL at the decision position, when a full-vocabulary cache was deliberately generated

The high-precision Decision Teacher is the reference for Accuracy, ECE, Brier, NLL, and teacher/student option KL.

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
