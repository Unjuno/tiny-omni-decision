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
        ↓
Option-only Decision Readout
  (compute only supplied option logits; no full-vocab projection at runtime)
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
                    ↓
          runtime profiling / mobile benchmark
                    ↓
          conditional attention acceleration research
          (only if attention is a measured bottleneck)
```

The final runtime is intended to use a task-adapted ternary base plus one Recovery LoRA. The Decision LoRA is a teacher/training artifact and is merged before ternary conversion rather than stacked as a second runtime adapter.

The runtime should also avoid computing the full vocabulary projection when only supplied option logits are required. The first implementation target is an option-only projection that reuses the existing LM-head rows for the active option labels so that decision semantics remain unchanged while unnecessary vocabulary work is removed.

Architecture-level attention replacement (for example linear attention or another softmax-attention alternative) is tracked as a separate acceleration path. It may be prototyped after Teacher v1 is stable, but production adoption is gated by measured end-to-end profiling and by preservation of multimodal decision quality.

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

Project-level hard gates:
- [x] pin exact base-model revision/hash
- [x] record upstream license/notice requirements
- [x] freeze and validate redistribution-safe train/validation/evaluation catalogs
- [ ] verify architecture-compatible ternary implementation/runtime
- [x] define disjoint held-out train/selection/evaluation splits and zero-overlap checks

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

## Phase 2 — Dataset integration — complete

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
and cross-split duplicates. Candidate catalog validation requires ALLOW rights, exact
manifest/catalog split agreement, and disjoint train/evaluation splits for a shared pinned
dataset revision. The durable teacher corpus is frozen with selected media materialized and
hashed; its source splits and zero-overlap checks are recorded in
[docs/DURABLE_TEACHER.md](docs/DURABLE_TEACHER.md).

Candidate inventory and verified rights decisions are in `docs/DATASETS.md`; separate
training and evaluation candidate catalogs are in `manifests/`. Safe minimum candidates now
cover text (Typed Decisions Synth/Open-Jev), image (Clevr-4), audio (Speech Commands), and
video (CLEVRER). Sampled normalization across these sources is checked for train/evaluation
disjointness; Speech Commands' adapter smoke uses tiny source-shaped fixtures, while its
upstream Parquet files are checksum-pinned. CLEVRER questions metadata was sampled from the
official train/validation files; video bytes were not downloaded. Clevr-4's nominal 10k
archive contains 10,531 annotation rows (8,424 train / 2,107 val); each image produces four
ten-class decisions, and media bytes are not copied during normalization.

Phase 2 exit criteria are complete: at least one rights-audited candidate per modality,
separate pinned train/evaluation catalogs, streaming adapters, reproducible normalization,
and automated source/content split checks. OneJev component-level rights review remains
open but OneJev is excluded from the frozen corpus. Ternary runtime compatibility remains
a later compression gate.

## Phase 3 — High-precision Decision/Omni LoRA — v0 complete, Teacher v1 in progress

Teacher v0 completed a 256-step four-modality run on the local RTX 3080 Laptop GPU.
Its 2,917-record evaluation has already been observed and is a legacy reference,
not a blind test or a model-selection target. Historical corpus provenance and
metrics remain in [docs/DURABLE_TEACHER.md](docs/DURABLE_TEACHER.md). The earlier
16-step pipeline test remains documented in [docs/PHASE3.md](docs/PHASE3.md).

Teacher v1 is the active high-precision quality phase. It first fixes CLEVRER
scene-level grouping and fail-closed independent validation, then freezes fresh
train/validation/sealed-audit splits with record, source-asset, media, and
normalized-content checks. Iteration loads train and validation only. The local
RTX 3080 is measured before extending the rank-16 learning curve; policies for
text/image/video emphasis are compared on validation. Architecture changes and
sealed audit are gated on those results. See [docs/TEACHER_V1.md](docs/TEACHER_V1.md).

The Teacher v1 phase does not merge the final Teacher, ternary-quantize, train a
Recovery adapter, or create a teacher-logit cache. Those remain later gates.

## Phase 4 — Larger multimodal decision adaptation

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

## Phase 5.5 — Option-only Decision Readout

Make the runtime decision path compute only the logits that are actually used.

Required first implementation:
- reuse the existing embedding/LM-head rows corresponding to the supplied option-label tokens
- compute only the active 2–20 option logits at the decision position
- avoid a full-vocabulary projection during decision inference
- preserve the exact option ordering and probability semantics
- verify numerical equivalence against the current full-vocabulary readout to normal floating-point tolerance

Measure:
- end-to-end latency by modality
- isolated readout latency
- peak VRAM/RAM
- output-logit/probability differences
- prediction agreement

This is a low-risk runtime optimization and should be implemented before the compression/recovery stack is finalized. It can be prototyped in parallel once the Teacher v1 decision interface is stable; it must not interfere with Teacher-quality model selection.

Do not replace the readout with a newly trained fixed-class classifier by default. If a learned compact scorer is later tested, treat it as a separate architecture experiment and compare it against the option-only LM-head projection.

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

## Phase 9 — Runtime profiling and mobile prototype

Target outputs:
- portable model package
- reproducible conversion
- PC inference
- Apple Silicon / Android / iOS feasibility notes
- measured end-to-end latency by modality
- startup time and resident RAM/VRAM
- isolated encoder / decoder / attention / MLP / readout timing where practical
- direct packed low-bit kernels only if required after v0.1

The product KPI is the final lightweight Omni Decision model's absolute quality, size, memory footprint, startup, and latency. Teacher fidelity is a diagnostic, not the product objective.

## Phase 10 — Conditional architecture-level acceleration

Investigate deeper attention changes only when profiling shows that attention is a material end-to-end bottleneck.

Candidate experiments:
- linear attention
- other softmax-attention approximations/replacements
- limited attention-layer replacement rather than an all-at-once rewrite

Requirements:
- start from the completed high-quality multimodal decision model rather than redesigning the backbone prematurely
- compare against the same text/image/audio/video decision benchmark
- measure absolute Accuracy, NLL, Brier, ECE, latency, RAM/VRAM, and model size
- preserve multimodal alignment; do not accept a speedup that materially breaks image/audio/video quality
- use adaptation/distillation if needed, but account for the resulting parameter and runtime cost
- adopt only if the end-to-end speedup is meaningful after encoder, MLP, readout, and I/O costs are included

This research may be prototyped in parallel after Teacher v1 is stable, but it is not a blocker for the v0.1 ternary + Recovery product path.

## v0.1 non-goals

- full-parameter fine-tuning
- reinforcement learning
- preserving long chain-of-thought generation
- quantizing every modality encoder to ternary
- custom mobile ternary kernels before model quality is proven
- making linear-attention replacement a prerequisite for v0.1
- large quantizer comparison ladders
- claiming speedup from lower bit-width or attention changes without an end-to-end benchmark
