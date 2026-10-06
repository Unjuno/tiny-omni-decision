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
EmbeddingGemma 2 student candidate
        ├── use the native embedding/readout path first
        └── fallback: replace the terminal pooling/readout path with a small classifier
                    ↓
          Decoder/backbone-focused ternary compression
                    ↓
          Quantization damage measurement
                    ↓
          single Recovery adapter if needed
                    ↓
          Final Tiny Omni Decision model
```

Gemma 4 remains the high-precision teacher/reference. EmbeddingGemma 2 is the preferred deployment/compression student candidate once it passes the same held-out decision-quality gates. The exact EmbeddingGemma 2 upstream model ID, revision, architecture cut point, and runtime support must be pinned and inspected before implementation.

The project does not require the teacher and final deployment model to share the same architecture.

The project does not depend on preserving long-form generation.

## Phase 0 — Freeze the design

**Status: implementation-ready with hard gates.**

Primary target:
- `google/gemma-4-E2B-it-qat-q4_0-unquantized`

Design decisions:
- E2B is used for the high-precision multimodal Decision Teacher.
- Text/image/audio/video decision quality matters more than long-context reasoning.
- Decision tuning may use labels and/or soft probability targets from stronger decision teachers.
- EmbeddingGemma 2 is the preferred student candidate for the deployment/compression path, subject to a pinned revision and paired held-out validation.
- Use the student's native embedding/readout path first; do not add a custom classifier unless the native path misses the decision-quality gate.
- Structural fallback: after inspecting the actual model graph, remove or bypass the terminal pooling/readout path (and only any terminal attention/projection block proven unnecessary) and attach a small classifier. Do not guess the cut boundary.
- Ternary compression focuses on the selected student's large linear weights first.
- Modality encoders, projectors, norms, pooling/readout, and classifier components remain higher precision until proven safe.

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

## Phase 3 — High-precision Decision/Omni LoRA — durable local teacher complete

The local RTX 3080 Laptop GPU completed a 256-step four-modality Decision LoRA run
on the frozen 4,859-record train corpus. The best validation checkpoint was
reloaded and evaluated on all 2,917 held-out examples. Corpus provenance, zero-overlap
checks, selection and final metrics, artifacts, and the merge/export instructions are
recorded in [docs/DURABLE_TEACHER.md](docs/DURABLE_TEACHER.md). The earlier 16-step
pipeline test remains documented in [docs/PHASE3.md](docs/PHASE3.md). PyTorch 2.6 is
required for Gemma 4's multimodal attention masking path.

The Decision LoRA remains an adapter artifact. No merge/export, ternary quantization,
Recovery adapter, or teacher-logit cache was produced in this phase. The merge/export
path is documented but has not yet been exercised.

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

## Phase 6 — Student gate and extreme ternary compression

### 6A — EmbeddingGemma 2 student gate

Keep the completed Gemma 4 Decision Teacher frozen as the quality reference.

Before ternary conversion:
- pin the exact EmbeddingGemma 2 model/processor revision and license metadata
- inspect the actual module graph and runtime requirements
- evaluate the native EmbeddingGemma 2 path on the same held-out decision splits
- adapt/distill only as much as required to make it a viable decision student
- compare Accuracy, ECE, Brier, NLL, and modality-specific results against the frozen Teacher

Preferred path:
- keep the native embedding/pooling/readout structure and use the smallest decision readout that satisfies the gate

Structural fallback only if the preferred path is insufficient:
- inspect the real terminal module graph
- remove or bypass the terminal pooling/readout path
- if a final attention/projection block is demonstrably unnecessary for the decision task, remove it as part of the same measured ablation
- attach a small classifier and re-run the same held-out evaluation
- do not assume module names or a cut point before inspection

Exit criterion: select one high-precision EmbeddingGemma 2 decision student as the pre-quantization reference. If neither native nor structural-fallback variants pass the quality gate, retain the existing Gemma 4 path rather than forcing the student substitution.

### 6B — Extreme ternary compression

Quantization source:
- the selected high-precision EmbeddingGemma 2 decision student
- the frozen Gemma 4 Decision Teacher remains the external quality reference

First target:
- large student backbone/decoder linear weights identified by actual-model inspection

Preserve initially:
- modality encoders when sensitivity requires it
- multimodal projectors
- norms
- pooling/readout or classifier
- other small decision-specific components

Representation target:
- group-wise ternary values `{-s, 0, +s}`
- measure actual whole-model BPW and bytes; never infer them from a format name

Do not assume `TQ1_0` or another runtime format supports EmbeddingGemma 2 until verified.

## Phase 7 — Quantization damage measurement

Paired evaluation on exactly the same held-out samples:

- selected high-precision EmbeddingGemma 2 decision student
- ternary student
- frozen Gemma 4 Decision Teacher as an external quality reference

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

## Phase 8 — Compact reference cache and Recovery adapter

Freeze the selected high-precision student and the task-adapted ternary student. Keep the Gemma 4 Decision Teacher frozen as the external quality reference.

For each training decision, cache only the reference signal needed for the decision task. Reuse the existing compact option-logit/probability cache format when the selected student exposes compatible option logits; otherwise store the smallest equivalent option-level score/probability signal required for recovery.

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

Quantization damage is measured against the selected high-precision student. The frozen Gemma 4 Decision Teacher remains the external reference for final Accuracy, ECE, Brier, NLL, and decision-quality comparisons.

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
