# Roadmap

## Goal

Build a compact multimodal decision model that accepts text, image, audio, and video and returns calibrated probabilities over typed options.

The v0.1 pipeline is:

```text
Gemma 4 E2B QAT alignment backbone
        ↓
Decision / Omni LoRA training
        ↓
freeze validation-selected Pre-compression Decision Master
(final 95%-per-modality product gate does not block compression entry)
        ↓
merge Decision LoRA into task-adapted quantization source
        ↓
decoder-focused ternary compression
        ↓
quantization damage measurement
        ↓
Recovery from the frozen Master + labeled supervision
        ↓
Variant A: recovered ternary Decision model
        ↓
lightweight Decision classifier / option scorer
        ↓
Variant B: lightweight Decision Core
        ↓
temporal Video specialization
(clips → compact states/scores → lightweight temporal aggregation)
        ↓
all-modal replay + final Recovery
        ↓
Variant C: lightweight Omni Decision
        ↓
profile A / B / C
        ↓
only if profiling justifies it:
token pooling / learned resampler + replacement Recovery
        ↓
Variant D
        ↓
runtime profiling / mobile benchmark
        ↓
optional deeper attention acceleration research
```

The architectural rationale and gate definitions are frozen in
[the compression-first video-specialization design](docs/superpowers/specs/2026-10-06-compression-first-video-specialization-design.md).

Each deployable artifact should carry at most one Recovery adapter. The Decision LoRA is a teacher/training artifact and is merged before ternary conversion. If token pooling/resampling is introduced later, train a new replacement Recovery adapter for that pooled student rather than stacking two Recovery adapters at runtime.

The runtime should avoid computing the full vocabulary projection when only supplied option scores are required. After the first recovered ternary baseline is stable, replace the full-vocabulary readout with a lightweight Decision classifier / option scorer. An option-only projection using the existing LM-head rows is the low-risk equivalence baseline; a learned compact scorer may be tested if it gives a better latency/quality trade-off.

Token pooling or a learned fixed-size resampler is a deliberate later acceleration stage, not a prerequisite for the first quantized model. It now comes after temporal Video specialization and the Variant C benchmark. Clip-based Video processing already introduces explicit compute/token budgeting, so generic pooling is added only if profiling shows a remaining worthwhile bottleneck. Any pooled Variant D uses a replacement Recovery adapter rather than stacking Recovery adapters.

Architecture-level attention replacement (for example linear attention or another softmax-attention alternative) remains a separate deeper acceleration path after the A/B/C runtime comparison.

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

## Phase 3 — High-precision Decision/Omni LoRA — complete; quality target unmet

Teacher v0 completed a 256-step four-modality run on the local RTX 3080 Laptop GPU.
Its 2,917-record evaluation has already been observed and is a legacy reference,
not a blind test or a model-selection target. Historical corpus provenance and
metrics remain in [docs/DURABLE_TEACHER.md](docs/DURABLE_TEACHER.md). The earlier
16-step pipeline test remains documented in [docs/PHASE3.md](docs/PHASE3.md).

Teacher v1 fixed CLEVRER scene-level grouping and fail-closed independent
validation, then froze fresh train/validation/sealed-audit splits with record,
source-asset, media, and normalized-content checks. Validation-only selection
compared sampling policies and three rank-16 seeds. Its candidate was frozen
before one full sealed-audit evaluation. Audio reached 94.6% audit Accuracy;
image reached 68.9%, text 71.3%, and video 52.0%, so even the then-used
four-modality 90% development gate remained open. The current final product
target is 95% Accuracy in every modality. Learning curves show a validation plateau/overfit
signal at the 2,048-step budget. Full hashes, per-source metrics, experiments,
and bottleneck analysis are in [docs/TEACHER_V1.md](docs/TEACHER_V1.md) and
[`manifests/teachers/tiny-omni-decision-teacher-v1.json`](manifests/teachers/tiny-omni-decision-teacher-v1.json).

Teacher v1 used only the local RTX 3080 Laptop GPU (about 14.2 GPU-hours across
logged attempts and the audit; cloud cost $0). It did not merge the final
Teacher, ternary-quantize, train a Recovery adapter, or create a teacher-logit
cache.

The compression-entry policy has since changed. The project no longer requires
all four modalities to reach the final product threshold before ternary work can start. The active
high-precision run should complete under its frozen rules, then its
validation-selected best checkpoint becomes the **Pre-compression Decision
Master** once reload/provenance checks pass. Text/Image/Audio/Video quality is
still reported in full, but remaining quality gaps move forward into Recovery
and post-compression specialization rather than blocking the compression
pipeline indefinitely. The final product gate is >=95% Accuracy for **each**
modality, including end-to-end Video.
The follow-up [Video Teacher v2 design](docs/superpowers/specs/2026-10-04-video-teacher-v2-design.md)
was executed on a separate branch and artifact tree; Teacher v1 remains frozen.
Candidates A/B/C, the B-cosine schedule comparison, and the video-native
Candidate E are complete. E improved Video over B-cosine on the exact same
mixed-task validation examples, but remains below the v1 reference on macro
Accuracy, minimum-modality Accuracy, and Video Accuracy. The 12-frame Candidate
D gate was not met, so it was not run. Full learning curves, overlap/hash
evidence, per-modality/source/question-type metrics, and the bottleneck limits
are recorded in [docs/VIDEO_TEACHER_V2.md](docs/VIDEO_TEACHER_V2.md). No sealed
audit data or ternary quantization was used for v2.

The separate 2,048-step E-long attempt is blocked at step 128 by a Windows
safetensors staging-path limit. A shorter staging path and regression test are
committed on `codex/video-teacher-v2-long-budget`, but the failed snapshot did
not contain the state required for exact resume. The partial run is not a
completed learning-curve result; see
[docs/VIDEO_TEACHER_V2_LONG_BUDGET.md](docs/VIDEO_TEACHER_V2_LONG_BUDGET.md).

## Phase 4 — Finalize and freeze the Pre-compression Decision Master

Complete the active clean-development run without changing its frozen
configuration mid-run. Select the best checkpoint only from the declared
development selector and verify reload reproducibility.

Compression-entry requirements:
- run completed its declared budget or predeclared stopping rule
- best checkpoint frozen and reload-verified
- data/config/checkpoint/environment hashes recorded
- train/development integrity checks pass
- no sealed/final audit used for tuning
- per-modality Accuracy, NLL, Brier, and ECE recorded

A 95% per-modality result is **not** required to leave this phase. Remaining
quality gaps are explicit inputs to later Recovery and specialization work.

Keep the current high-precision training policy as historical experiment
configuration rather than redefining it after seeing results:
- modality encoders frozen
- projector frozen unless a separately versioned experiment changes it
- decoder LoRA
- portable Decision readout
- randomized option ordering
- CE + Brier, with optional future teacher-distribution terms only in a
  separately declared experiment

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
- the frozen validation-selected **Pre-compression Decision Master**
- merge the Decision LoRA into a task-adapted checkpoint before ternary conversion
- do not carry a separate Decision LoRA into the final runtime stack
- do not block this phase solely because one or more modalities remain below the final 95% product target;
  preserve their pre-compression metrics as paired baselines

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

Freeze the high-precision Pre-compression Decision Master (the Recovery Teacher)
and the task-adapted ternary base.

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
- KL over Recovery-Teacher vs student **option distributions**

Supervised correction objectives:
- CE on the labeled option
- Brier loss

Recovery is not pure imitation: ground-truth supervision remains active so the
student can correct Master mistakes instead of being forced to reproduce them.

Optional experiment only:
- low-weight full-vocabulary KL at the decision position, when a full-vocabulary cache was deliberately generated

The high-precision Decision Teacher is the reference for Accuracy, ECE, Brier, NLL, and teacher/student option KL.

## Phase 8.5 — Lightweight Decision classifier / readout

Start from the completed **Variant A** recovered ternary model.

Goal:
- remove the full-vocabulary runtime projection from the hot decision path
- emit only the 2–20 supplied option scores
- preserve the same typed-option semantics and calibration interface

Implementation order:
1. benchmark an option-only projection that reuses the existing LM-head rows for the active option labels; this should be numerically equivalent to the current readout apart from normal floating-point tolerance
2. if worthwhile, test a learned compact Decision classifier / option scorer distilled from the high-precision Teacher

Do not conflate this with replacing the transformer's attention mechanism. This phase changes the final Decision readout only.

Save this as **Variant B** and benchmark it independently before introducing token pooling.

Measure:
- Accuracy, NLL, Brier, ECE by modality
- p50 / p95 model latency by modality
- isolated readout latency
- resident RAM / VRAM
- cold-start/startup latency where practical
- artifact size

## Phase 8.6 — Temporal Video specialization

Start only after Variant B is frozen and benchmarked.

Keep the Variant B full-video path as the baseline, then train the Video
capability on the architecture intended for deployment:

```text
raw video
  ↓
fixed clips / sampled temporal windows
  ↓
compact clip representation and/or Decision scores
  ↓
lightweight temporal aggregator
  ↓
final option distribution
```

First compare:
1. the retained full-video Variant B baseline
2. fixed non-overlapping clips plus aggregation
3. overlapping or event-dense clips only if the simpler clip path justifies
   the extra compute

Clip-level Accuracy is diagnostic, not the product target. The gate is
end-to-end Video Decision quality after aggregation.

Use scene-disjoint clean development data. Mix Text/Image/Audio replay into
Video specialization and select replay strength from validation evidence so
Video gains cannot silently erase the Decision Core.

Save the result after all-modal final Recovery as **Variant C**.

Final product targets:
- Text >=95% Accuracy
- Image >=95% Accuracy
- Audio >=95% Accuracy
- end-to-end Video >=95% Accuracy

Also report NLL, Brier, ECE, per-video-task metrics, clip/frame budget,
p50/p95 latency, memory, and modality-ablation evidence.

## Phase 8.7 — All-modal final Recovery

After the temporal Video module is trained, run a final Recovery pass using
the frozen Master signal where applicable plus ground-truth supervision and
non-video replay.

The deployed Variant C must contain one coherent final Recovery state. Do not
stack multiple Recovery adapters at runtime. If the architecture change
requires a replacement adapter, replace the previous Recovery state.

Benchmark Variant C against A and B before adding another compression
mechanism.

## Phase 8.8 — Optional token pooling / learned resampler + replacement Recovery

Attempt this phase only after Variant C profiling shows a material remaining
token/latency bottleneck.

Add token-count reduction for expensive multimodal paths, prioritizing the
measured bottleneck rather than assuming Video first. Prefer a learned
fixed-size resampler or another hardware-friendly mechanism over blind average
pooling.

Evaluate explicit token budgets. Preserve Variant C unchanged for comparison.
After token reduction:
- distill from the frozen Master / relevant teacher signals
- train a **replacement** Recovery adapter
- do not stack Recovery adapters
- retain CE + Brier supervision

Save the result as **Variant D** only if it improves the measured
quality/latency/memory frontier.

## Phase 9 — Runtime profiling and mobile prototype

Benchmark the preserved artifacts under identical conditions:

- **Variant A:** ternary + Recovery
- **Variant B:** Variant A architecture with the lightweight Decision classifier / option scorer
- **Variant C:** Variant B + temporal Video specialization + all-modal final Recovery
- **Variant D:** optional Variant C + pooling/resampler + replacement Recovery

Target outputs:
- portable model packages for A/B/C and D when D exists
- reproducible conversion
- PC inference
- Apple Silicon / Android / iOS feasibility notes
- Accuracy, NLL, Brier, and ECE by modality
- p50 / p95 model latency by modality
- raw-media end-to-end latency where measurable
- startup time and resident RAM/VRAM
- isolated encoder / decoder / attention / MLP / readout timing where practical
- model/artifact size
- direct packed low-bit kernels only if required after the first runtime implementation

Use the same benchmark samples, preprocessing, option ordering, device, warmup, and timing protocol for all variants. Keep intermediate artifacts instead of overwriting them. Video comparisons additionally record clip/window count, sampled-frame count, and end-to-end aggregation latency.

The product KPI is the final lightweight Omni Decision model's absolute quality, size, memory footprint, startup, and latency. Teacher fidelity is a diagnostic, not the product objective. The final quality gate is >=95% Accuracy in Text, Image, Audio, and end-to-end Video; later compression/recovery work should also report retention relative to the high-precision Master.

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

This research comes after the Variant A/B/C comparison (and Variant D when attempted) unless profiling provides a strong reason to move it earlier. It is not a prerequisite for the first ternary + Recovery baseline.

## v0.1 non-goals

- full-parameter fine-tuning
- reinforcement learning
- preserving long chain-of-thought generation
- quantizing every modality encoder to ternary
- custom mobile ternary kernels before model quality is proven
- making linear-attention replacement a prerequisite for v0.1
- large quantizer comparison ladders
- claiming speedup from lower bit-width or attention changes without an end-to-end benchmark
