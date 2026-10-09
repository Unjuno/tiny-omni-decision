# Roadmap

## Goal

Build a compact multimodal decision model that accepts text, image, audio, and video and returns calibrated probabilities over typed options.

The next experiment is **ternary first → Teacher recovery → LoRA only if recovery is insufficient**:

```text
Gemma 4 E2B → Decision / Omni LoRA → frozen high-precision Decision Teacher
                                                    │
                                                    │ option supervision
EmbeddingGemma 2 pretrained checkpoint               │
        ↓                                           │
Aggressive ternary conversion                       │
        ↓                                           │
Unrecovered damage measurement                      │
        ↓                                           ▼
Ternary-constrained recovery with the Teacher (QAT + distillation)
        ↓
Meets validation quality and deployment requirements?
        ├── yes → packed export / reload / final evaluation / runtime
        └── no  → freeze the quantized base
                    → add one Recovery LoRA
                    → train that LoRA with the same Teacher
                    → export / reload / re-evaluate
```

Gemma 4 supplies recovery supervision and remains the external quality reference. EmbeddingGemma 2 is the student for this experiment; successful recovery is a hypothesis, not a measured result. Do not require a separately trained high-precision decision student before ternary conversion.

This sequence supersedes the previous native-student-training → classifier fallback → quantization plan. It does not change the active Gemma 4 Teacher run, datasets, seeds, checkpoints, or historical results. Direct Gemma 4 compression is not the next experiment, but its existing artifacts are retained.

The project does not require the Teacher and student to share an architecture and does not depend on preserving long-form generation.

## Phase 0 — Freeze the design

**Status: design remains frozen; local student implementation and QAT/Recovery/Phase 9 experiments are recorded below as non-promoted research results.**

Teacher target:
- `google/gemma-4-E2B-it-qat-q4_0-unquantized`

Student target:
- EmbeddingGemma 2; pin the exact upstream model/processor revision and license metadata before implementation

Design decisions:
- Keep the high-precision Gemma 4 Decision Teacher fixed during student recovery.
- Initialize the student from pretrained EmbeddingGemma 2 and impose ternary constraints before task recovery training.
- Use Teacher option distributions with the existing labeled CE and Brier objectives. No second teacher or new representation-loss stack is required.
- Retain the native backbone/pooling structure for the first experiment. Establish only the minimal readout needed to score supplied options; an embedding vector is not itself an option-probability distribution.
- Target large text/backbone, embedding, vision, and audio weight tensors for ternary conversion. Record all higher-precision exceptions and their bytes rather than silently excluding whole encoders from the size claim.
- First recover by updating the student under ternary forward constraints. Only if that is insufficient, freeze the quantized base and train one separate Recovery LoRA against the same Teacher.
- Pooling plus a small classifier, with any justified terminal-path removal, remains a deferred structural fallback after recovery and LoRA are insufficient. No attention-block removal is assumed in advance.

Project-level hard gates:
- [x] pin exact Teacher base-model revision/hash
- [x] record Teacher upstream license/notice requirements
- [x] freeze and validate redistribution-safe train/validation/evaluation catalogs
- [x] define disjoint held-out train/selection/evaluation splits and zero-overlap checks
- [x] pin and inspect the exact EmbeddingGemma 2 model/processor revision and rights
- [x] verify student option readout and numerical forward/backward behavior
- [x] verify architecture-compatible ternary conversion, packed export, and the supported Transformers reload path after BF16 dequantization

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

### Teacher handoff

Finish the current Teacher run under its existing configuration, select the best checkpoint using validation, verify reload, then freeze the Master Teacher and its provenance. This is a handoff requirement, not a claim that the current local run has completed. Do not restart or alter the active Teacher run for this plan change. Keep different experiment series separate.

## Phase 6 — EmbeddingGemma 2: ternary first

Start from the pretrained EmbeddingGemma 2 checkpoint, not from a separately task-trained high-precision student.

The dedicated `codex/ternary-student-recovery` branch pins and hash-verifies
the complete model and processor, loads the BF16 checkpoint with an isolated
Transformers 5.19.0 environment, and records the instantiated module/tensor
inventory. CPU smokes completed synthetic text forward, supplied-option
readout, CE+Brier backward, and finite projection gradients, plus generated
image/audio/one-frame-video inputs through the actual model with finite
768-dimensional outputs. These path checks ran CPU-only and are not a quality baseline. The actual
loaded checkpoint has a fail-closed candidate target inventory. A CPU
real-checkpoint smoke converts all 483 candidate tensors, exports five-trit
codes and FP32 group scales, reloads the overlay onto a fresh pinned model, and
reproduces the ternary-QAT synthetic embeddings exactly. The overlay contains
160,577,615 tensor-file bytes; high-precision parameter exceptions are 483,648
bytes and runtime buffers are 5,258 bytes. The standard runtime dequantizes to
BF16 and does not provide packed compute or reduced inference memory. The
unquantized diagnostic and unrecovered ternary overlay have now both been
evaluated on the same hashed 256-example validation subset and exact ID/option
order using the RTX 3080 Laptop GPU. No sealed audit data has been read.
Conversion and paired initial-damage results are in
[docs/EMBEDDINGGEMMA2_STUDENT.md](docs/EMBEDDINGGEMMA2_STUDENT.md).

Before conversion:
- pin the exact model/processor revision, inspect the actual graph, and record the execution backend and supported dtypes
- retain the native backbone/pooling path and define the minimal option-score readout, with stable option identity/order for supervision
- record an unquantized baseline using that same input/readout setup; this is a diagnostic, not a high-precision student-training stage
- record numeric quality and deployment acceptance limits and a bounded recovery budget before selecting results; use validation for decisions, not the final evaluation split

Convert the intended large weight tensors directly to scaled ternary values: negative scale, zero, or positive scale. Include text/backbone, embeddings, and vision/audio weights in the target inventory. Inspect actual tensor paths rather than reusing Gemma 4 module names. List every excluded tensor, dtype, and byte cost; any unsupported or unstable component is an explicit exception, not evidence that the whole model is ternary.

The approximately 1.58-bit goal concerns densely packed ternary weight codes. It is not a guaranteed whole-model size, memory footprint, or performance result. Count scales, packing/alignment, metadata, higher-precision exceptions, the decision readout, and any later LoRA in the exported package. A 2-bit container is not 1.58-bit storage, and a fake-quantized floating-point tensor is not a packed low-bit artifact.

Fail closed on numerical or shape errors. Low initial decision quality is expected to be measured, not used to reject the student before attempting recovery. Verify the intended export/runtime path; do not assume that a ternary format or kernel supports this architecture. Training shadow weights and optimizer memory must be accounted for separately from deployment memory.

## Phase 7 — Measure initial damage — validation comparison complete

Record the frozen Teacher, unquantized student diagnostic, and unrecovered ternary student on the same fixed validation examples and option ordering. Isolate quantization damage using the same student input/readout setup before and after conversion; Teacher-to-student differences also include architecture and task-adaptation differences.

The fixed 256-example comparison is complete. Unquantized student accuracy was
44.53% and unrecovered ternary accuracy was 19.14%; same-example option
order/target checks passed. This is initial-damage evidence on a development
validation subset, not a trained-student result or sealed evaluation.
Teacher-supervised recovery remains the next gate; detailed per-modality and
per-source metrics, hashes, and runtime limits are recorded in
[docs/EMBEDDINGGEMMA2_STUDENT.md](docs/EMBEDDINGGEMMA2_STUDENT.md).

Retain the existing metrics:
- Accuracy, NLL, Brier, and ECE, overall and by modality
- top-1 and probability-rank preservation; option-distribution KL / JS
- actual model bytes, latency, and peak RAM / VRAM with hardware, backend, dtype, input sizes, and batch recorded

Proceed to recovery after a numerically valid damaged baseline. This phase does not require the unrecovered student to match the Teacher.

## Phase 8 — Teacher recovery, then LoRA if needed — local QAT and Recovery experiments complete

The previously active QAT run is no longer in progress. QAT attempt 08 supplied
the selected ternary overlay for a separate Recovery LoRA experiment; neither
the QAT nor Teacher artifacts were overwritten. The historical attempt 05
record remains available at
[docs/EMBEDDINGGEMMA2_QAT_ATTEMPT05.md](docs/EMBEDDINGGEMMA2_QAT_ATTEMPT05.md).

### Shared supervision

Freeze the Gemma 4 Decision Teacher. Reuse the existing training/validation/evaluation separation. Training uses only training examples and their Teacher signals; validation selects checkpoints and the fallback decision. Keep final evaluation out of training, calibration fitting, and model selection.

Use a compact option-level cache, not a full-vocabulary cache:

```text
sample_id
option_ids / option_order
teacher_option_token_ids (Teacher-side metadata only)
teacher_option_logits / teacher_option_probabilities
target
teacher_temperature / normalization metadata
```

Teacher and student must refer to the same actual choices in the same order. Teacher token IDs are not student vocabulary IDs. Keep loss coefficients and temperature fixed in the recorded experiment configuration.

Primary objective: Teacher-to-student KL over option distributions. Auxiliary objectives: labeled cross entropy and Brier loss. No additional embedding-geometry objective or teacher is required for this experiment.

### 8A — Recover under ternary constraints

Train the quantized student with the frozen Teacher through quantization-aware distillation. Higher-precision shadow weights may receive gradient updates, but every forward uses their ternary-quantized values for the target tensors. Export those tensors as ternary; do not remove the constraint during training and merely quantize again at the end.

This stage updates the student under the constraint; it is not the old frozen-base LoRA-only recovery plan. Use a numerically supported activation/accumulation dtype, with BF16/FP32 as the reference choices. Lower-bit weights do not require lower-bit activations. Verify save/reload and exported predictions before accepting the recovered student.

If the agreed validation quality and deployment limits are met, proceed to Phase 9 without adding LoRA.

### 8B — Recovery LoRA only if 8A is insufficient

Freeze the selected quantized student base, including its ternary codes and scales. Attach one small higher-precision Recovery LoRA to verified compatible modules and train the adapter with the same frozen Teacher and option-level objectives. Do not change the Teacher or stack multiple recovery adapters.

One local Recovery LoRA run has completed on the selected QAT attempt 08
overlay. It consumed 4,859 unique examples in one pass; the selected checkpoint
reached 25.78% macro accuracy on the fixed 256-example validation set, with
Audio 12.50%, Image 4.69%, Text 54.69%, and Video 31.25%. This is not a
product promotion and does not establish a general capacity ceiling. See
[docs/EMBEDDINGGEMMA2_TERNARY_RECOVERY_V0.md](docs/EMBEDDINGGEMMA2_TERNARY_RECOVERY_V0.md)
for hashes, full metrics, and limitations.

The deployed artifact is the quantized base plus that LoRA. Count adapter bytes and runtime overhead. Keep the adapter separate by default: merging its update into the base generally breaks the ternary weight constraint. Any later merge/requantization requires a new quality check and is not assumed in this plan.

Reload the exported base and adapter together and repeat the same validation checks. Recovery is not guaranteed; failure within the recorded budget remains a failed experiment rather than a reason to claim success or silently increase precision.

### Deferred structural fallback

If ternary-constrained recovery and the LoRA fallback are still insufficient, retain the earlier pooling-plus-small-classifier idea as a separate structural fallback, not a mandatory pre-quantization stage. Inspect the graph before replacing a terminal readout or removing a terminal attention/projection block. Do not equate an attention block with the output head or assume either can be removed without loss. No such surgery is part of the first experiment.

## Phase 9 — Export, final evaluation, and runtime

Freeze the selected recovered student, package its packed ternary weights and any required readout/Recovery LoRA, then reload in the intended runtime. Verify predictions against the training-time quantized path.

Run the final paired evaluation on the same frozen held-out samples for the Teacher, unquantized student diagnostic, unrecovered ternary student, and selected recovered artifact. Report per-modality quality together with actual package bytes and measured latency/RAM/VRAM. Do not claim pure whole-model 1.58-bit storage when higher-precision exceptions or LoRA remain.

Target outputs:
- portable model package and reproducible conversion
- PC inference
- Apple Silicon / Android / iOS feasibility notes
- direct packed low-bit kernels only if required after v0.1

Local Phase 9 export/reload and paired evaluation are complete for the current
research candidate. The hash-verified package is 1.707 GB including the pinned
base checkpoint, and reload exactly reproduced the selected 256-example
validation predictions. On the same previously observed 2,917-example
evaluation, the Recovery candidate scored 31.98% overall and 23.97% macro
accuracy, versus 73.43% and 65.24% for Teacher v0. It is not a product
promotion and this historical evaluation is not blind. A bounded, opt-in
one-scene video frame cache achieved bitwise identical processor tensors and
reduced a three-question scene preprocessing microbenchmark from 3.42 s to
1.17 s; full paired-evaluation speedup is not yet measured. Detailed metrics,
hashes, and limitations are in
[docs/EMBEDDINGGEMMA2_TERNARY_RECOVERY_V0.md](docs/EMBEDDINGGEMMA2_TERNARY_RECOVERY_V0.md).

A no-training component ablation has now compared all-ternary, Audio-path
BF16, Vision-path BF16, and shared-decoder BF16 on the same frozen 256-example
validation snapshot. Restoring any one component did not improve all four
modalities; the results point to a mixed shared-decoder/modality interaction,
not a demonstrated tower-only failure. Keep the quantized research artifact
unchanged. A follow-up early/middle/late decoder-block BF16 sensitivity scan
also failed the predefined clear-improvement rule, so stop layer-level search.
Results and hashes are in
[docs/EMBEDDINGGEMMA2_TERNARY_COMPONENT_ABLATION.md](docs/EMBEDDINGGEMMA2_TERNARY_COMPONENT_ABLATION.md)
and
[docs/EMBEDDINGGEMMA2_TERNARY_DECODER_DEPTH_ABLATION.md](docs/EMBEDDINGGEMMA2_TERNARY_DECODER_DEPTH_ABLATION.md).

A separate frozen-backbone Decision Head readout diagnostic was completed on the same fixed 256-example validation set. The ternary head improved macro NLL/Brier but failed the predefined per-modality Accuracy guard; the unquantized head was worse on macro Accuracy, NLL, and Brier. Neither is adopted. See [docs/EMBEDDINGGEMMA2_DECISION_HEAD.md](docs/EMBEDDINGGEMMA2_DECISION_HEAD.md).

## Scope and non-goals

- This remains a staged student experiment, not a product promotion. Existing Teacher training inputs and historical artifacts remain unchanged; old Gemma 4 quantization/recovery configs are not EmbeddingGemma 2 implementations.
- No separate unconstrained high-precision student fine-tuning stage before this ternary experiment.
- No reinforcement learning or long chain-of-thought preservation.
- No mandatory pooling/classifier surgery, extra teacher, or broad quantizer comparison ladder.
- No custom mobile ternary kernels before model quality is proven.
- No speedup claim from a bit-width or storage label without a kernel/runtime benchmark.
