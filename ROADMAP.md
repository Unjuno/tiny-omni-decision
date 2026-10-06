# Roadmap

## Goal and current contract

Build one compact Omni Decision model returning calibrated probabilities over supplied choices. Final product Accuracy must be Text >=95%, Image >=95%, Audio >=95%, and end-to-end Video >=95%. Model confidence is not Accuracy. Declared fusion capabilities and target-runtime budgets have independent mandatory release checks.

Approved direction B remains: finish the active high-precision run under its existing rules and freeze its validation-selected Master; a 95% Master is not required before a bounded compression experiment. Final quality requirements are not relaxed.

```text
Multimodal base -> Decision adaptation -> frozen Master -> verified LoRA merge
  -> input/quantizer feasibility -> Q0-Q4 comparison
  -> fixed Recovery versus Recovery-to-Re-specialization -> selected Variant A
  -> equivalent option-only readout first -> Variant B
  -> verified simultaneous-media contract
  -> temporal Video + genuine joint all-modal specialization -> Variant C
  -> profiling -> conditional pooling/resampling + replacement adaptation -> Variant D
  -> untouched release audit and measured target-runtime qualification
```

The two-stage policy is an experiment, not a forced winner. Fixed Recovery remains valid if it performs better. Ternary constraints may support Decision specialization, but removing only unwanted generative capabilities is not an established mechanism. Option-only KD already targets decisions rather than the whole language model.

Read the [design spec](docs/superpowers/specs/2026-10-06-compression-first-video-specialization-design.md), [sync plan](docs/superpowers/plans/2026-10-06-decision-respecialization-sync.md), [executable experiment plan](docs/superpowers/plans/2026-10-06-post-quantization-respecialization-experiment.md), and [review addendum](docs/superpowers/plans/2026-10-06-postquant-review-addendum.md). This roadmap incorporates the concurrent planning updates through `b049c28580a3c62ea6bd770b480ff5fc81b60f19`; it does not replace their interfaces or candidate values.

## Phase 0 — Design and provenance

Primary target: `google/gemma-4-E2B-it-qat-q4_0-unquantized`, at the pinned revision in `manifests/base-model.example.yaml`.

Preserve encoder/projector alignment initially, adapt verified decoder matrices and retain the existing variable typed-option contract. Long-form generation preservation is not required. Model/processor revisions, rights/attribution and data integrity remain hard gates.

- [x] pinned base and processor provenance
- [x] historical rights-gated data adapters and split checks
- [ ] tested ternary reference conversion and adaptation
- [ ] tested packed deployment runtime
- [ ] tested declared simultaneous-media capability matrix

## Phases 1-3 — Historical foundation; product-quality gate unmet

The repository established Decision smoke training, frozen-base LoRA, metadata/manifest tests and single-media text/image/audio/video training. These are historical results, not checks rerun by this planning update.

Teacher v0's evaluation is observed legacy evidence. Teacher v1's audit reported Audio 94.6%, Image 68.9%, Text 71.3% and Video 52.0%; its exact metrics and hashes remain unchanged in `docs/TEACHER_V1.md` and the teacher manifest. None of these statements means the current final four-modality 95% requirement has been achieved.

Preserve all underlying reports: `docs/DURABLE_TEACHER.md`, `docs/PHASE3.md`, `docs/DATASETS.md`, `docs/TEACHER_V1.md`, `docs/VIDEO_TEACHER_V2.md`, `docs/VIDEO_TEACHER_V2_LONG_BUDGET.md`, and `docs/TEACHER_QUALITY_NEXT.md`. Historical Video Candidate A/B/C/E names are not the Q experiment arms or product versions below.

The failed staging-path attempt, completed data-coverage run, and later clean-dev-v2 run are separate experiments. Never concatenate their metrics into a learning curve or promote a historical checkpoint as the current best without its run lock. Historical configs, corpora, weights, failed attempts and observed audit results are not rewritten.

## Phase 4 — Freeze Master and experimental contract

Complete the active run without changing its selector, data or schedule. Verify selected-checkpoint reload and LoRA-merge equivalence in a separate artifact; record the complete model/config/code/data/preprocessing identity and per-modality scores. No minimum Accuracy at compression entry.

For the new comparison predeclare train/dev IDs, option order, target paths, initial adapter tensors, quantizer recipe, losses, LR policy, update budget, endpoint/selection rules and resource caps. Keep verified gold labels separate from teacher predictions. No final-audit use in cache creation, calibration or selection.

Initial input contract: Text with no media or exactly one declared image/audio/video reference. Reject other supplied media before the existing processor; do not silently filter them. This strict limited-profile experiment does not establish fusion and does not replace the final shared Omni product.

## Phase 5 — Resources and execution authorization

Local feasibility checks precede full experiments. A planning document is not a GPU launch manifest. Require explicit model/data paths, approved local GPU-hour/VRAM/disk bounds and environment before execution. Stop on a cap rather than silently lowering options, frame count or batch semantics.

No automatic paid GPU/API use. Preserve the historical first-rented-experiment ceiling of US$15 unless explicitly revised. Record device, backend and library versions, clocks/power mode when known, batch/accumulation, tokens/frames, wall time, memory and artifact hashes. Unknown conditions stay unknown.

## Phase 6 — Reference ternary conversion and feasibility

Quantize approved large decoder attention/MLP matrices in a copy of the merged Master. Preserve encoders, projector, norms and sensitive embeddings/readout. The initial recipe is the one in the companion experiment plan: row-wise groups of 128 actual weights, threshold multiplier 0.7, FP32 assignment/scale calculation, strict threshold ties and explicit tail handling. These are candidate choices, not demonstrated optima.

Measure exact zero counts, reconstruction error, code/scale metadata bytes and all unquantized/residual storage. A BF16-dequantized reference is allowed for quality research but does not prove packed speed or nominal 1.58-bit storage. The real deployment format/backend remains a separate gate.

Run tensor/one-layer checks first. The separate 8-update end-to-end smoke must wait until the real runner is implemented; it is not run by the converter alone. Verify finite gradients, frozen-base stability and reload before the full comparison. Do not resume that smoke as the larger experiment.

## Phase 7 — Q0/Q1 paired damage baseline

Q0 is the merged, equivalence-verified frozen Master; Q1 is the same Master after reference ternary conversion. Both are evaluation-only. Use the same development IDs, options, labels and preprocessing. Report Accuracy/NLL/Brier/ECE, divergence, zero/reconstruction statistics and measured reference memory/cost. Final audits remain untouched.

## Phase 8 — Fixed Recovery versus Decision Re-specialization

Keep one current adapter state and a frozen base for the initial experiment. Cache ordered option logits and bind them to all relevant sample, option, media, model and preprocessing hashes. Full-vocabulary KL is disabled. A dense residual is accounted for separately; do not merge it into three-valued weights without re-quantization and reevaluation.

The executable plan fixes the first bounded candidate at 1,024 optimizer updates, with 256 Recovery updates and 768 annealing updates, gradient accumulation 4 and seed 17. Q2 uses KL/CE/Brier coefficients 1.0/0.2/0.2 throughout. Q3/Q4 start there and linearly move to 0.2/1.0/0.2 in stage two. Temperature is 1.0. These values are not an optimum claim, and no run is authorized merely by documenting them.

Sum KL across each example's active options, then average examples; do not divide by option count. Compute stable FP32 losses, ignore padded choices correctly and detach teacher targets. Keep one coefficient set for all microbatches in an optimizer update. Persist optimizer/LR, RNG, sampler and exact stage/update state for resume; no optimizer reset at the phase boundary.

| Arm | Role |
|---|---|
| Q0 `q0_master` | Master reference, evaluation only |
| Q1 `q1_ternary_raw` | Raw ternary reference, evaluation only |
| Q2 `q2_fixed_recovery` | Ternary plus fixed Recovery |
| Q3 `q3_recovery_to_respecialization` | Ternary plus two-stage policy |
| Q4 `q4_high_precision_control` | Frozen non-quantized Master plus the exact Q3 additional-learning policy |

Only Q2/Q3/Q4 train. Match actual initial adapter tensors, topology, data/order, option permutations, optimizer/LR, update budget and evaluation rules; match Q3/Q4 coefficient schedules too. Primary comparisons use the same final update; separately report development-selected checkpoints under the companion plan's frozen rule. Equal updates do not imply equal compute, so report runtime separately.

The initial research adoption rule is a minimum-modality gain of at least 2 percentage points for Q3 over Q2, with no other modality losing more than 1 point; report asset-aware uncertainty and replicate promising comparisons. These are predeclared research tolerances, not final product gates. Noisy results remain UNCERTAIN. Q3 need not beat Q4 to make a useful compact product, and Q3 beating Q4 would not isolate sparsity alone.

**Variant A:** the selected feasible ternary Decision core. Retain fixed Recovery if annealing loses. Failed fixed-base LoRA may motivate a separate residual-initialization or constrained-QAT experiment; it does not automatically authorize a large search or more compute.

## Phase 8.5 — Lightweight readout: Variant B

Start with equivalent existing LM-head rows, preserving 2-62 options, case-sensitive single-token labels, permutations, bias and all output transformations. Test the 60-intent Text task. Equivalence needs no extra learning merely because the projection is smaller. A learned compact scorer needs its own quality/cost comparison.

Readout replacement is not transformer-attention replacement. Freeze and benchmark B before introducing temporal aggregation.

## Phase 8.6 — Fusion contract and temporal Video

Before final Omni specialization, implement all declared simultaneous-media combinations end-to-end. Every supplied reference reaches the model or fails clearly. Test timestamp units and alignment, supported payloads, missing evidence and contradictory evidence. Use legally usable, aligned examples that actually require combined information; do not equate interleaved single-media tasks with fusion.

Use question-conditioned time-stamped clip representations and/or evidence with a lightweight temporal aggregator. Score-only aggregation is a baseline, not assumed sufficient. Preserve event order and supervise video-level labels after aggregation unless valid clip labels exist.

Compare fixed clips against the original full-video task under explicit frame/compute budgets. Overlapping or event-dense clips are conditional later candidates. More clips are not automatically faster. Record decode, sampling, per-clip inference, aggregation and end-to-end latency; a synthetic soundtrack is not observed CLEVRER evidence.

## Phase 8.7 — Final shared all-modal specialization: Variant C

Train the same shared model with Video and Text/Image/Audio replay. Replay prevents forgetting but cannot substitute for relevant labeled data and learning capacity to improve still-weak Text/Image tasks. The teacher is an anchor, not an infallible labeler.

Ablations and counterfactuals require valid labels and matched controls. Removing required evidence may make a question unanswerable; handle that explicitly rather than scoring confident guessing as fusion.

Preserve A/B. Final point Accuracy remains Text >=95%, Image >=95%, Audio >=95%, end-to-end Video >=95%. Additional declared cross-modal release slices cannot be waived because the input path is unsupported.

## Phase 8.8 — Conditional pooling/resampling: Variant D

Attempt only after C profiling identifies a worthwhile remaining token/latency bottleneck. Keep C, use explicit budgets and retrain one replacement adaptation state rather than stacking obsolete adapters. Include resampler/readout/temporal modules in all resource totals. D must meet the same release gates.

## Phase 9 — Release evaluation and deployment

Freeze task/source populations, options, labels, preprocessing, denominators and fusion criteria before final selection. Keep Audio random-negative and validated hard-negative suites separate; report both. Retain old Video task results when adding a new task family. No easier benchmark substitution or post-result deletion of hard items.

Freeze model, calibration and actual runtime before an untouched asset-disjoint audit. Require all four modality Accuracy values at least 95%, with denominators, independent asset counts and uncertainty. A population-level lower-bound guarantee is a stronger separate claim, not implied by a 95% point estimate. Do not repeatedly audit candidates until one passes.

Before any speed/release claim, set target hardware and numeric latency/size/memory/startup budgets. Report batch-one warm p50/p95, cold start, raw-media-to-decision latency and all resident/artifact bytes. For streaming Video separate observation-window delay from processing delay. Dense-reference and packed-backend measurements are different results; revalidate quality after backend changes.

## Phase 10 — Optional attention acceleration

Only measured end-to-end bottlenecks justify deeper attention replacement. Preserve task contracts, multimodal quality and adaptation-cost accounting. This is not required for the initial controlled compression experiment.

## Evidence-preservation and status

Spec/plan/README alignment does not implement YAML parsing, numerical loss/schedule wiring, a quantizer, a runner or fusion. Preserve the legacy fixed-loss Recovery config until the versioned implementation exists. Code/config/tests, historical reports and weights are not changed by this documentation revision.

Do not infer speed from zeros or nominal bits, Accuracy from confidence, or product success from engineering completion. Do not directly modify main, force-push, rewrite historical results or silently alter an in-flight experiment.
