# LM-to-Decision conversion with ternary bottleneck and post-compression specialization

## Status

Approved architectural direction: **B**, refined around the actual product transformation:

> The project is not preserving a language model at smaller size. It is converting a pretrained multimodal generative model into a compact Decision model.

The ternary stage is therefore not treated only as compression damage. It is also an intentional capacity bottleneck that may remove redundant generative freedom before the model is re-optimized for typed decisions.

The final product quality gate remains strict:

- Text Accuracy >= 95%
- Image Accuracy >= 95%
- Audio Accuracy >= 95%
- end-to-end Video Accuracy >= 95%

The 95% target is a shipping gate, not a prerequisite for entering ternary conversion.

## Product transformation

The target transformation is:

```text
pretrained multimodal generative LM
        ↓
high-precision Decision adaptation
        ↓
Pre-compression Decision Master
        ↓
hard ternary bottleneck
        ↓
Decision Recovery
        ↓
Decision Re-specialization
        ↓
lightweight Decision readout
        ↓
temporal Video specialization
        ↓
final all-modal Decision specialization
        ↓
compact Omni Decision model
```

The project does not optimize for preserving long-form generation, free-form token distribution, or general generative behavior unless those capabilities are shown to be necessary for Decision quality.

## Current evidence and why compression is not blocked

The latest completed tracked data-coverage run selected step 1,792 with:

| Modality | Accuracy |
|---|---:|
| Audio | 1.0000 |
| Image | 0.8359 |
| Text | 0.8125 |
| Video | 0.5156 |

Text, Image, and Audio improved substantially relative to the frozen Teacher v1 reference on the same development subset. Video remained near 50%, and prior Video Teacher v2 experiments did not establish that more frames, broader decoder LoRA targeting, or more rank would move the high-precision model toward the final 95% Video target.

This is sufficient evidence to stop treating a 95% high-precision Master as a prerequisite for the rest of the project.

It is **not** evidence that 50% Video is an acceptable product result.

## Terminology

Use the following terms consistently.

- **Pre-compression Decision Master**: the validation-selected high-precision E2B Decision model before ternary conversion.
- **Recovery Teacher**: the frozen Pre-compression Decision Master when it supplies option-distribution targets after quantization.
- **Ternary Student**: the quantized E2B Decision model during post-quantization training.
- **Decision Recovery**: the first post-quantization stage that restores useful behavior lost to ternary conversion.
- **Decision Re-specialization**: the second post-quantization stage that reduces dependence on the Recovery Teacher and optimizes the constrained model directly for the Decision objective.
- **External Teacher**: an optional stronger multimodal model used for ability distillation. It is separate from the Recovery Teacher.

## Quality gates

### Compression-entry gate

Ternary conversion may begin when:

1. the active high-precision run completes its declared budget or stopping rule;
2. a validation-selected best checkpoint is frozen;
3. checkpoint reload reproduces its selected predictions and metrics;
4. train/development integrity checks pass;
5. no sealed/final audit was used for tuning;
6. the Master artifact records config, corpus hashes, checkpoint hash, environment, and per-modality Accuracy/NLL/Brier/ECE.

The final 95% product threshold does not block compression entry.

### Final product gate

The final deployable model must achieve all of:

- Text Accuracy >= 95%
- Image Accuracy >= 95%
- Audio Accuracy >= 95%
- end-to-end Video Accuracy >= 95%

The final benchmark also reports:

- NLL
- Brier
- ECE
- p50/p95 latency
- startup latency
- resident memory
- artifact size
- whole-model effective bits per weight
- ternary zero rate and nonzero rate
- Recovery/re-specialization parameter and compute overhead
- cross-modal and modality-ablation checks where supported by the runtime input contract

Audio must be reported on both the historical random-distractor baseline and a separate hard-negative evaluation.

## Revised architecture flow

```text
High-precision E2B Decision adaptation
        ↓
freeze Pre-compression Decision Master
        ↓
merge Decision LoRA
        ↓
hard decoder-focused ternary conversion
        ↓
measure quantization damage + ternary sparsity
        ↓
Stage 1: Decision Recovery
  teacher-heavy anchor + labeled supervision
        ↓
Stage 2: Decision Re-specialization
  reduce teacher pressure
  optimize Decision labels/calibration directly
        ↓
Variant A: ternary Decision Core
        ↓
lightweight Decision classifier / option scorer
        ↓
Variant B
        ↓
Video Temporal Specialization
  video → clips/windows → compact states/scores
        → lightweight temporal aggregator
        ↓
all-modal replay + final Decision Re-specialization
        ↓
Variant C
        ↓
profile quality / latency / memory
        ↓
only if profiling justifies it:
token pooling / learned resampler
        ↓
replacement post-change specialization
        ↓
Variant D
```

## Ternary bottleneck

### Role

Ternary conversion has two purposes:

1. reduce storage and enable a low-bit execution path;
2. constrain the model's weight-space freedom before Decision-specific re-optimization.

The second purpose is a hypothesis to test, not a guaranteed benefit.

### Initial quantization scope

Start with large decoder linear weights:

- attention projections;
- MLP projections.

Preserve higher precision initially for:

- modality encoders;
- multimodal projector;
- norms;
- embeddings/readout where sensitivity requires it;
- small Decision-specific components.

Use group-wise ternary values:

```text
{-s, 0, +s}
```

and record the actual zero fraction per tensor/group and for the whole quantized target set.

Do not infer sparsity, effective BPW, or runtime speed from the format name.

### Sparse-base invariant

The ternary base remains frozen during the first Recovery/Re-specialization experiment.

A dense low-rank adapter may be used initially, but it must be accounted for separately:

- adapter parameter count;
- adapter bytes;
- adapter FLOPs / latency contribution;
- base ternary zero rate;
- effective runtime memory.

Do **not** claim that the complete model is sparse merely because the frozen base contains zeros.

Do not merge a dense Recovery adapter into ternary weights for deployment unless the merged model is explicitly re-quantized and re-evaluated.

## Two-stage post-quantization training

### Stage 1 — Decision Recovery

Purpose: recover useful Decision behavior damaged by ternary conversion without forcing the student to reproduce every generative property of the Master.

Conceptual loss:

```text
L_recovery =
  lambda_KD * KL(Master option distribution || Student option distribution)
+ lambda_GT * CE(label, Student)
+ lambda_B  * Brier(label, Student)
```

At the beginning of this stage, the Master distribution is a relatively strong anchor.

Only option distributions at the Decision position are required by default. Full-vocabulary KL is not part of the default objective.

### Stage 2 — Decision Re-specialization

Purpose: allow the constrained ternary model to move away from Master mistakes and specialize for the final product objective.

During this stage:

- decrease teacher/KD pressure;
- increase the relative importance of labeled Decision supervision;
- retain calibration supervision;
- retain multimodal replay so gains in one modality do not silently destroy another.

The exact schedule is an experiment parameter, but the direction is fixed:

```text
teacher influence: higher → lower
ground-truth influence: lower/moderate → higher
```

The Recovery Teacher is therefore an **anchor**, not a performance ceiling.

A fixed teacher-heavy objective and an annealed Recovery→Re-specialization objective must be compared directly before the annealed policy becomes default.

## Required causal comparison

The first post-quantization experiment must preserve enough controls to separate three effects:

1. ordinary extra Decision training;
2. ternary compression damage/recovery;
3. possible regularization or capacity-allocation benefit from the ternary bottleneck.

Minimum comparison:

- **A — Master**: frozen high-precision Pre-compression Decision Master.
- **B — Ternary raw**: the same Master after ternary conversion, no post-training.
- **C — Ternary + fixed Recovery**: teacher-heavy post-training.
- **D — Ternary + Recovery→Re-specialization**: teacher influence annealed down and Decision supervision strengthened.
- **E — High-precision control**: non-quantized Master given the same additional Decision data/update budget as D, without ternary conversion.

Interpretation:

- C > B measures recoverability.
- D > C supports the value of Decision Re-specialization over pure recovery.
- D > A shows that the constrained student can exceed its Master on the tested Decision task.
- D > E is stronger evidence that the ternary bottleneck itself contributed beyond ordinary additional training.

No single seed or repeatedly tuned development subset is sufficient to claim a general regularization benefit.

## Escalation policy

Do not start with full ternary-aware training complexity.

### First attempt

```text
post-training ternary conversion
+
frozen ternary core
+
trainable Recovery/Re-specialization adapter
```

### Escalate only if needed

If quality cannot be recovered or improved with the first approach, test a separately versioned ternary-aware experiment, such as:

- trainable group scales;
- trainable ternary thresholds;
- straight-through-estimator ternary-aware fine-tuning;
- constrained/quantized Recovery modules.

Do not mix these changes into the first experiment because doing so makes the cause of any improvement ambiguous.

## Lightweight Decision readout

After Variant A is stable, remove unnecessary full-vocabulary work from the hot Decision path.

First test an option-only projection that reuses the equivalent existing output rows and preserves the same option scores within numerical tolerance.

Only if the equivalence path is not sufficient should a learned compact classifier/scorer be introduced.

The readout must preserve the repository's current variable option-count contract rather than assuming a fixed 20-class output.

## Video Temporal Specialization

### Purpose

Do not force the high-precision Master to solve the final long-video product problem before compression.

Train temporal Video capability on the constrained architecture intended for deployment.

### Path

```text
raw video
  ↓
clips / temporal windows
  ↓
compact time-stamped representations and/or Decision evidence
  ↓
lightweight temporal aggregator
  ↓
final video-level option distribution
```

The aggregator must be able to preserve temporal order when the task requires it.

Video-level labels supervise the final aggregated output by default. Do not assign the video-level label independently to every clip unless a task explicitly provides valid clip-level supervision.

### Baselines

Compare:

1. retained full-video Variant B;
2. fixed non-overlapping clips plus aggregation;
3. overlapping/event-dense clips only if the simpler path justifies the extra compute.

Record total decoded frames, clip count, sequential/batched execution mode, and end-to-end latency.

Clip-level Accuracy is diagnostic only. The gate is final video-level Accuracy.

## Final all-modal specialization

After the temporal module is introduced, run final Decision specialization with replay across Text/Image/Audio and Video.

The purpose of replay is not only anti-forgetting. The final model must continue to improve weak modalities toward the shared 95% product gate.

Candidate selection prioritizes:

1. minimum modality Accuracy;
2. whether all four modalities approach/meet 95%;
3. NLL/Brier/ECE;
4. latency/memory/size frontier.

Do not improve the average score by sacrificing the weakest modality.

## Variant contract

Preserve all major artifacts:

- **Variant A** — ternary core + two-stage Decision Recovery/Re-specialization.
- **Variant B** — Variant A architecture with the lightweight Decision readout.
- **Variant C** — Variant B plus temporal Video specialization and final all-modal Decision specialization.
- **Variant D** — optional Variant C plus pooling/resampling and a replacement post-change specialization state.

Each deployable variant carries only the adaptation state required for that architecture. Do not stack obsolete Recovery adapters.

All claimed comparisons use the same preprocessing, option ordering, device, warmup, and timing protocol.

## Pooling/resampler order

Token pooling/resampling remains after Variant C.

Reason: clip-based Video processing already changes token/compute budgeting. Generic pooling before that point would confound two reductions and make attribution harder.

Attempt Variant D only if Variant C profiling shows a material remaining bottleneck.

## Preservation of existing evidence

Do not rewrite historical experimental results.

Keep as immutable evidence:

- durable Teacher / Teacher v0 records;
- Teacher v1 validation and sealed audit;
- Video Teacher v2 and long-budget records;
- Teacher Quality Next / data-coverage learning curves;
- the active clean-development run once committed.

This design changes the interpretation and next action, not the historical measurements.

## Audio evaluation

Keep the existing LibriSpeech four-choice random-distractor evaluation as a historical baseline.

Add a separate hard-negative Audio evaluation using distractors chosen to be near the target in phonetic, lexical, and/or duration characteristics.

Report both. Do not silently replace the easier historical metric.

## Evaluation integrity

Development selection and final audit remain separate.

Requirements:

- never use a sealed/final audit to tune Recovery schedules or ternary parameters;
- version each new development generation;
- preserve sample/asset overlap checks;
- cluster uncertainty by the natural asset unit where relevant, such as video scene or speaker;
- do not redefine task difficulty after seeing the final result;
- do not claim 95% from an easier substitute benchmark when the product gate was defined on a harder task.

## Out of scope for this revision

- implementing ternary kernels;
- implementing the temporal aggregator;
- changing an in-flight high-precision run;
- touching a sealed audit;
- making an External Teacher mandatory;
- replacing transformer attention before profiling demonstrates that it is a material end-to-end bottleneck;
- claiming that sparsity alone guarantees a quality gain.

## Implementation sequence after this design

1. finish and freeze the current high-precision run;
2. create the reproducible Pre-compression Decision Master artifact;
3. implement/verify ternary conversion and sparsity/damage measurement;
4. run the controlled A/B/C/D/E post-quantization comparison;
5. freeze Variant A using the selected two-stage policy if it wins;
6. implement and benchmark the lightweight Decision readout for Variant B;
7. implement temporal Video specialization and final all-modal specialization for Variant C;
8. profile A/B/C;
9. attempt Variant D only if the measured frontier justifies pooling/resampling.
