# Compression-first roadmap with post-compression video specialization

## Status

Approved architectural direction: **B**.

The project will no longer block ternary compression on achieving the final product quality target in all four modalities at the high-precision stage. The active high-precision run should complete, its validation-selected best checkpoint should be frozen reproducibly, and that checkpoint becomes the **Pre-compression Decision Master** even if one or more modalities remain below the final 95% product target and Video remains near its current full-video baseline.

The final product quality gate remains strict. Compression is allowed to begin before the final quality gate is met; shipping is not.

## Why this changes the roadmap

The current evidence does not support spending the entire project budget on forcing the high-precision E2B model to solve long-video reasoning before compression.

The latest completed tracked data-coverage run selected step 1,792 with:

| Modality | Accuracy |
|---|---:|
| Audio | 1.0000 |
| Image | 0.8359 |
| Text | 0.8125 |
| Video | 0.5156 |

The run improved Text, Image, and Audio substantially over the frozen Teacher v1 reference on the same development subset, while Video remained roughly flat around 50%. Earlier Video Teacher v2 experiments also failed to show a reliable route from more frames, broader decoder LoRA targets, or rank changes to the final 95% Video target.

This is evidence for changing **where** video specialization happens, not for accepting 50% video as final quality.

## Terminology

Use the following terms consistently:

- **Pre-compression Decision Master**: the validation-selected high-precision E2B Decision model that is merged and used as the quantization source.
- **Recovery Teacher**: the frozen Pre-compression Decision Master used to recover the ternary student after quantization.
- **Ternary Student**: the quantized E2B during recovery.
- **External Teacher**: an optional stronger multimodal model used for ability distillation. This is separate from the Recovery Teacher and is not required for this roadmap revision.

## Quality gates

### Compression entry gate

Compression may begin when all of the following are true:

1. the active high-precision run has completed its declared budget or stopped under its predeclared rule;
2. a validation-selected best checkpoint is frozen;
3. checkpoint reload reproduces the selected predictions and metrics;
4. train/development split-integrity checks pass;
5. no sealed/final audit was used for tuning;
6. the Master artifact records config, data hashes, checkpoint hash, environment, and per-modality Accuracy/NLL/Brier/ECE.

**No final-product Accuracy threshold is required at this gate. In particular, the 95% per-modality product gate does not block compression entry.**

### Final product gate

The final lightweight model must achieve:

- Text Accuracy >= 95%
- Image Accuracy >= 95%
- Audio Accuracy >= 95%
- end-to-end Video Accuracy >= 95%

The video target applies to the complete deployed video path, not to an individual clip classifier.

The final evaluation must also report NLL, Brier, ECE, latency, startup, memory, artifact size, and the same cross-modal/ablation checks used to verify that the model actually uses the supplied modalities.

## Revised architecture flow

```text
High-precision E2B Decision adaptation
        ↓
freeze validation-selected Pre-compression Decision Master
        ↓
merge Decision LoRA
        ↓
decoder-focused ternary compression
        ↓
paired quantization-damage measurement
        ↓
Recovery from frozen Master + labeled supervision
        ↓
Variant A: recovered ternary Decision model
        ↓
lightweight Decision classifier / option scorer
        ↓
re-distillation / Recovery
        ↓
Variant B: lightweight Decision Core
        ↓
Video Temporal Specialization
  raw video → clips → compact clip states/scores
            → lightweight temporal aggregator
        ↓
all-modal replay + final Recovery
        ↓
Variant C: lightweight Omni Decision with temporal video
        ↓
profile quality / latency / memory
        ↓
only if profiling justifies it:
token pooling / learned resampler
        ↓
replacement Recovery
        ↓
Variant D
```

## Recovery objective

Recovery must not be pure teacher imitation. The default conceptual objective is:

```text
L =
  lambda_KD * KL(Master option distribution || Student option distribution)
+ lambda_GT * CE(label, Student)
+ lambda_B  * Brier(label, Student)
```

This permits the quantized model to preserve useful Master behavior while still correcting Master mistakes where ground-truth labels are available.

Exact weights are an experiment parameter and must be selected on clean development data, not a sealed audit.

## Video Temporal Specialization

### Purpose

Do not require the high-precision Master to solve the final long-video product problem before compression. Build the video-specific temporal mechanism on the architecture that is actually intended for deployment.

### Input path

The first implementation should compare a small number of explicit strategies:

1. full-video baseline retained from Variant B;
2. fixed non-overlapping clips plus temporal aggregation;
3. overlapping or event-dense clips plus the same aggregator, only if the fixed-clip baseline justifies the added compute.

Each clip produces either a compact hidden representation, compact Decision scores, or both. A lightweight temporal aggregator combines those signals into the final option distribution.

### Training

Video specialization should use clean scene-disjoint video data. It should include Text/Image/Audio replay so video tuning cannot silently destroy the existing Decision Core.

The replay ratio is selected from validation evidence rather than hard-coded in the roadmap. Candidate selection prioritizes final end-to-end quality and preservation of non-video modalities.

### Evaluation

Report:

- full-video baseline versus clip/aggregator end-to-end Accuracy;
- Video NLL/Brier/ECE;
- per-task video metrics;
- clip count and sampled-frame count;
- p50/p95 latency;
- memory and startup cost;
- modality-ablation and contradiction tests where applicable.

Clip-level Accuracy is diagnostic only. The product gate is the final aggregated Video decision.

## Variant contract

Preserve every major artifact rather than overwriting it:

- **Variant A** — ternary + Recovery.
- **Variant B** — Variant A architecture with the lightweight Decision readout.
- **Variant C** — Variant B plus temporal video specialization and all-modal final Recovery.
- **Variant D** — Variant C plus optional pooling/resampling and a replacement Recovery, only if profiling demonstrates a worthwhile frontier improvement.

All variants must be benchmarked on identical preprocessing, option ordering, warmup, device, and timing protocol where comparison is claimed.

## Pooling/resampler order

Token pooling/resampling moves **after** temporal video specialization.

Reason: clip-based video processing is already a form of compute/token budgeting. Designing an aggressive generic pooling layer first risks optimizing the same bottleneck twice and makes causal attribution harder.

Variant C must be profiled before Variant D is attempted.

## Preservation of existing evidence

Do not rewrite historical results to fit this decision.

The following remain immutable historical evidence:

- Teacher v0 / durable Teacher records;
- Teacher v1 validation and sealed audit;
- Video Teacher v2 and long-budget records;
- Teacher Quality Next / data-coverage learning curves;
- the active clean-development run once its result is committed.

The roadmap may reinterpret what those experiments imply for the next phase, but their metrics, hashes, failures, and negative results remain unchanged.

## Audio evaluation

The existing LibriSpeech four-choice transcript-selection metric remains a baseline, but its random distractors are potentially too easy. Add a separate hard-negative audio evaluation using phonetic/lexical/length-near distractors.

Do not replace the historical baseline silently. Report both.

## Out of scope for this revision

- implementing ternary kernels;
- implementing the temporal aggregator;
- changing current in-flight training;
- touching a sealed audit;
- declaring the current Video result sufficient for product quality;
- making an external GPT teacher mandatory;
- replacing transformer attention before profiling proves it is a material bottleneck.

## Implementation sequence after this design

1. finish and freeze the current high-precision run;
2. create a reproducible Master artifact record;
3. implement/verify ternary conversion and paired damage evaluation;
4. implement Recovery and Variant A;
5. implement the lightweight Decision readout and Variant B;
6. implement temporal video specialization and final all-modal Recovery for Variant C;
7. profile A/B/C;
8. attempt Variant D only if the measured frontier justifies pooling/resampling.
