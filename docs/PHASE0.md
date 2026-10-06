# Phase 0 — Locked Design Decisions

## High-precision teacher target

`google/gemma-4-E2B-it-qat-q4_0-unquantized`

Phase 1 pinned the upstream repository and processor to immutable revision
`6befbaca7398925921802abd1f277b495b78b738`. See
`manifests/base-model.example.yaml` for the API-reported parameter count,
architecture, license metadata, and file identifiers. The model card identifies
Apache-2.0 and links Google DeepMind's Gemma 4 terms; comply with both when
redistributing derivative artifacts. The unquantized QAT checkpoint contains
BF16 weights and is not itself a 4-bit runtime checkpoint.

The project uses this checkpoint to build the high-precision Gemma 4 Decision Teacher. Keep the current Teacher training and its artifacts unchanged; finish the run, select on validation, verify reload, and freeze the Master Teacher before student recovery. This document does not assert that the active local run has completed.

## Student experiment: ternary first

EmbeddingGemma 2 is the student for the next experiment. Pin its exact upstream model ID, immutable revision, processor, license metadata, module graph, and execution requirements before implementation.

The sequence is:

```text
Pretrained EmbeddingGemma 2
  → aggressive ternary conversion
  → measure unrecovered damage
  → recover under ternary forward constraints with the frozen Gemma 4 Teacher
  → if insufficient: freeze quantized base and add/train one Recovery LoRA
  → packed export, reload, final evaluation, and runtime
```

Do not require a separately task-trained high-precision student first. The Teacher supervises both the main constrained-recovery stage and, if needed, the LoRA fallback. Successful recovery is an experiment outcome to establish, not an assumption.

Keep the native backbone/pooling structure initially. Define only the minimal readout needed to produce scores over the supplied choices, with stable option identity/order. Teacher vocabulary token IDs need not match student token IDs.

The earlier pooling-plus-small-classifier structural fallback remains available only after recovery and the LoRA fallback are insufficient. Inspect actual modules before replacing a readout or removing any terminal attention/projection block. Do not assume a cut point or make structural surgery a prerequisite.

## What the Teacher base is expected to provide

- aligned text representations
- image understanding
- native audio understanding
- video through the model's supported multimodal path
- a trainable decoder suitable for lightweight decision adaptation

## What the project adds

1. typed decision behavior and calibrated probability outputs;
2. aggressive ternary conversion of an EmbeddingGemma 2 student;
3. Teacher-supervised recovery while retaining ternary forward constraints;
4. one Recovery LoRA on the frozen quantized base only if constrained recovery is insufficient;
5. reproducible edge/mobile packaging and quality/size accounting.

## Precision policy

| Component | Planned policy |
|---|---|
| Large student text/backbone weights and embeddings | ternary target after module inspection |
| Large vision/audio weights | included in the ternary target inventory, not silently excluded |
| Norms, biases, scales, and unsupported/sensitive tensors | explicit higher-precision exceptions with byte accounting |
| Native pooling and minimal decision readout | retain initially; record any parameter and precision cost |
| Activations and accumulation | supported numerical dtype, BF16/FP32 reference; validate any reduction separately |
| Training shadow weights and optimizer state | training-only higher precision; excluded from deployment-size claims |
| Optional Recovery LoRA | separate higher-precision adapter; included in deployment-size claims |

The approximately 1.58-bit target describes packed ternary weight codes, not a guaranteed whole-model size. Count scales, metadata, packing overhead, exceptions, readout, and any LoRA. Fake-quantized floating-point storage is not a packed low-bit export. Actual size, memory, latency, and quality remain unmeasured for this student experiment.

## Teacher, constrained recovery, and LoRA lifecycle

The Teacher stays frozen. First convert the pretrained student to ternary, record the initial damage, then update student shadow weights under quantization-aware distillation. Every forward quantizes target weights; exporting or evaluating an unconstrained high-precision copy does not satisfy this stage.

If that recovered student meets the predefined validation quality and deployment limits, export it without LoRA. Otherwise freeze its quantized codes, scales, and base parameters, attach one Recovery LoRA to verified compatible modules, and train that adapter with the same Teacher.

Keep the LoRA separate at deployment by default. Its correction is not itself ternary and its bytes/overhead count toward the package. Merging into the base generally breaks the ternary constraint; any merge/requantization needs fresh validation rather than being assumed lossless.

Use option-distribution KL, labeled CE, and Brier objectives with compact Teacher option signals. Use training data for updates, validation for selection and the fallback decision, and a frozen final evaluation split for the final paired report. An embedding vector alone is not a calibrated distribution over supplied choices.

## Hard-gate status

Teacher provenance and the first corpus's disjoint splits are recorded in the existing manifests. Student integration, recovery, and packed runtime support remain planned, not implemented.

- [x] pin exact Teacher upstream model and processor revision
- [x] record Teacher upstream license/notice requirements
- [x] record Teacher base and corpus file hashes
- [x] freeze non-overlapping train/validation/evaluation manifests
- [ ] pin and inspect the exact EmbeddingGemma 2 student revision
- [ ] verify student option readout, ternary-constrained training, and save/reload
- [ ] set numeric acceptance limits and a bounded recovery budget before selecting results
- [ ] verify packed export and architecture-compatible runtime, including LoRA if used

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
- optional future scale-up after local training memory and budget are reviewed

A small inference artifact does not establish that shadow-weight QAT and optimizer state fit the local training budget.

## Research boundary

This is a product-oriented compression project, not a benchmark survey. Comparisons are added only when they answer an engineering decision. This documentation change does not alter running jobs, training code, or configs. Existing Gemma 4 quantization/recovery configs remain historical starting points, not a working EmbeddingGemma 2 training path. See [ROADMAP.md](../ROADMAP.md) for the sequence and deferred structural fallback.
