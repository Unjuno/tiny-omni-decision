# Tiny Omni Decision

Tiny Omni Decision adapts Gemma 4's multimodal representation to return probabilities over supplied choices. The existing training path supports text and a single image, audio, or video media input per example. Simultaneous multi-media fusion is a required later capability, not an implemented property of that path. The Decision readout scores supplied choices instead of generating a response.

## Pinned base

- Model and processor: `google/gemma-4-E2B-it-qat-q4_0-unquantized`
- Immutable Hugging Face revision: `6befbaca7398925921802abd1f277b495b78b738`
- Architecture in upstream `config.json`: `Gemma4ForConditionalGeneration`
- Processor/tokenizer in pinned metadata: `Gemma4Processor` / `GemmaTokenizer`
- Upstream license metadata: Apache-2.0; preserve Google DeepMind attribution and follow [Gemma 4 terms](https://ai.google.dev/gemma/docs/gemma_4_license) when distributing derived artifacts.
- HF API metadata reports 5,104,297,539 BF16 parameters. The pinned safetensors header has 5,104,298,467 total elements, including 928 scalar QAT metadata entries. Transformers 5.6.2's config-only model, after tying shared embeddings, counts 5,104,297,504 parameters (35 fewer than the API); see the manifest. The unquantized QAT weights occupy about 10.2 GB on disk; normal training also requires activations, gradients, and optimizer state.

The immutable provenance is in [manifests/base-model.example.yaml](manifests/base-model.example.yaml). Model weights are downloaded from Hugging Face and are never committed here. The tokenizer and processor use the same pinned revision.

## Setup

Python 3.11 is supported. For CPU-only schema/loss tests:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install ".[dev]"
```

For model loading and CUDA training, install the ML extra in an environment with a PyTorch CUDA build matching your driver:

```bash
python -m pip install ".[ml,dev]"
```

Gemma 4 multimodal attention masking currently requires PyTorch 2.6 or newer;
the supported ML extra therefore pins PyTorch 2.6.x and matching torchvision
0.21.x. On Windows with a CUDA 12.4-compatible driver, the official wheel command
is `python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url
https://download.pytorch.org/whl/cu124`.

The pinned model is public. Do not put Hugging Face tokens or other credentials in repository files. Fresh-clone CPU preflight:

```bash
tiny-omni-decision validate-model-manifest manifests/base-model.example.yaml
tiny-omni-decision validate-dataset-manifest manifests/dataset.example.yaml
tiny-omni-decision preflight
ruff check .
pytest -q
```

## Inspect the actual model

This loads the approximately 10.2 GB checkpoint and prints the instantiated architecture, parameter count, module names, candidate linear LoRA targets, modality-related modules, dtypes, memory estimate, and processor/tokenizer classes. Use `--config-only` to instantiate the pinned architecture on `meta` without downloading weights; this does not verify checkpoint tensor shapes:

```bash
tiny-omni-decision inspect-model
tiny-omni-decision inspect-model --config-only
```

No guessed module names are used in the config. The smoke selects exact decoder module paths after loading and verifies that PEFT adapted those paths only.

## Text decision smoke training

The explicit GPU command runs a synthetic one-sample forward pass, CE + Brier loss, backward, optimizer step, LoRA save, base unload/reload, adapter reload, and probability inference:

```bash
tiny-omni-decision smoke-text-decision --seed 17
```

The command refuses CPU execution, guessed target modules, non-finite loss, missing adapter gradients, or gradients on frozen base parameters. It saves the adapter and `run-metadata.json` under `artifacts/smoke-text-decision/`, including base revision/config and seed.

### Hardware expectation

The Phase 1 text smoke was run on an NVIDIA GeForce RTX 3080 Laptop GPU with
16,384 MiB VRAM, PyTorch 2.5.1+cu121, Transformers 5.6.2, and PEFT 0.21.2.
Gemma 4's multimodal path requires PyTorch 2.6. The earlier bounded pipeline
check and the durable teacher run are recorded in
[docs/PHASE3.md](docs/PHASE3.md) and
[docs/DURABLE_TEACHER.md](docs/DURABLE_TEACHER.md).

## Teacher v1 quality run — historical procedure

Teacher v0's evaluation metrics have already been observed. They are retained as
a legacy reference in [docs/DURABLE_TEACHER.md](docs/DURABLE_TEACHER.md) and are
not used for hyperparameter selection. Teacher v1 froze fresh train and
validation corpora plus a separate sealed audit set. The following commands
document that historical procedure; do not rerun them over existing artifacts:

```bash
tiny-omni-decision freeze-teacher-v1-corpus
tiny-omni-decision train-decision \
  --train-manifest data/processed/durable-teacher-v1/train.jsonl \
  --validation-manifest data/processed/durable-teacher-v1/validation.jsonl \
  --config configs/decision/teacher_v1_weak_modalities.yaml \
  --output artifacts/tiny-omni-decision-teacher-v1/candidates/seed17-rank16-512-weak-modalities-normalized
```

The trainer requires independent validation and has no evaluation-manifest
argument. It records every attempt, sample/asset accounting, validation learning
curves, per-modality metrics, and the best validation-selected checkpoint.
The validation-only policy comparison, three-seed learning curves, selected
checkpoint, and final sealed-audit report are documented in
[`docs/TEACHER_V1.md`](docs/TEACHER_V1.md). The selected policy weights are
audio 1, image 1.5, text 1.5, video 2. The alternative video-priority policy is
in `configs/decision/teacher_v1_video_priority.yaml`; seed-specific reproducible
configs are retained for seeds 19 and 23.

After the candidate, config, train/validation hashes, sampling policy, and
selection rule were frozen, the historical sealed audit was evaluated once:

```bash
tiny-omni-decision freeze-teacher-selection \
  --candidate artifacts/tiny-omni-decision-teacher-v1/candidates/seed17-rank16-512-weak-modalities-normalized/best \
  --config-path configs/decision/teacher_v1_weak_modalities.yaml \
  --train-path data/processed/durable-teacher-v1/train.jsonl \
  --validation-path data/processed/durable-teacher-v1/validation.jsonl
tiny-omni-decision evaluate-sealed-audit \
  --candidate artifacts/tiny-omni-decision-teacher-v1/candidates/seed17-rank16-512-weak-modalities-normalized/best \
  --config-path configs/decision/teacher_v1_weak_modalities.yaml \
  --train-path data/processed/durable-teacher-v1/train.jsonl \
  --validation-path data/processed/durable-teacher-v1/validation.jsonl
```

The audit JSONL lives under `data/sealed/durable-teacher-v1/`; normal training
loads only `train.jsonl` and `validation.jsonl`. Its immutable manifest records
the audit hash, and an exclusive claim file prevents a second evaluation.
This audit is now observed; it is not a fresh final audit for a new candidate.

The one-time sealed audit scored 0.6718 Accuracy overall (macro across
modalities: 0.7170). Audio reached 0.9459; image 0.6890, text 0.7130, and video
0.5200 did not meet the then-used 0.90-per-modality development goal. The
current final product target is 95% in every modality. The selected adapter
and provenance hashes remain in
[`manifests/teachers/tiny-omni-decision-teacher-v1.json`](manifests/teachers/tiny-omni-decision-teacher-v1.json).

The prior 16-step pipeline test remains historical context in
[docs/PHASE3.md](docs/PHASE3.md). No ternary quantization or Recovery LoRA was
performed during Teacher v1. Later development runs are separately identified
in [docs/TEACHER_QUALITY_NEXT.md](docs/TEACHER_QUALITY_NEXT.md); do not combine
their validation generations into one learning curve.

## Planned compression, Recovery and Decision Re-specialization

Approved direction B allows a frozen, reload/merge-verified Master to enter
bounded compression experiments below final product accuracy. The final
artifact still requires Text >=95%, Image >=95%, Audio >=95%, and end-to-end
Video >=95% Accuracy, plus declared fusion and runtime qualification.

```text
E2B base -> Decision adaptation -> frozen Master -> verified LoRA merge
  -> input/quantizer feasibility -> Q0-Q4 experiment
  -> Recovery -> candidate Decision Re-specialization -> Variant A
  -> equivalent option-only readout first -> Variant B
  -> verified multi-media fusion + temporal Video + joint specialization -> Variant C
  -> optional profiling-justified pooling/resampling -> Variant D
  -> untouched final audit and actual target-runtime benchmark
```

Q0 is Master; Q1 raw ternary; Q2 fixed Recovery; Q3 two-stage Recovery and
Re-specialization; Q4 non-quantized control with the same additional learning
as Q3. These are experiment arms, not product variant names. Match adapter
initialization/topology, sample order, update budget, optimizer/LR policy and
Q3/Q4 loss schedules; compare fixed-update endpoints separately from selected
checkpoints. Quantization-induced improvement is a hypothesis, not a guarantee.

The initial candidate uses Recovery for the first quarter of updates, then
reduces KD influence and strengthens gold-label supervision. Keep fixed
Recovery when it wins. Cache only ordered option logits by default; this does
not require preserving long-form generation or full-vocabulary distributions.

Preserved product artifacts:
- **Variant A:** selected feasible ternary Decision core and one adaptation state.
- **Variant B:** A with equivalent or independently validated lightweight readout, preserving 2-62 choices.
- **Variant C:** B with temporal Video and final joint all-modal specialization after the fusion gate.
- **Variant D:** optional C with pooling/resampling and replacement specialization.

Do not stack obsolete Recovery adapters or silently merge a dense residual
into three-valued weights. Count the residual/readout/temporal modules in
all bytes, memory and latency reports. A BF16-dequantized ternary reference
can test quality but cannot establish packed-runtime speed.

Read the [design spec](docs/superpowers/specs/2026-10-06-compression-first-video-specialization-design.md),
[remaining contract sync plan](docs/superpowers/plans/2026-10-06-decision-respecialization-sync.md),
and [actual bounded experiment plan](docs/superpowers/plans/2026-10-06-post-quantization-respecialization-experiment.md).
The existing `configs/recovery/probability_distillation.yaml` is still a fixed-loss
legacy control. A versioned two-stage config, its parser, schedule/loss/resume
code and measured experiments are required before claiming the new policy runs.

## Current boundary and limitations

- CPU CI and prior tests do not establish current GPU quality, packed execution speed or 95% release accuracy. This documentation revision does not run model training.
- The ML extra and pinned model/backend compatibility must be checked in the actual execution environment. Historical smoke conditions are not a new benchmark.
- Current single-media input processing does not implement general simultaneous image/audio/video fusion. New experiments must reject unsupported extra inputs before calling it; final Omni specialization requires a tested capability matrix and correctly aligned multi-evidence data.
- Teacher v0/v1 and Video v2 results, including failed staging-path attempts and negative results, remain historical reports. Their observed evaluations are not fresh audits for new models.
- The source tasks are controlled candidates, not proof of broad real-world quality. Keep Audio random-negative results and add validated hard-negative evaluation separately.
- All option labels must map to distinct single continuation tokens; preserve the current 2-62 choice contract and test the 60-intent task.
- OneJev unresolved rights remain excluded. Architecture-compatible ternary reference conversion, Recovery/Re-specialization GPU execution, packed runtime, fusion and temporal specialization are planned until implementation and measurements demonstrate them.
- Attention replacement is separate optional profiling-gated research, not the lightweight-readout stage.

See [ROADMAP.md](ROADMAP.md), [docs/PHASE0.md](docs/PHASE0.md), and [THIRD_PARTY.md](THIRD_PARTY.md).
