# Tiny Omni Decision

Tiny Omni Decision adapts Gemma 4's multimodal representation to return probabilities over supplied choices. Phase 1 established a **text-only decision** path; the current training pipeline supports text, image, audio, and video. The decision readout scores supplied choices and returns a probability distribution instead of generating a response.

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

## Teacher v1 quality run

Teacher v0's evaluation metrics have already been observed. They are retained as
a legacy reference in [docs/DURABLE_TEACHER.md](docs/DURABLE_TEACHER.md) and are
not used for hyperparameter selection. Teacher v1 freezes fresh train and
validation corpora plus a separate sealed audit set:

```bash
tiny-omni-decision freeze-teacher-v1-corpus
tiny-omni-decision train-decision \
  --train-manifest data/processed/durable-teacher-v1/train.jsonl \
  --validation-manifest data/processed/durable-teacher-v1/validation.jsonl \
  --config configs/decision/teacher_v1.yaml \
  --output artifacts/tiny-omni-decision-teacher-v1/candidates/seed17-rank16-512-balanced
```

The trainer requires independent validation and has no evaluation-manifest
argument. It records every attempt, sample/asset accounting, validation learning
curves, per-modality metrics, and the best validation-selected checkpoint.
Sampling policies for weaker modalities are in
`configs/decision/teacher_v1_weak_modalities.yaml` and
`configs/decision/teacher_v1_video_priority.yaml`.

After the candidate, config, train/validation hashes, sampling policy, and
selection rule are frozen, evaluate the sealed audit once:

```bash
tiny-omni-decision freeze-teacher-selection \
  --candidate artifacts/tiny-omni-decision-teacher-v1/candidates/seed17-rank16-512-balanced/best \
  --config-path configs/decision/teacher_v1.yaml \
  --train-path data/processed/durable-teacher-v1/train.jsonl \
  --validation-path data/processed/durable-teacher-v1/validation.jsonl
tiny-omni-decision evaluate-sealed-audit \
  --candidate artifacts/tiny-omni-decision-teacher-v1/candidates/seed17-rank16-512-balanced/best \
  --config-path configs/decision/teacher_v1.yaml \
  --train-path data/processed/durable-teacher-v1/train.jsonl \
  --validation-path data/processed/durable-teacher-v1/validation.jsonl
```

The audit JSONL lives under `data/sealed/durable-teacher-v1/`; normal training
loads only `train.jsonl` and `validation.jsonl`. Its immutable manifest records
the audit hash, and an exclusive claim file prevents a second evaluation.
Teacher v1 procedure, split hashes, coverage, experiments, curves, metrics, and
remaining bottlenecks are documented in
[docs/TEACHER_V1.md](docs/TEACHER_V1.md).

The prior 16-step pipeline test remains historical context in
[docs/PHASE3.md](docs/PHASE3.md). No ternary quantization or Recovery LoRA is
performed during Teacher v1.

## Planned compression and recovery flow

The intended v0.1 path is:

```text
E2B QAT base
  → Decision / Omni LoRA
  → high-precision Decision Teacher
  → merge Decision LoRA into task-adapted quantization source
  → ternary decoder compression
  → Recovery
  → Variant A: recovered ternary model
  → lightweight Decision classifier / option scorer
  → Variant B
  → token pooling / learned resampler
  → re-distillation + replacement Recovery
  → Variant C
  → identical A/B/C quality + latency benchmark
```

Teacher distillation caches **option logits at the single decision position** by default. Full-vocabulary logits are optional diagnostics, not a required cache.

The runtime plan deliberately preserves three benchmarkable artifacts instead of overwriting intermediate stages:

- **Variant A:** ternary + Recovery
- **Variant B:** Variant A with the full-vocabulary runtime readout replaced by a lightweight Decision classifier / option scorer
- **Variant C:** Variant B plus token pooling / a learned resampler, followed by re-distillation and a replacement Recovery adapter

Every deployable variant carries at most one Recovery adapter. Variant C retrains/replaces Recovery after token reduction; it does not stack two Recovery adapters. All three variants are evaluated on the same quality, memory, size, startup, and p50/p95 latency protocol. Deeper attention replacement remains optional after this comparison.

## Current boundary and limitations

- CPU CI covers schema, manifest, token-label mapping, probability normalization, Brier loss, and option reordering. It does not download model weights.
- The ML extra follows the model card's documented Transformers minimum (`>=5.6.2`) and requires PyTorch 2.6 for actual Gemma 4 multimodal forward. Install a wheel matching the local CUDA driver; GPU model loading is not covered by CPU CI.
- Teacher v0's final evaluation has been observed and is only a legacy reference. Teacher v1 uses fresh split assets where available, rejects record/media/state overlap, and isolates its sealed audit until validation-only selection is frozen; see [docs/TEACHER_V1.md](docs/TEACHER_V1.md). The sources remain controlled candidates, not broad real-world benchmarks.
- Option labels must each tokenize to exactly one distinct token; the command fails closed if this assumption is false.
- The synthetic smoke example is a plumbing check, not a quality or calibration evaluation.
- GitHub Actions PR checks validate proposed commits; push checks on `main` validate the resulting merge commit. Both run install, lint, CPU tests, and manifest validation without downloading model weights. The separate 16 GB GPU smoke was run locally.
- OneJev component-level rights review remains unresolved and excluded. Ternary runtime compatibility and paired quality checks after compression remain open later-phase gates; no ternary or Recovery training has been run.

See [ROADMAP.md](ROADMAP.md), [docs/PHASE0.md](docs/PHASE0.md), and [THIRD_PARTY.md](THIRD_PARTY.md).
