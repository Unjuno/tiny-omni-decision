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

EmbeddingGemma 2 requires Transformers 5.19.x. Install the separate student
extra in its own environment with a compatible PyTorch build:

```bash
python -m pip install ".[student,dev]"
```

Do not combine `[student]` with the Teacher-oriented `[ml]` extra. The current
student manifest, loaded graph inventory, CPU forward/backward smoke, and
verification boundary are recorded in
[docs/EMBEDDINGGEMMA2_STUDENT.md](docs/EMBEDDINGGEMMA2_STUDENT.md).

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

## Durable Decision Teacher

The reproducible train, validation, and held-out evaluation corpus can be frozen
with:

```bash
tiny-omni-decision freeze-corpus \
  manifests/durable-training-corpus.yaml \
  manifests/durable-heldout-corpus.yaml \
  data/processed/durable-teacher-v0 \
  --max-records-per-source 2048
```

Then run the four-modality Decision LoRA training, validation selection, and
full held-out evaluation:

```bash
tiny-omni-decision train-decision \
  --train-manifest data/processed/durable-teacher-v0/train.jsonl \
  --eval-manifest data/processed/durable-teacher-v0/eval.jsonl \
  --validation-manifest data/processed/durable-teacher-v0/validation.jsonl \
  --config configs/decision/durable_teacher.yaml \
  --output artifacts/tiny-omni-decision-teacher-v0
```

Corpus provenance, hashes, split checks, training setup, measured metrics,
limitations, artifacts, and the later merge/export path are in
[docs/DURABLE_TEACHER.md](docs/DURABLE_TEACHER.md). The earlier 16-step
pipeline test is retained as historical context in [docs/PHASE3.md](docs/PHASE3.md).

## Student, compression, and recovery flow

**Next experiment: ternary first, recover with the Teacher, add LoRA only if recovery is insufficient.**

```text
Gemma 4 Decision Teacher: finish → select → reload → freeze
                                              │
                                              │ supervises recovery
Pretrained EmbeddingGemma 2                    │
  → aggressive ternary conversion             │
  → measure initial damage                    │
  → ternary-constrained QAT + distillation ←───┘
  → if sufficient: export / reload / final evaluation / runtime
  → if insufficient: freeze quantized base
       → add one Recovery LoRA and train it with the same Teacher
       → export / reload / re-evaluate
```

Do not first train a separate high-precision decision student. Gemma 4 remains both the source of option-level supervision and the quality reference. The existing Teacher run and artifacts are unchanged. EmbeddingGemma 2's pinned weights and processor have been loaded in an isolated Transformers 5.19.0 environment, and a synthetic text batch completed a CPU forward/readout/backward smoke. This is not a student quality or recovery result.

Target the student's large text/backbone, embedding, vision, and audio weights for ternary conversion after pinning and inspecting the actual model. Record every higher-precision exception. During the first recovery stage, each forward uses ternary-quantized target weights even when gradient updates use higher-precision shadow weights.

If that recovery is insufficient, freeze the selected quantized base and train one separate Recovery LoRA. Count its bytes and runtime overhead. Do not silently merge it into the ternary base: a merged correction is generally no longer ternary. The approximately 1.58-bit packing goal is not a measured whole-model size or a performance guarantee.

Keep native backbone/pooling initially and define only the minimal supplied-option readout. The earlier pooling-plus-small-classifier structural fallback is deferred until constrained recovery and LoRA are insufficient; it is not a mandatory stage before quantization. Inspect actual modules before any terminal-path removal.

Reuse compact Teacher option signals, matching the actual choices and their order rather than assuming shared vocabulary IDs. Training, validation selection, and final held-out evaluation remain separate. The detailed sequence is in [ROADMAP.md](ROADMAP.md).

## Current boundary and limitations

- CPU CI covers schema, manifest, token-label mapping, probability normalization, Brier loss, and option reordering. It does not download model weights.
- The Teacher-oriented ML extra requires PyTorch 2.6 for Gemma 4 multimodal forward; the separate student extra pins Transformers 5.19.x. Install each in a separate environment and match the PyTorch wheel to the local CUDA driver. CPU CI does not download or load model weights; the EmbeddingGemma 2 checkpoint smoke was performed locally on CPU.
- The durable Teacher run uses frozen text, image, audio, and video corpora with zero pairwise source-ID and normalized-content overlap. Clevr-4 (CC BY 4.0), Speech Commands (CC-BY-4.0), and CLEVRER (CC0) provide controlled multimodal candidates; OneJev remains excluded pending component-level rights review. The dataset is sampled and the training budget is capped, so the results are not benchmark-generalizing quality claims; see [docs/DURABLE_TEACHER.md](docs/DURABLE_TEACHER.md).
- The current Gemma 4 vocabulary readout requires option labels to tokenize to exactly one distinct token each; it fails closed otherwise. This does not imply a shared Teacher/student tokenizer.
- The synthetic smoke example is a plumbing check, not a quality or calibration evaluation.
- GitHub Actions PR checks validate proposed commits; push checks on `main` validate the resulting merge commit. Both run install, lint, CPU tests, and manifest validation without downloading model weights. The separate 16 GB GPU smoke was run locally.
- OneJev component-level rights review remains unresolved and excluded. The pinned EmbeddingGemma 2 checkpoint and packed 483-tensor ternary overlay pass CPU path/reload smokes, but the standard runtime expands ternary weights back to BF16 and provides neither packed compute nor inference-memory reduction. The same 256 fixed validation examples were evaluated on the RTX 3080 Laptop GPU: the unquantized pretrained-student diagnostic scored 44.53% accuracy and the unrecovered ternary overlay 19.14%. Teacher-supervised recovery and a self-contained deployment package remain open; detailed metrics and hashes are in [docs/EMBEDDINGGEMMA2_STUDENT.md](docs/EMBEDDINGGEMMA2_STUDENT.md).
- The EmbeddingGemma 2 readout prototype is separate from the active Teacher training path. Existing Teacher configs, corpora, checkpoints, and artifacts remain unchanged. The Gemma 4 quantization/recovery configs do not implement this student path.
- A local ternary Recovery LoRA experiment completed one pass over 4,859 unique training examples. Its selected checkpoint reached 25.78% macro accuracy on the fixed validation snapshot, with substantial modality gaps; it is not a product promotion. Metrics, hashes, hardware, and limitations are recorded in [docs/EMBEDDINGGEMMA2_TERNARY_RECOVERY_V0.md](docs/EMBEDDINGGEMMA2_TERNARY_RECOVERY_V0.md).
- Phase 9 exported and hash-verified a portable bundle, confirmed exact best-validation reload parity, and paired the unquantized, ternary, Recovery LoRA, and Teacher on the same historical 2,917-example evaluation. Recovery reached 31.98% overall / 23.97% macro accuracy versus Teacher 73.43% / 65.24%; this is not a blind audit or product promotion. An opt-in bounded video decode cache also passed bitwise processor-input parity on a repeated-scene microbenchmark; full-evaluation speedup remains unmeasured. See the Phase 9 section of the Recovery report.

See [ROADMAP.md](ROADMAP.md), [docs/PHASE0.md](docs/PHASE0.md), and [THIRD_PARTY.md](THIRD_PARTY.md).
