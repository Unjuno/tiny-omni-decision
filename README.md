# Tiny Omni Decision

Tiny Omni Decision adapts Gemma 4's multimodal representation to return probabilities over supplied choices. Phase 1 implements **text decision** only. It does not generate a response: it scores one-token labels (`A`, `B`, …) from the model's vocabulary logits and returns a distribution over the supplied options.

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

The smoke path was run successfully on an NVIDIA GeForce RTX 3080 Laptop GPU with 16,384 MiB VRAM, PyTorch 2.5.1+cu121, Transformers 5.6.2, and PEFT 0.21.2. It loaded the pinned checkpoint (about 10.21 GB parameter memory), then completed forward, CE+Brier, backward, optimizer step, LoRA save/reload, and same-sample inference. It uses `device_map="auto"`, gradient checkpointing, a short prompt, and micro-batch 1; it adds no secondary quantization. This is evidence for that setup, not a guarantee for every 16 GB GPU or software stack.

## Current boundary and limitations

- CPU CI covers schema, manifest, token-label mapping, probability normalization, Brier loss, and option reordering. It does not download model weights.
- The ML extra follows the model card's documented Transformers minimum (`>=5.6.2`). Install a PyTorch build that matches the local CUDA driver; GPU model loading is not covered by CPU CI.
- Only text inputs are trained. Image/audio/video processing, dataset training, ternary conversion, Recovery LoRA, and rented GPU runs are out of scope for Phase 1.
- Option labels must each tokenize to exactly one distinct token; the command fails closed if this assumption is false.
- The synthetic smoke example is a plumbing check, not a quality or calibration evaluation.
- GitHub Actions has not yet run for this working tree; local lint, CPU tests, and manifest validation pass.
- Dataset licensing, held-out evaluation splits, and ternary runtime compatibility remain hard gates before Phase 2/3 durable work.

See [ROADMAP.md](ROADMAP.md), [docs/PHASE0.md](docs/PHASE0.md), and [THIRD_PARTY.md](THIRD_PARTY.md).
