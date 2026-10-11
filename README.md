# Tiny Omni Decision

Tiny Omni Decision adapts Gemma 4's multimodal representation to return probabilities over supplied choices. Phase 1 established a **text-only decision** path; the current training pipeline supports text, image, audio, and video. The decision readout scores supplied choices and returns a probability distribution instead of generating a response.

The current pretrained-representation research follows the R0–R8 process in [ROADMAP.md](ROADMAP.md). Frozen-feature text, synthetic image, CLEVRER video (including a binary explanatory/predictive/counterfactual readout, a new-scene text/video control, and a temporal-order reversal control), and closed-set English keyword audio probes are recorded in [docs/pretrained_reuse/R2_PATH_A_PROBES.md](docs/pretrained_reuse/R2_PATH_A_PROBES.md). A post-hoc Speech Commands control shows that the saved Audio+Text readout depends on its audio features, but the selected validation is not blind and the task remains narrow ([report](docs/pretrained_reuse/AUDIO_MULTIQUERY_FEATURE_DEPENDENCE_20261011.md)). A full-coverage static CLEVR-4 image probe with frozen VideoPrism reached 87.9% on the previously used 512-image selector and 88.1% on a one-time 1,595-image validation subset; a packaged local replay matched 128 saved decisions across 32 images. The synthetic task, changed encoder and expanded training coverage limit the quality comparison; the runtime bundle is 458 MB and CPU-only replay used about 4.26 GB peak working set. The CLEVRER controls found no clear scene-specific video contribution or temporal-order sensitivity for the current small readout; this is not a backbone capability claim. A dispatcher now replays eight saved assets per modality and records latency distributions, parity, caching, and memory ([report](docs/pretrained_reuse/R2_PATH_A_LATENCY_DISTRIBUTION_20261011.md)). Joint-modality quality remains pending. These development probes are not a product Teacher or release result.

The Gemma 4 material below documents the legacy implementation. The pretrained-representation research direction is tracked in [ROADMAP.md](ROADMAP.md); the isolated inventory, pinned checkpoint audits, evaluation protocol and frozen-feature probes are recorded under [docs/pretrained_reuse](docs/pretrained_reuse/). These are research prototypes, not a complete multimodal Decision pipeline or selected product model.

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

## Planned student, compression, and recovery flow

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

Do not first train a separate high-precision decision student. Gemma 4 remains both the source of option-level supervision and the quality reference. The existing Teacher run and artifacts are unchanged; student recovery is planned, not yet demonstrated.

Target the student's large text/backbone, embedding, vision, and audio weights for ternary conversion after pinning and inspecting the actual model. Record every higher-precision exception. During the first recovery stage, each forward uses ternary-quantized target weights even when gradient updates use higher-precision shadow weights.

If that recovery is insufficient, freeze the selected quantized base and train one separate Recovery LoRA. Count its bytes and runtime overhead. Do not silently merge it into the ternary base: a merged correction is generally no longer ternary. The approximately 1.58-bit packing goal is not a measured whole-model size or a performance guarantee.

Keep native backbone/pooling initially and define only the minimal supplied-option readout. The earlier pooling-plus-small-classifier structural fallback is deferred until constrained recovery and LoRA are insufficient; it is not a mandatory stage before quantization. Inspect actual modules before any terminal-path removal.

Reuse compact Teacher option signals, matching the actual choices and their order rather than assuming shared vocabulary IDs. Training, validation selection, and final held-out evaluation remain separate. The detailed sequence is in [ROADMAP.md](ROADMAP.md).

## Current boundary and limitations

- CPU CI covers schema, manifest, token-label mapping, probability normalization, Brier loss, and option reordering. It does not download model weights.
- The ML extra follows the model card's documented Transformers minimum (`>=5.6.2`) and requires PyTorch 2.6 for actual Gemma 4 multimodal forward. Install a wheel matching the local CUDA driver; GPU model loading is not covered by CPU CI.
- The durable Teacher run uses frozen text, image, audio, and video corpora with zero pairwise source-ID and normalized-content overlap. Clevr-4 (CC BY 4.0), Speech Commands (CC-BY-4.0), and CLEVRER (CC0) provide controlled multimodal candidates; OneJev remains excluded pending component-level rights review. The dataset is sampled and the training budget is capped, so the results are not benchmark-generalizing quality claims; see [docs/DURABLE_TEACHER.md](docs/DURABLE_TEACHER.md).
- The current Gemma 4 vocabulary readout requires option labels to tokenize to exactly one distinct token each; it fails closed otherwise. This does not imply a shared Teacher/student tokenizer.
- The synthetic smoke example is a plumbing check, not a quality or calibration evaluation.
- GitHub Actions PR checks validate proposed commits; push checks on `main` validate the resulting merge commit. Both run install, lint, CPU tests, and manifest validation without downloading model weights. The separate 16 GB GPU smoke was run locally.
- OneJev component-level rights review remains unresolved and excluded. EmbeddingGemma 2 revision pinning, integration, ternary-constrained recovery, optional LoRA recovery, and packed runtime support remain open. No student recovery or student-size result is claimed.
- This is a documentation-only plan change. Existing training code, configs, manifests, and checkpoints are unchanged. The Gemma 4 quantization/recovery configs do not implement this new student path.

See [ROADMAP.md](ROADMAP.md), [docs/PHASE0.md](docs/PHASE0.md), and [THIRD_PARTY.md](THIRD_PARTY.md).


### STARSS23 measured update (2026-10-11)

All 78 approved paired recordings and 2,720 five-second windows were processed
with frozen audio/video encoders. The preregistered readout comparison completed:
audio-only validation NLL 0.2704, video-only 0.4108, joint 0.3290. Joint input
did not improve audio-only; zeroing joint video improved NLL to 0.2978.
This closes the initial paired-data execution gap, but not the joint-quality
gate. Two validation rooms, class imbalance and event-specific losses limit
interpretation. See [STARSS23 comparison](docs/pretrained_reuse/R2_STARSS23_COMPARISON_20261011.md)
for metrics, curves, controls and artifact hashes. No product promotion,
compression, additional seed or sealed evaluation followed.

### Bounded text adaptation result (2026-10-12)

Approved final-block adaptation of a 22.7M Cross-Encoder completed three epochs.
On the same 2,540 development questions, Accuracy rose from zero-shot 45.98%
to 58.11%, with NLL 1.19869 -> 0.93779. Only 1.775M existing parameters were
updated; frozen hashes and best reload checks passed. Calibration worsened
slightly in the last epoch. This is synthetic English text evidence, not an
Omni quality or quantized deployment result. See
[bounded adaptation report](docs/pretrained_reuse/R2_TEXT_CROSSENCODER_ADAPT_20261011.md).

### Conditional text limitation (2026-10-12)

The fixed adapted Cross-Encoder failed the approved64-state numerical condition
probe:35.94% Accuracy, always selecting B, with0/128 answer changes when switching
lowest/highest. No input truncation occurred. The earlier58.11% synthetic corpus
result therefore does not qualify general conditional Decision behavior. See
[condition probe](docs/pretrained_reuse/R2_TEXT_CONDITION_PROBE_20261012.md).
No additional training or product promotion followed.

### Full-coverage conditional repair (2026-10-12)

The fixed 2,048-update run completed with 16,384 distinct questions and exact
best reload. Best mean-NLL checkpoint was step768. Curriculum Accuracy changed
24.37% at512 to23.14% at2048; original Accuracy57.83% to57.28%. Slight NLL gains
did not establish useful conditional reasoning. No95% claim or product promotion.
See [full curve and uncertainty](docs/pretrained_reuse/R2_CONDITIONAL_FULL_COVERAGE_20261012.md).

### Encoder-coverage comparison (2026-10-12)

Updating all six existing22.7M-model encoder blocks, with embeddings/pooler frozen,
completed the same2,048-update protocol. Curriculum Accuracy80.57% versus23.14%
for final-block-only; original Accuracy60.24% versus57.28%. Best reload and source
hash checks passed. This is synthetic development evidence; count/conditional
selection remain weak and no four-modality95% claim follows. See
[full curve and controlled comparison](docs/pretrained_reuse/R2_ENCODER_COVERAGE_20261012.md).

### Transfer limitation (2026-10-12)

The selected encoder-coverage weights scored33.59% on the unchanged known-failure
256-question diagnostic, with10/128 fully correct opposite-condition pairs.
Thus80.57% on the training generator's validation does not establish general
Decision reasoning. No truncation occurred; no prompt or checkpoint search followed.
See [transfer evidence](docs/pretrained_reuse/R2_ENCODER_TRANSFER_20261012.md).

### Controlled transfer diagnosis (2026-10-12)

Fixed-weight, paired128-state comparisons found79.79% Accuracy in control,
25.88% after an order-preserving numeric-range shift,54.39% after paraphrasing,
and22.95% with all five perturbations combined. Identifier/field renaming had
small effects. This supports range/wording fragility under the tested conditions;
it does not isolate candidate-count effects or establish broad multimodal quality.
See [all arms, task metrics and uncertainty](docs/pretrained_reuse/R2_TRANSFER_CONTROLS_20261012.md).

### Numeric-range coverage result (2026-10-12)

The numeric-only augmentation run completed2,048updates and best reload. On the
predeclared new transfer set, selected-model Accuracy24.27% versus prior30.66%;
the confidence loss improved but useful reasoning did not. Original curriculum
Accuracy fell80.57%→24.51% at the same budget. No promotion or95%claim. See
[fixed-budget range experiment](docs/pretrained_reuse/R2_NUMERIC_RANGE_20261012.md).


The learned-initialization numeric-range control completed2,048updates: selected
narrow-curriculum validation82.23%, final expanded-range training fit83.50%,
with worse original-text NLL/calibration and extra inherited compute. No new
transfer/audit or multimodal quality claim. See
[record](docs/pretrained_reuse/R2_RANGE_WARMSTART_20261012.md).


A fixed selected-weight comparison on new synthetic states found range
interpolation improvement24.12%→71.00%, but wording regression56.54%→42.09%
and poor extrapolation calibration. No general-model promotion.
[Results](docs/pretrained_reuse/R2_RANGE_GENERALIZATION_RESULTS_20261012.md).


Lexical-range control completed2,048updates (selected1408); wording transfer
remains unmeasured for that checkpoint. Separately, frozen TinyCLIP scored48.19%
macro on CLEVR-4 vs9.81% with rotated images; count/texture remain weak.
[Training](docs/pretrained_reuse/R2_LEXICAL_RANGE_RESULTS_20261012.md) /
[image grounding](docs/pretrained_reuse/R2_TINYCLIP_CLEVR4_RESULTS_20261012.md).


The fixed lexical-transfer follow-up found only a small change on its novel
minimum/maximum wording (80.71%→81.05%), with numeric-calibration trade-offs.
It does not repair the previously observed smallest/largest failure by evidence.
[Results](docs/pretrained_reuse/R2_LEXICAL_TRANSFER_RESULTS_20261012.md).


A paired factorial text diagnosis confirms substantial comparison-word
sensitivity at fixed state/range/structure. Lexical training helps but leaves
smallest/largest far below minimum/maximum. No general-quality claim.
[Diagnosis](docs/pretrained_reuse/R2_COMPARATIVE_WORD_RESULTS_20261012.md).


Comparison-vocabulary training completed2,048updates (selected1664). On fixed new
states/templates, smallest/largest improves59.96%→82.81%, but domain terms remain
40.04% (expanded range34.08%). General four-modality95% remains unmet.
[Training](docs/pretrained_reuse/R2_COMPARISON_VOCABULARY_RESULTS_20261012.md) /
[transfer](docs/pretrained_reuse/R2_VOCABULARY_TRANSFER_RESULTS_20261012.md).
