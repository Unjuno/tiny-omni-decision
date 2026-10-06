# Ternary-first student implementation

This is an experimental PyTorch reference implementation of the approved flow:
pretrained EmbeddingGemma 2 -> ternary -> Teacher-supervised constrained recovery
-> one frozen-base Recovery LoRA only if the validation quality gate fails.
The active Teacher trainer, its configuration and its checkpoints are unchanged.
No classifier surgery or extra teacher is introduced.

## What is implemented and what is not

Implemented: option-bound Teacher caches, generic matrix/embedding/convolution
weight parametrization, straight-through ternary QAT, option KL + CE + Brier,
validation checkpoint selection, conditional LoRA, integrity-checked export and
reload, and train/validation/final-evaluation separation.

Tested locally on CPU with small differentiable fixtures. Full EmbeddingGemma 2
loading, multimodal forward/backward, GPU memory fit and quality recovery have
**not** been validated by those tests. A failing experiment remains FAIL.
The runtime checks the pinned architecture and actual tensor inventory; shared
quantization-target parameters and unsupported inputs fail explicitly.

The first implementation keeps native pooling and uses cosine scores between the
question/context embedding and the supplied option embeddings, with a fixed score
temperature. This is the minimal decision readout, not a claim that similarity
scores are calibrated probabilities. The Teacher supplies supervision for that
calibration. Student preprocessing uses Sentence Transformers' native defaults;
the existing Teacher video bridge uses four frames. Record the library versions,
and keep the student's preprocessing fixed before and after quantization.

## Environments and offline smoke

Keep the existing Teacher environment unchanged. Use Python 3.11 or 3.12 in a
separate student environment. From the repository root, after installing the
appropriate PyTorch wheel for your device:

```bash
python -m venv .venv-student
# Linux/macOS: source .venv-student/bin/activate
# Windows PowerShell: .venv-student\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pip install -r requirements-student.txt
python -m tiny_omni_decision.student smoke --output artifacts/student-smoke-001
```

`smoke` downloads no model, runs three QAT and three LoRA steps, and checks packed
reload. Its deliberately unattainable quality gate returns FAIL while successful
pipeline execution exits zero. It is not a benchmark. `train` also exits zero
when a completed experiment fails its acceptance gate: inspect `result.json`.
Malformed inputs or runtime errors exit with code 2.

All output paths must be new. No command overwrites an existing artifact or
starts training implicitly. Model loads use local caches unless
`--allow-download` is explicitly supplied. The `pin` command accesses upstream
metadata and configuration but does not download model weights.

## Freeze and cache the Teacher

Only after the current Teacher has finished, been selected, reloaded and frozen,
run `cache-teacher` in the **existing Teacher environment**. Use the actual frozen
corpus and adapter paths; the paths below are placeholders, not a request to reuse
an older corpus. The command reads but does not update the adapter.

```bash
python -m tiny_omni_decision.student cache-teacher --examples TRAIN.jsonl --adapter FROZEN_TEACHER/best --data-root DATA_ROOT --role train --output artifacts/student-input/train.jsonl
python -m tiny_omni_decision.student cache-teacher --examples VALIDATION.jsonl --adapter FROZEN_TEACHER/best --data-root DATA_ROOT --role validation --output artifacts/student-input/validation.jsonl
```

Keep `--base-manifest` (default `manifests/base-model.example.yaml`),
`--max-length`, adapter and processing environment identical across caches.
`cache-teacher` uses the existing Gemma 4 input path and training-license check.
It binds each row to exact options/order, source group, normalized content and
local media hashes. It detects changes to the adapter/source while caching.
Teacher token IDs are never treated as student token IDs. Existing prediction
JSONL files lacking these bindings are not silently accepted as training caches.

## Pin, inspect, then run the bounded student experiment

In the separate student environment:

```bash
python -m tiny_omni_decision.student pin --output artifacts/student-input/student-pin.json
python -m tiny_omni_decision.student inspect --pin artifacts/student-input/student-pin.json --config configs/student/ternary_first.example.json --data-root DATA_ROOT --allow-download --output artifacts/student-inventory.json
python -m tiny_omni_decision.student train --pin artifacts/student-input/student-pin.json --config configs/student/ternary_first.example.json --data-root DATA_ROOT --train-cache artifacts/student-input/train.jsonl --validation-cache artifacts/student-input/validation.jsonl --output artifacts/student-run-001
```

The example config specifies 128 steps per stage and illustrative, editable
validation tolerances. They are **not measured performance claims or agreed
product requirements**. Set the acceptance limits and bounded budget before
looking at results. The gate checks overall and each modality's Accuracy, NLL,
Brier and ECE. Checkpoint selection uses only validation. No final-evaluation
cache is accepted by `train`.

`inspect` inventories actual parameters rather than guessing module names.
By default all floating rank-2-or-higher weight tensors are targets, including
text embeddings, vision and audio. Explicit exclusions must be exact existing
parameter names. Their dtype and bytes are reported. Biases, norms and buffers
are retained. No part of the model is silently described as ternary if excluded.

The reference trainer uses FP32 master parameters and FP32 execution. It retains
high-precision shadows/optimizer state, so it does **not** claim 1.6-bit training
memory or guaranteed fit on a 16 GB GPU. Start with a bounded local trial; reduce
input lengths only deliberately. Overlength input is rejected, not silently
truncated. This implementation has no exact-resume checkpoint or mixed-precision
optimizer yet. Saved bundles are inference/selection artifacts, not resume state.

During QAT, every target tensor uses ternary values in the forward pass. The
straight-through estimator updates the shadows. If quality still fails, the
selected codes/scales and all base parameters are frozen; only one LoRA is
trained. LoRA is not added for a size-only failure, and is not merged into the
ternary codes. The Teacher never changes during either student stage.

## Results, packing and final evaluation

`result.json` records stage history, acceptance failures, whether LoRA was used,
reload error, actual tensor-bundle bytes, and the selected bundle directory.
`max_artifact_bytes` limits **this tensor bundle including its manifest and LoRA**;
it is not an application RAM or complete mobile-package limit.

Exports use safetensors: five ternary codes per byte (1.6 code bits/weight), FP32
group scales, dense exceptions and optional dense LoRA factors. This approaches
but does not equal the ideal ternary coding rate. Metadata and small tensors add
overhead. No claimed 147 MB result is built into the code.

Reload verifies checksums, shapes, dtypes and quantization inventory, and restores
the exact quantized values. **It dequantizes for ordinary PyTorch execution.**
There is no packed ternary compute kernel, GGUF export, mobile runtime, or proven
latency/RAM reduction. Reference LoRA also materializes its dense weight delta.
The bundle requires the same pinned upstream architecture/processor; those assets
are not copied into the tensor bundle. Do not call it a standalone mobile package.

After freezing the chosen result, create the evaluation cache in the Teacher
environment using `--role evaluation`, then in the student environment:

```bash
python -m tiny_omni_decision.student evaluate --bundle artifacts/student-run-001/SELECTED_BUNDLE --cache artifacts/student-input/evaluation.jsonl --data-root DATA_ROOT --output artifacts/student-final-evaluation.json
```

Replace `SELECTED_BUNDLE` with `result.json`'s `selected` value. Evaluation checks
Teacher identity and rejects overlap with **both** training and selection records,
source groups, content and media hashes recorded in the export. It reports paired
Teacher/student metrics and option logits. Pooling/classifier fallback remains
deferred until constrained recovery plus LoRA have proved insufficient.

## Verification

```bash
python -m pytest tests/test_student_ternary.py tests/test_student_recovery.py tests/test_student_model_cli.py -q
python -m tiny_omni_decision.student --help
```

The separate student CI installs CPU PyTorch and safetensors, runs the complete
repository test suite and lints the additions. It downloads no model weights.
