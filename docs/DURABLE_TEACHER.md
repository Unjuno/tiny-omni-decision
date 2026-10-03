# Durable high-precision Decision Teacher

## Result and scope

The first reproducible four-modality Decision Teacher was trained locally on the
pinned Gemma 4 E2B base. It uses the frozen train/validation/evaluation corpus
below, a 256-step decoder LoRA run, and a validation-selected adapter. The final
held-out evaluation uses the selected adapter reloaded from disk and evaluates
the full 2,917-record evaluation split. It is a small controlled teacher run,
not a benchmark-generalizing quality claim.

No Decision LoRA merge, ternary conversion, teacher-logit cache, or Recovery
LoRA was performed. The intended later export path is documented at the end.

## Frozen corpus and leakage checks

Corpus seed is `17`. Source row selection is deterministic reservoir sampling;
option order is deterministically shuffled per sample. The corpus manifest
records the catalog revisions, source-file hashes, and selected media hashes.

| Split | Records | Text | Image | Audio | Video | JSONL SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| Train | 4,859 | 2,875 | 256 | 1,024 | 704 | `c3e6796a2df47471d42e671272b3050f9db150ac6b7ef7289e3902253f36d84d` |
| Validation | 2,901 | 1,397 | 128 | 1,024 | 352 | `afa9232ec60980de28f846654beac4f3b297c394ac79093fc2d1ffae57d3faf7` |
| Evaluation | 2,917 | 1,413 | 128 | 1,024 | 352 | `19812d5211a45e8eedee2aa0c6b36795f613d8a1f8e0592c881edfabd13a2d9d` |

The train/validation/evaluation pair SHA-256 is
`ed178fbc300c95794e17e5216726c499a46b7ee29f113b24124ee402d617d316`.
All three pairwise checks report zero shared source IDs and zero shared
normalized-content fingerprints. The corpus manifest records the pinned base
model manifest SHA-256 as `manifest_sha256`:
`3073b756114904fbec8b8b590c087484aa6769dc510f8f95939509e6657fbfca`. The raw
local `corpus-manifest.json` file SHA-256 recorded by the run is
`a0eecdbfb28dd92e31648a7ce7763a3156d99a950abc94d920ea969bb0400b03`.
A second freeze produced byte-identical train, validation, and evaluation JSONL
files. The local JSON manifest also records operational download counters.

| Source | Train | Validation | Evaluation |
|---|---:|---:|---:|
| Typed Decisions Synth | 898 | 445 | 425 |
| Open-Jev | 1,977 | 952 | 988 |
| Clevr-4 | 256 | 128 | 128 |
| Speech Commands v0.02 | 1,024 | 1,024 | 1,024 |
| CLEVRER | 704 | 352 | 352 |

Open-Jev validation and Speech Commands validation are selection-only;
their official test splits are used for final evaluation. The other sources
use deterministic disjoint partitions. MMAU, MVBench, and OneJev are excluded.
All selected image, audio, and video members were materialized and hashed;
the frozen manifest records 64 train images, 1,024 train audio clips, and 64
train videos, plus 64 validation and 64 evaluation images, 1,024 audio clips
per held-out split, and 64 videos per held-out split.

The checked-in corpus definitions are
[`manifests/durable-training-corpus.yaml`](../manifests/durable-training-corpus.yaml)
and [`manifests/durable-heldout-corpus.yaml`](../manifests/durable-heldout-corpus.yaml).
Generated JSONL and media live under ignored `data/processed/durable-teacher-v0/`.

## Base model and training

- Model and processor: `google/gemma-4-E2B-it-qat-q4_0-unquantized`.
- Immutable revision: `6befbaca7398925921802abd1f277b495b78b738`.
- Hardware: NVIDIA GeForce RTX 3080 Laptop GPU, 16,384 MiB VRAM.
- Environment: Python 3.11, PyTorch 2.6.0+cu124, Transformers 5.6.2,
  PEFT 0.21.2, Accelerate 1.15.0.
- Adapter: decoder LoRA rank 16, alpha 32, dropout 0.05; the base,
  modality encoders, and projector remain frozen.
- Objective: cross entropy plus `0.2 ×` Brier loss; seed `17`.
- Schedule: 256 optimizer steps, gradient accumulation 4, maximum sequence
  length 1,024, learning rate `5e-5`, and maximum gradient norm 1.0.
- Every optimizer step consumes one microbatch from each modality. The saved
  trainer state records 256 audio, 256 image, 256 video, 128 Open-Jev text,
  and 128 Typed Synth text examples consumed (1,024 microbatches total).
- The selected checkpoint is step 256. Its 256-example selection subset has
  accuracy `0.6602`, NLL `0.8288`, Brier `0.4066`, ECE `0.0490`, and mean
  confidence `0.6721`; the same subset's untouched-base values are
  `0.3555`, `3.0235`, `0.9686`, `0.3703`, and `0.7258` respectively.
- Checkpoint save/reload is checked by requiring identical validation option
  probabilities. The run also checks the frozen-base/trainable-LoRA gradient
  invariants and records optimizer/checkpoint state.
- Run duration: 3,670.344 seconds (about 61 minutes), including baseline,
  validation, training, and final evaluation. Peak allocated VRAM was
  11,655,203,840 bytes (about 10.85 GiB); this ran on a local workstation, so
  rental cost was `$0`.
- Base weights SHA-256:
  `33fe0cece08fb527ffefbd1a3a9ce73bd71073727993a283506293e5c6bf0137`.

## Full held-out evaluation

Baseline and selected-teacher metrics below are computed on the same complete
2,917 evaluation records. ECE uses 15 equal-width top-label confidence bins.
These metrics are computed on the same 2,917 examples from
`baseline-evaluation.json` and `decision-teacher-evaluation.json`. Positive
accuracy deltas and negative NLL/Brier/ECE deltas indicate improvement.

| Metric | Untouched base | Decision Teacher |
|---|---:|---:|
| Accuracy | 0.5235 | 0.7343 |
| NLL | 2.0083 | 0.6380 |
| Brier | 0.7407 | 0.3344 |
| ECE | 0.2402 | 0.0344 |
| Mean confidence | 0.7600 | 0.7501 |

Teacher minus baseline deltas are `+0.2108` accuracy, `-1.3704` NLL,
`-0.4062` Brier, and `-0.2058` ECE. All four modalities improve on these
metrics; modality-specific results are:

| Modality | N | Accuracy base → teacher | NLL base → teacher | Brier base → teacher | ECE base → teacher |
|---|---:|---:|---:|---:|---:|
| Audio | 1,024 | 0.7041 → 0.9492 | 1.2271 → 0.1854 | 0.4289 → 0.0644 | 0.0968 → 0.0295 |
| Image | 128 | 0.0859 → 0.5703 | 4.4938 → 1.3744 | 1.2805 → 0.5650 | 0.5541 → 0.0529 |
| Text | 1,413 | 0.4904 → 0.6723 | 1.9047 → 0.7323 | 0.8285 → 0.4232 | 0.3391 → 0.0442 |
| Video | 352 | 0.2898 → 0.4176 | 3.7932 → 1.3080 | 1.0984 → 0.6798 | 0.4788 → 0.0589 |

Per-source results:

| Source | N | Accuracy base → teacher | NLL base → teacher | Brier base → teacher | ECE base → teacher |
|---|---:|---:|---:|---:|---:|
| Speech Commands | 1,024 | 0.7041 → 0.9492 | 1.2271 → 0.1854 | 0.4289 → 0.0644 | 0.0968 → 0.0295 |
| Clevr-4 | 128 | 0.0859 → 0.5703 | 4.4938 → 1.3744 | 1.2805 → 0.5650 | 0.5541 → 0.0529 |
| CLEVRER | 352 | 0.2898 → 0.4176 | 3.7932 → 1.3080 | 1.0984 → 0.6798 | 0.4788 → 0.0589 |
| Open-Jev | 988 | 0.5111 → 0.6356 | 1.6324 → 0.7817 | 0.7557 → 0.4594 | 0.2801 → 0.0508 |
| Typed Decisions Synth | 425 | 0.4424 → 0.7576 | 2.5379 → 0.6175 | 0.9978 → 0.3392 | 0.4825 → 0.0558 |

The path-sanitized checked-in record is
[manifests/teachers/durable-teacher-v0.json](../manifests/teachers/durable-teacher-v0.json).
The local `teacher-manifest.json` and `run-metadata.json` record the source and
modality metrics, hashes, dynamically resolved target module paths,
configuration, environment, optimizer-step count, and artifact checksums.
The raw Teacher prediction file has exactly 2,917 records. Generated JSONL,
checkpoints, predictions, and model weights are intentionally ignored by Git;
the durable corpus manifests and this report are checked in.

## Reproduction

Build the frozen corpus (source/media access required), then run:

```bash
tiny-omni-decision freeze-corpus \
  manifests/durable-training-corpus.yaml \
  manifests/durable-heldout-corpus.yaml \
  data/processed/durable-teacher-v0 \
  --max-records-per-source 2048

tiny-omni-decision train-decision \
  --train-manifest data/processed/durable-teacher-v0/train.jsonl \
  --eval-manifest data/processed/durable-teacher-v0/eval.jsonl \
  --validation-manifest data/processed/durable-teacher-v0/validation.jsonl \
  --config configs/decision/durable_teacher.yaml \
  --output artifacts/tiny-omni-decision-teacher-v0
```

The exact local corpus can instead be re-used by verifying the three JSONL
hashes above and its generated `corpus-manifest.json`. The run's raw predictions
and checkpoints are local ignored artifacts, not source inputs.

## Later merge/export path (not run)

For a future task-adapted high-precision export, load the exact pinned base and
processor, load the selected PEFT adapter, merge it into the base, then save the
merged model and processor together. The expected PEFT path is:

```python
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "google/gemma-4-E2B-it-qat-q4_0-unquantized"
revision = "6befbaca7398925921802abd1f277b495b78b738"
base = AutoModelForMultimodalLM.from_pretrained(model_id, revision=revision)
processor = AutoProcessor.from_pretrained(model_id, revision=revision)
adapted = PeftModel.from_pretrained(base, "artifacts/tiny-omni-decision-teacher-v0/best")
merged = adapted.merge_and_unload(safe_merge=True)
merged.save_pretrained("artifacts/task-adapted-teacher-merged", safe_serialization=True)
processor.save_pretrained("artifacts/task-adapted-teacher-merged")
```

This export snippet is documented but has not been executed. Validate the
merged model against the selected adapter before using it as a ternary
quantization source. Ternary conversion and Recovery LoRA remain later phases.
