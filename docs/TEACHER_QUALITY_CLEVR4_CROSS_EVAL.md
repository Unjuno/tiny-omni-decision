# Teacher Quality Cross-Evaluation on Fresh Clevr-4 Development Data

## Purpose and status

This is a read-only cross-evaluation of the selected checkpoint from the
seed-17 clean-dev-v2 long-budget run on a separate Clevr-4 development
validation. It asks whether the candidate's image performance transfers to
new image identities beyond the 512-example clean-dev-v2 selector. It is not a
new training run, a model-selection gate, or a final audit. The Clevr-4
development validation had already been evaluated with Teacher v1; it is now
observed development data and must never be described as blind or sealed.

No training configuration, corpus, optimizer, sampler, checkpoint, Teacher v1
artifact, or sealed audit was changed or loaded.

## Candidate and data provenance

- Candidate: `teacher-v2-clean-dev-v2-long-budget-step1536-clevr4-dev-v1`.
- Parent run: seed-17 clean-dev-v2 E-long, exact continuation from verified
  step-1,408 state; selected step 1,536. The run itself completed 2,048
  updates, but its separate final-result serialization failed as recorded in
  `docs/VIDEO_TEACHER_V2_LONG_BUDGET.md`.
- Candidate checkpoint: adapter SHA-256
  `0497a25742451daeb2e08c7879dc221d174a17bf451e4ab2de0f3af6223b5009`.
- Candidate training source commit: `c3b7f187b90cdaaf9538214011a8e890294508b4`.
- Evaluation code commit: `287301d609621e0a61600be30472896dff6f9d48`.
- Base and processor: `google/gemma-4-E2B-it-qat-q4_0-unquantized`, revision
  `6befbaca7398925921802abd1f277b495b78b738`.
- Evaluation config: `configs/decision/teacher_quality_clean_dev_v2.yaml`,
  SHA-256 `4ce9af8d0aa83c5526d3d3828b77df3a5cd727351f912ed58cd9aca8c7e1773a`.
  Configured and effective maximum sequence length were both 1,024.
- Candidate train corpus SHA-256:
  `2ebf1e75ed91b6e7021b5b202fce9ddd3cfa9feb3a9fec0cce17d3623cd4aaea`
  (98,188 rows).
- Candidate selector validation SHA-256:
  `cad3fe4f87e64fcc4b908ae6ab59ae6973cbe32093551956bb3164a49c6e8edf`
  (7,901 rows).
- Clevr-4 development validation SHA-256:
  `73c5fa8de932877750614d73542630937f90a297e419538ffe813374778d098d`
  (3,708 decisions over 927 unique images; four attribute questions per image).
- Clevr-4 sample-ID order SHA-256:
  `5dbeef2b479d65a20e23ba1100211faeb026657f9c1cc54a7f22655add026d1e`.

The repository's `check_train_eval_splits` gate was run against the actual
JSONL files. Clevr-4 validation had zero shared source IDs, source assets,
media identities, or normalized content fingerprints with the E-long train
corpus. It also had zero overlap by all four checks with E-long's selector
validation. The training corpus itself remains frozen.

## Evaluation procedure

The candidate adapter was loaded from its saved `best/` directory with
`PeftModel.from_pretrained(..., is_trainable=False)` and evaluated on all 3,708
validation decisions using the existing inference/evaluation path. Teacher v1
was evaluated earlier on these exact same IDs, order, and preprocessing, so
the comparison pairs predictions by sample ID. No rows or answer options were
removed.

Command (PowerShell, from the repository root):

```powershell
$env:PYTHONPATH='src'
python scripts/evaluate_quality_candidate.py `
  --corpus data/processed/teacher-quality-next/clevr4/validation.jsonl `
  --adapter 'C:\Users\junny\AppData\Local\CodexArtifacts\tiny-omni-decision-teacher-v2\clean-dev-v2-seed17-2048-resume-step1408-exact\best' `
  --config configs/decision/teacher_quality_clean_dev_v2.yaml `
  --output artifacts/teacher-quality-next/evaluations/teacher-v2-clean-dev-v2-long-budget-step1536-clevr4-dev-v1 `
  --candidate-id teacher-v2-clean-dev-v2-long-budget-step1536-clevr4-dev-v1 `
  --modalities image
```

The initial invocation without `PYTHONPATH=src` stopped before loading model
weights with `ModuleNotFoundError`; the corrected invocation above completed.
Evaluation took 2,439.313 seconds and peak allocated VRAM was 11,117,007,360
bytes on the local RTX 3080 Laptop GPU. The model-evaluation output directory
is local and Git-ignored.

## Results

| Candidate | N | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Teacher v1 reference | 3,708 | 0.7201 | 0.7937 | 0.3822 | 0.0880 | 0.8073 |
| E-long selected step 1,536 | 3,708 | 0.8015 | 0.5729 | 0.2782 | 0.0831 | 0.8843 |
| Candidate minus v1 | — | +0.0814 | -0.2208 | -0.1040 | -0.0049 | +0.0771 |

This is an 8.14 percentage-point Accuracy increase on this development source,
but the candidate remains 9.85 points below the 90% per-modality goal.

### Image-identity paired uncertainty

Predictions were paired by sample ID and grouped by the underlying
`task_group_id` image identity. A cluster bootstrap resampled 927 images with
replacement for 10,000 replicates using `random.Random(17)`. The E-long minus
v1 Accuracy delta was +0.08145, with a percentile 95% interval of
[+0.06796, +0.09466]. There were 516 decisions corrected by E-long and 214
regressed; 2,456 were correct for both and 522 were wrong for both. This is a
descriptive development-set uncertainty interval, not an independent audit or
a correction for prior checkpoint-selection effects elsewhere.

### Clevr-4 task slices

| Question type (n=927 each) | Teacher v1 Accuracy / NLL / Brier / ECE | E-long Accuracy / NLL / Brier / ECE |
|---|---|---|
| Color | 0.8350 / 0.4797 / 0.2568 / 0.0692 | 0.8943 / 0.3137 / 0.1682 / 0.0713 |
| Texture | 0.5771 / 1.4686 / 0.6385 / 0.2130 | 0.7379 / 0.8701 / 0.3899 / 0.1348 |
| Count | 0.5016 / 1.1234 / 0.5849 / 0.0945 | 0.5987 / 1.0204 / 0.5161 / 0.1132 |
| Shape | 0.9666 / 0.1032 / 0.0488 / 0.0196 | 0.9752 / 0.0875 / 0.0387 / 0.0189 |

The task breakdown suggests that the single-source Image gap is not uniform:
shape is already strong, color approaches 90%, while texture and especially
count remain weak. The benchmark covers synthetic Clevr-4 attributes; it does
not establish generalization to natural images or broader visual decisions.

## Interpretation and next decision

The selected E-long candidate is better than Teacher v1 on this disjoint
development validation under all four aggregate metrics, and the paired
image-cluster interval supports a real improvement on this source. This is
useful evidence that the candidate's Image weakness is not simply fixed at
Teacher v1's 72%. It is not evidence that the four-modality goal is reached:
Image is still at 80.15% overall, Text and Video remain far below 90% on the
candidate's selector, and no fresh final audit was opened.

The 4,318 unique Clevr-4 training images available in the rights-approved
candidate corpus remain a meaningful data-coverage opportunity. Future image
work should prioritize that unique-image coverage and the texture/count task
gap, while preserving the full four-modality evaluation. Reusing this
Clevr-4 validation for further selection is allowed only as observed
development evidence and should be counted toward validation-experiment
exposure. It cannot serve as a final audit.

The sealed Teacher v1 audit was not loaded. No ternary conversion, Recovery
LoRA, training, checkpoint modification, or cloud compute was performed.

## Output hashes

- Metrics JSON SHA-256:
  `35a0da103bb0416ccf73e6d00a6b5fdd5ef8ba2405e77b4972c3db5746fca5fe`.
- Predictions JSONL SHA-256:
  `1e0ff37a3b212bb561fdc70a1ff6c91c07fec35d2368e5122f8041e56590437b`.
- Local output directory:
  `artifacts/teacher-quality-next/evaluations/teacher-v2-clean-dev-v2-long-budget-step1536-clevr4-dev-v1/`.
