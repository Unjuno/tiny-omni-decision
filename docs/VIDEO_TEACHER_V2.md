# Video Teacher v2 experiment log

## Status

Candidate A's configuration and frame-count plumbing are implemented on
`codex/video-teacher-v2`. The processor-only smoke confirmed eight decoded
frames reach the pinned Gemma 4 processor and remain within the configured
sequence limit. Candidate A has **not been trained yet**. Candidate B, dataset
expansion, ternary conversion, and sealed-audit evaluation have not started.

Teacher v1 remains the fixed reference. Its selected run used
`configs/decision/teacher_v1_weak_modalities.yaml`; SHA-256
`a68fae06267078dbbbcc8afc0ce4680cb553ed28b968e64e2bde50955c5a6409` still
matches both the Teacher v1 manifest and
`artifacts/tiny-omni-decision-teacher-v1/selected/training-config.yaml`. The
original config, train/validation corpus, selected adapter, learning curves,
selection lock, and sealed-audit files were not edited or reloaded. A new
`teacher_v1_explicit_reference.yaml` spells out the effective v1 runtime values
for comparison without changing the historical run bytes.

## Candidate A

| Setting | Value |
|---|---|
| Artifact ID | `tiny-omni-decision-teacher-v2-candidate-a` |
| Config | `configs/decision/teacher_v2_candidate_a.yaml` |
| Config SHA-256 | `9eea882d00f10746a0a1190c1dfac67b8ed7f213127fb52f98d1dd5d0e97a81d` |
| Video frames | 8 |
| LoRA policy / rank | q/v / 16 |
| Learning rate | fixed `5e-5` (same as v1) |
| Training data | frozen Teacher v1 train JSONL |
| Validation data | frozen Teacher v1 validation JSONL; same selection code |
| Train SHA-256 | `5a9d8f3c1608b9cedef0c4c4b1513dcf5e1face4a67acc99c8002849cf07fb2b` |
| Validation SHA-256 | `ebfa46399f3139f4c97a6e126b3036e00afc274de9433fdb5755c089bb9995a6` |
| Base / processor revision | `google/gemma-4-E2B-it-qat-q4_0-unquantized` at `6befbaca7398925921802abd1f277b495b78b738` |

### Processor smoke

The smoke used one existing **training** video (`MIT-IBM/CLEVRER:638:0`, media
SHA-256 `7d4255a20ce41be2c775ff03e1d2a8d4741e19f44abf488a7ed5291af99b4c2f`) and
the local pinned processor. It did not load model weights, train, or touch
validation or audit records.

```text
video_num_frames from Candidate A config: 8
pixel_values_videos: [1, 8, 630, 768]
video_position_ids: [1, 8, 630, 2]
input_ids: [1, 610]
configured maximum sequence length: 1024
result: 610 <= 1024 (pass)
processor time after load: 1.313 s
```

The local processor emitted a warning that `torchcodec` is not installed and
fell back to the available `torchvision` decoder. The fallback produced all
eight frames successfully; its deprecation warning is retained as an
environment limitation. The machine is the local RTX 3080 Laptop 16 GiB, but
this processor-only smoke used CPU and consumed no GPU training time.

The detailed smoke record is stored locally at
`artifacts/tiny-omni-decision-teacher-v2/candidate-a-smoke/processor-smoke.json`
and is ignored by Git, like model weights and other generated artifacts.

## Verification so far

- `ruff check src tests`: passed.
- Full CPU test suite: **98 passed**.
- Candidate A config regression test: changing only `video_num_frames` from 4
  to 8 leaves the effective training configuration unchanged otherwise.
- Recording processor test: receives `videos_kwargs.num_frames == 8`.
- Real processor smoke: produced an eight-frame tensor and stayed below the
  sequence-length guard.

## Next gates

Train Candidate A first in a new candidate-specific output directory, starting
with a 512-step measurement and extending only if validation continues to
improve. Use seed 17, the same frozen v1 training and validation files, and
the v1 selected adapter as an explicitly identified validation reference.
Record best checkpoint, full history/config/data hashes, all modality metrics,
sample/asset accounting, peak VRAM, and wall time. Do not read the sealed audit.

Only after Candidate A has a measured validation result should Candidate B be
trained. A and B will hold the fixed `5e-5` schedule constant so LoRA target
coverage is the changed factor. Cosine scheduling with 3% warmup is reserved
for a separately identified schedule experiment.
