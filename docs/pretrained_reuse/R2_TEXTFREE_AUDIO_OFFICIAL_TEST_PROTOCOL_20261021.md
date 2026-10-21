# Frozen text-free audio official-test evaluation protocol

Status: **opened; pre-inference integrity gate failed; no predictions or metrics were produced**.

## Attempt record (2026-10-21)

The fixed run launched through HotUGPU after cooperative reservation. The
pinned Parquet shard was loaded, but `build_test_records` stopped before
feature extraction because retained rows had duplicate WAV-byte SHA-256 values
(`duplicate official test audio content`). The run lock is preserved at
`C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-official-test-eval-v1-attempt-01\run-lock.json`.
No logits, predictions, or test metrics were generated; no checkpoint,
temperature, or preprocessing was changed. The official test has been opened,
so this attempt must be disclosed in any later protocol. Do not reuse this
output directory or describe this as a completed evaluation.

## Question

Does the existing text-free frozen-Whisper readout retain its observed
performance on the official, previously unopened Speech Commands v0.02 test
split? This only evaluates the narrow, closed-set, English keyword task. It is
not evidence for general speech understanding or for the whole multimodal
Decision model.

## Frozen source and task

- Source: `google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da`.
- Split: official `test`, manifest `manifests/candidates/speech-commands-test.yaml`.
- Parquet shard: `v0.02/test-00000-of-00001.parquet`, expected SHA-256
  `6446af16af8ef28423c57aac7ec0faf091905e2580afcaca5e7733188b6ee85a`.
- Decision: select one of the fixed ten labels
  `yes/no/up/down/left/right/on/off/stop/go` from audio features only. The
  source's non-keyword labels, including `_unknown_` and `_silence_`, are
  omitted exactly as declared by the candidate manifest. Report the omitted
  count and per-class counts; do not rebalance or subsample the official test.
- The full source shard hash must match before rows are decoded. Every retained
  recording is content-hashed; sample IDs and speaker identifiers are stored
  only as hashes. Test speaker and media hashes must be disjoint from the
  frozen train, validation, and calibration groups.

## Frozen systems and comparison

Run the same Whisper Tiny encoder revision and audio pooling for every system:

| System | Frozen readout | Training/calibration | Test-time temperature |
|---|---|---|---:|
| Legacy baseline | `textfree-audio-decision-03/best-readout.pt`, SHA-256 `46808a882ab75f8f19a138e4a91d13ceb42cfdc82534c4608c42ee615774d3cb` | 1,280 sampled training examples; the already observed development-selected checkpoint | 1.0 |
| Calibration candidate, raw | `textfree-audio-calibration-03/last-readout.pt`, SHA-256 `171f6a9303b2fd9ad63efe2bee6af119549fb4f26ea663e9c1297f51a678e1c3` | 1,016 fit examples; 264 speaker-held-out training examples used only to fit temperature | 1.0 |
| Calibration candidate, fixed temperature | Same exact readout and logits | Same frozen fit/calibration | `0.42702477413437295` |

The calibration report itself is pinned by SHA-256
`16e4396e8d1dc9f0a977dee18d8022f5093bbee0bf963359ed3a66c8a59d8d21`. The
Whisper weights are pinned by SHA-256
`7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`.
Neither readout, temperature, nor preprocessing can be changed after test
rows are read. No test-driven fitting, checkpoint selection, class balancing,
or prompt/search iterations are allowed. The test is intended to be evaluated
once, with a unique output directory whose `run-lock.json` prevents reuse after
an interrupted attempt. This attempt did not reach model inference; any
follow-up requires a documented integrity-policy correction before a new
one-time evaluation, with model and preprocessing choices still frozen.

The candidate and legacy baseline are not a perfectly matched fit: the
calibration candidate reserves 264 training examples for speaker-held-out
temperature fitting, while the legacy readout uses all 1,280 sampled training
examples and selected its checkpoint on the previously inspected 640-row
validation set. Report this training-data difference and do not attribute any
test difference solely to calibration.

## Metrics and execution

Report overall and per-class Accuracy, NLL, multiclass Brier, ECE-15 and mean
confidence for the three fixed outputs. Save each example's hashed source ID,
target, predictions, and probabilities. Report a paired percentile bootstrap
over whole speaker groups (10,000 resamples, seed 17), including 95% intervals
for metrics and raw/calibrated differences versus the legacy baseline. Mark
fewer than 200 independent speakers as low power.

Feature inference uses the local pinned checkpoint and only the RTX 3080 Laptop
already present. The process must be launched through
`scripts/run_with_hotgpu_guard.py`; its child independently verifies the
reservation. If HotUGPU does not cooperatively yield, the child must not start.
No other GPU process is stopped or signaled. No training or cloud compute is
used.

## Reproduction command

Run from the repository root after the implementation commit is pushed and CI
is green. Use a new, empty output path; never retry into an existing path.

```powershell
$env:PYTHONPATH='src;.'
py -3.11 scripts/run_with_hotgpu_guard.py --max-wait-seconds 3600 --reserve-mib 3200 --block-if-free-mib 14000 --min-child-free-mib 8000 -- py -3.11 scripts/evaluate_textfree_audio_official_test.py `
  --calibration-report C:\CodexArtifacts\pretrained-reuse-r2\textfree-audio-calibration-03\run-report.json `
  --checkpoint C:\CodexArtifacts\pretrained-reuse-r2\textfree-audio-calibration-03\last-readout.pt `
  --baseline-checkpoint C:\CodexArtifacts\pretrained-reuse-r2\textfree-audio-decision-03\best-readout.pt `
  --sample-dir C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-sample-seed17-128-64-retry2 `
  --encoder-dir C:\CodexArtifacts\pretrained-reuse-r2\models\audio-whisper-tiny `
  --source-dir C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-official-test-source-v1 `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-official-test-eval-v1-attempt-01
```

This protocol does not open a sealed audit, change any training run, or close
the broader audio, video, image, text, multimodal, quantization, or product
acceptance gates.
