# BSD10k Path A audio Decision experiment

This is a narrow, source-disjoint development experiment for broad sound
classification. It adds a real audio task with manual labels and uploader
provenance; it does not claim general audio understanding or multimodal
quality. Source admission and its per-file rights, label caveats, and split
hashes are recorded in [the source audit](R2_BSD10K_SOURCE_AUDIT_20261128.md).

## Fixed experiment

- `configs/decision/bsd10k_audio_path_a_seed17.yaml` freezes the source,
  license allowlist, encoder revision, feature windows, optimization, selector,
  and calibration partition.
- The only encoder is frozen EfficientAT DyMN-0.4. It receives audio samples,
  not labels or textual metadata. Each WAV is resampled to 32 kHz, represented
  by 2-second windows at a 1-second hop, mean pooled, and L2 normalized.
- A linear 384-to-23 `TextFreeAudioReadout` is trained. The model exposes
  probabilities for a caller-supplied subset/order of the 23 class IDs in the
  existing typed JSON contract.
- 20% of train uploader groups, deterministically selected with seed 17, are
  reserved for temperature fitting. Validation is used for early stopping and
  best-checkpoint selection by macro NLL, never for calibration. The test
  split is not loaded by the training script.
- Confidence scores are retained as reporting strata; they do not weight the
  loss or filter examples. Titles, tags, descriptions, and bundled CLAP
  embeddings are excluded.
- The feature receipt records original clip duration. Validation reports
  Accuracy, NLL, Brier, and ECE by fixed duration bands (`<0.5s`, `0.5–2s`,
  `>=2s`) with example and uploader-group counts. These are diagnostic only;
  duration does not change training, calibration, or checkpoint selection.
- Feature and readout receipts include Python, PyTorch, torchaudio, platform,
  CPU model, and logical CPU count so CPU-only throughput can be reproduced.
- The readout run report also records train/validation examples and distinct
  uploader groups per class, plus validation confidence-stratum group counts,
  so low independent support is visible beside every class-level result.

## Data integrity safeguards

The media downloader enforces the pinned metadata hash and exact per-file
CC-BY-4.0/CC0 allowlist. It reads local ZIP headers and compressed payload
ranges only for selected files and checks ZIP size/CRC before writing. The
download audit verifies split membership, file SHA-256/CRC, WAV format, and
cross-split duplicate identity both on the file bytes and normalized PCM.
An initial whole-split-in-memory implementation was stopped before any full
split write after measuring only 2.9 GiB free system RAM; the downloader was
changed to flush each bounded response batch. The resumed process currently
keeps its working set below 100 MiB.

## Execution state

The following stages are committed/pushed and CI checked:

- Metadata/license preflight and deterministic seed-17 uploader-group split.
- Range-only archive inventory and 25-file WAV smoke.
- Bounded-memory selective downloader and exact split/content audit.
- Pinned CPU-only feature smoke on eight already-downloaded train WAVs: the
  extractor produced an 8x384 finite feature matrix in 0.52 seconds; labels
  and metrics were not loaded/computed, and the EfficientAT state SHA-256 was
  identical before and after extraction. A separate 75-file run extracted
  609 windows in 8.27 seconds (about 9 records/second, excluding model load).
  Receipts are stored outside Git under
  `C:\CodexArtifacts\pretrained-reuse-r2-local\bsd10k-v1.2-source-audit-20261128-attempt-01`.
- The shortest clip in the partial train acquisition was 0.0156 seconds. It
  passed the actual CPU preprocessing/model path as one right-padded 2-second
  window and produced finite 384-dimensional features; the pooled norm was
  1.8375 and the model-state hash did not change. No label or metric was read.
  This checks runtime handling only; the clip remains in the corpus and the
  eventual duration-stratified validation will determine whether such short
  examples are a task-quality limitation.
- The 2026-11-28 local measurement used Windows 11 Home (build 26300), Intel
  Core i7-12700H (14 cores / 20 logical CPUs; PyTorch intra-op threads set to
  4), Python
  3.11.9, PyTorch/torchaudio 2.6.0+cu124, NumPy 2.4.6, FP32 and CPU inference.
  The RTX 3080 Laptop GPU was 8,270 MiB / 16,384 MiB allocated at 81% utilization
  by another process during inspection; this experiment did not use the GPU.
- Unit tests for allowlist behavior, uploader separation, duplicate PCM
  rejection, fixed audio windows, frozen feature receipts, calibration, and
  save/reload.

The full train download is running to an external `C:\CodexArtifacts` directory
with SHA-256 and CRC receipts. Validation download, full feature extraction,
readout fitting, measured task metrics, and the one-shot custom test remain
unfinished. No sealed audit or DCASE evaluation data has been used.

The extractor suppresses the pinned upstream model's architecture dump while
loading it; upstream deprecation warnings remain visible. The downloader
retries bounded HTTP 429/5xx failures while retaining exact
range, ZIP CRC, per-file hash, and 2 GiB free-space checks. A CI failure on the
optional ML test was traced to PyTorch not being installed in the repository's
standard `.[dev]` CI environment; that test now skips when the optional PyTorch
dependency is absent. Local full tests pass, and CI is green on commit
`f0819013`.

When the held-out metrics are available, they will be reported with per-class
counts and grouped uncertainty. Confidence or uploader imbalance will be
treated as part of the data limitation, not hidden by aggregate accuracy.

