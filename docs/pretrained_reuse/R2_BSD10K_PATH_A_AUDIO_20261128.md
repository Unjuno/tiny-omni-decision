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
- The shortest clip in train was 0.0156 seconds. It
  passed the actual CPU preprocessing/model path as one right-padded 2-second
  window and produced finite 384-dimensional features; the pooled norm was
  1.8375 and the model-state hash did not change. No label or metric was read.
  This checks runtime handling only; the clip remains in the corpus.
- The 2026-11-28 local measurement used Windows 11 Home (build 26300), Intel
  Core i7-12700H (14 cores / 20 logical CPUs; PyTorch intra-op threads set to
  4), Python
  3.11.9, PyTorch/torchaudio 2.6.0+cu124, NumPy 2.4.6, FP32 and CPU inference.
  The RTX 3080 Laptop GPU was 8,270 MiB / 16,384 MiB allocated at 81% utilization
  by another process during inspection; this experiment did not use the GPU.
- Unit tests for allowlist behavior, uploader separation, duplicate PCM
  rejection, fixed audio windows, frozen feature receipts, calibration, and
  save/reload.

The full train download completed with 4,214 unique IDs and 4,595,945,044 PCM
bytes. The external `download-train-result.json` SHA-256 is
`4ae6f01c988ffd2013e848e7938bbc4e2080ad745501553ecbf08449a387b03e`; the
train media audit SHA-256 is
`8a32cb9dd4153657bcfc967f7efceee18c65fe0ccb47ce66acb544d4e085eb51`. The
audit passed all train file-size, SHA-256, CRC-32 and WAV-format checks. It
found zero duplicate file hashes and zero duplicate normalized PCM hashes
within train. The split manifest independently assigns each uploader group
to one split; the completed train/validation media-hash check is recorded
below. The complete split manifest remains pinned by SHA-256
`e1ae3d4088b792619a08a87023e4aeb2b1b5215091ba9772c0dc47fe88842c33`.
The full train content/duration audit external artifact SHA-256 is
`b62cf2011aa15363c9b277fdeb5c6d6eed4f19545c47b2120fdf757e89ab55ef`.
Label-free full-train feature extraction has now completed on CPU in 529.04 s:
4,214 x 384 FP32 features over 49,927 windows, with no labels or metrics
loaded. The feature matrix SHA-256 is
`d01df57b74d862c4220a38d0205601be0216bfccc465ba9e5e91f785a7e49380`, sample
index SHA-256 is
`fceb6946379e132768cc884f4c106fac11858ad2a2c0c16a97f06cb6e2fecc95`, and
run receipt SHA-256 is
`8f08c5892d4d86fc82bae39609803729bb539dfc6aefc85671f9c54297d9d053`. The
runtime used the existing `embeddinggemma2-v518` Python environment with site
initialization disabled to avoid an unrelated Unicode `.pth` startup failure;
no packages were installed or changed. The pinned EfficientAT weights and
source revision matched the manifest.
The 23 classes range from 64 `is-w` examples across 14 uploader groups to 453
`fx-h` examples. Among train clips, 929 (22.1%) are shorter than two seconds,
including 215 (5.1%) shorter than 0.5 seconds; none were filtered. Annotation
confidence counts are 1: 38, 2: 302, 3: 1,326, 4: 2,293, and 5: 255. The
license counts are 2,226 CC-BY-4.0 and 1,988 CC0-1.0. These are explicit data
limitations/strata, not model-quality results.
Validation acquisition completed with 1,436 examples / 1,584,339,402 bytes;
receipt SHA-256 is
`27a1396b47813ea6bf5a76bb5c5a28258c627efdee00c111691161e888d173c6`. The
combined train/validation media audit SHA-256 is
`cd6e65f878a2c8b2e2adfe8b9aa070fea35223419d6ec116b11d4836ba3979d4`; it
reports zero sound-ID, uploader-group, file-SHA, or normalized-PCM overlaps.
This is a two-split check; test media is not included.

The full validation feature extraction completed on CPU in 183.39 s:
1,436 x 384 FP32 features across 17,207 windows. Feature matrix SHA-256 is
`a3c76c2f17a498407411fa104398bc9a0353a2658e4dff717d82f4e892863dc0`; sample
index SHA-256 is
`517b5d397217ffce104569962bad5676d76684acafc63faab30a89ebd185778d`.

The seed-17 linear readout was then fit on the 80% fit-group portion of train,
with the remaining 20% of train groups used only for scalar temperature
calibration. Checkpoint selection used validation macro NLL. The best epoch
was the maximum scheduled epoch 100 (not an early-stop point); validation
accuracy and NLL were still improving at that boundary. At epochs 1 / 25 / 50
/ 75 / 100, validation macro Accuracy was 0.0765 / 0.2617 / 0.3507 / 0.3930 /
0.4326 and macro NLL was 3.1165 / 2.6292 / 2.1948 / 1.9496 / 1.8051. The
selected artifact has 8,855 trainable parameters, is 36,918 bytes, and reloads
with exact logits.
Validation (1,436 examples, 254 uploader groups) was 0.4326 macro Accuracy,
1.8051 macro NLL, 0.7140 macro Brier, 0.2775 macro ECE before calibration;
after temperature scaling these were 0.4326 / 1.7906 / 0.7155 / 0.2745.
Sample-weighted calibrated metrics were 0.5508 Accuracy / 1.4965 NLL /
0.6080 Brier / 0.2401 ECE. The 2,000-replicate uploader-cluster bootstrap
95% interval for macro Accuracy is 0.3816–0.5050. Calibration cannot change
accuracy and here slightly worsened macro Brier; both calibrated and
uncalibrated values are retained.

Data support remains a material limitation: validation class support ranges
from 16 to 316 examples and from 3 to 201 uploader groups. The least-supported
class has only 3 validation groups. Validation Accuracy is 0 for `fx-el`
(16 examples) and 0.0698 for `fx-ex` (43 examples), while `fx-n` is 0.9324
(74 examples). The corresponding train-side class support ranges from 64 to
453 examples; 22.1% of train clips and 20.1% of validation clips are shorter
than two seconds. Temperature-scaled validation accuracy/NLL were 0.5849/1.3917
on clips under 0.5 seconds (53 examples, 20 groups), 0.4873/1.7869 on 0.5–2s
(236, 76 groups), and 0.5623/1.4416 on >=2s (1,147, 212 groups). The short
stratum is small and does not show a simple duration-driven failure pattern.
These gaps and label-source uncertainty prevent a broad quality claim. No
95–98% target was reached. The selected epoch hitting the 100-epoch boundary
means this run does not establish a clean optimization plateau; it also does
not prove that more epochs alone will fix class/source limitations.

The serialized effective readout config SHA-256 is
`4393065be5e4622d38ae4d40d93a026471b0d2a2bfbd7cc8dee398fd390a5944`, selected
readout SHA-256 is
`00fbc35a82f0205dfc57917a35d048a1faf716f47fec34aaf6ec50abc3384380`, and
run-report SHA-256 is
`7c154f3598bbaf9a2e7ad0f137104e5bd2ffb59fd69d8f87b5a4e1fc71ad1d4a`. The
checked-in YAML now pins the exact split manifest used; its post-correction
SHA-256 is
`25234c0686e55ce98bf045d5fd148282e06d73c80a86925ce39d87fc0d6c8046`.

The custom test split, sealed audit, and DCASE evaluation data have not been
opened. These validation metrics are development evidence, not blind results.
The selected checkpoint and temperature are preserved outside Git under the
attempt-01 artifact directory; no source weights or data were modified.

The extractor suppresses the pinned upstream model's architecture dump while
loading it; upstream deprecation warnings remain visible. The downloader
retries bounded HTTP 429/5xx failures while retaining exact
range, ZIP CRC, per-file hash, and 2 GiB free-space checks. A CI failure on the
optional ML test was traced to PyTorch not being installed in the repository's
standard `.[dev]` CI environment; that test now skips when the optional PyTorch
dependency is absent. Local full tests pass, and CI is green on commit
`f0819013`.

The current validation results are reported with per-class counts, duration
strata and grouped uncertainty. Confidence and uploader imbalance remain
explicit data limitations rather than grounds to hide weak class-level
results.

