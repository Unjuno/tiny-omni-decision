# R2 IntPhys Path A temporal-prediction diagnostic

**Status: implementation and data acquisition in progress; no real-data model result exists.**

## Purpose and scope

This is a task-specific Path A experiment for temporal physical-plausibility
judgment. It reuses the pinned frozen VideoPrism-B frame representation and
fits a small causal LSTM predictor on possible-only IntPhys training videos.
The scorer is the mean cosine error between predicted and observed next-frame
features. It is an anomaly score, not a general Decision model and not a
calibrated probability over JSON options.

The implementation does not add a BERT text model, shared multimodal Lens, or
sensor input. Those require separate admitted tasks and evidence. Existing
BERT-Mini results are task-specific and do not justify broadening this video
experiment.

## Data gates

- The source is the official IntPhys training shard (`train.1.tar.gz`),
  declared CC BY 4.0 by the publisher. The extraction refuses any size or
  published MD5 mismatch before reading the archive.
- Train status metadata must mark every record possible and describe exactly
  100 frames. Each record must have all 100 scene, depth, and mask indices.
- Eight uniformly spaced RGB frames feed the frozen feature extractor. All
  100 original RGB PNG payloads are SHA-256 hashed in temporal order for
  duplicate grouping and split integrity. Only the eight sampled RGB frames
  are decoded for feature extraction; the remaining PNG payloads are byte-
  hashed but not decoded, which remains a data-quality limitation.
- The deterministic internal holdout groups duplicate full status metadata or
  identical complete RGB byte sequences before splitting. It is used only for
  training early stopping and threshold fixation.
- IntPhys dev is a fixed development diagnostic, not a blind sealed audit. It
  contains 90 scene groups, each with four renditions and a 2 possible / 2
  impossible label balance. The extractor reports exact cross-label collisions
  among the eight sampled RGB inputs. Labels do not enter fitting, checkpoint
  selection, or score-threshold selection.
- The official hidden test is not accessed.

## Model and evaluation protocol

The train-only runner selects the best checkpoint by internal validation
cosine prediction loss. The evaluator validates the archive, feature, and
checkpoint receipts before scoring. It compares the learned LSTM against a
last-feature persistence baseline. Each model's binary threshold is fixed
independently at the 95th percentile of its internal possible-only validation
anomaly scores.

The report includes ROC AUC, accuracy, sensitivity, specificity, and balanced
accuracy overall and by IntPhys block. A paired bootstrap resamples complete
scene groups to estimate uncertainty for learned-minus-persistence differences.
Per-example IDs, labels, and raw anomaly scores are saved. NLL, Brier, and ECE
are intentionally omitted because neither anomaly score is a calibrated class
probability.

## Current evidence snapshot

At 2026-11-27 07:04:19 JST, the official train archive download process was
alive and the partial file was 26,886,806,304 bytes of the expected
37,016,506,106 bytes. Its expected MD5 is
`3428bcf398441a489111528634530212`. No model features have been extracted from
the partial archive; no real-data training or dev scoring has started.

The Path A implementation, data-integrity checks, and CPU tests are on branch
`codex/r2-intphys-patha-20261127`. The current local full suite reports
2,995 passed / 11 skipped; `ruff check .`, the synthetic evaluator CLI
end-to-end test, and CLI `--help` pass. CI #5125 is green for the preceding
data-audit commit; the evaluator commit is being pushed for its own CI run.
The training archive, pinned checkpoint, source tree, existing runs, and legacy
worktrees remain untouched by model training.

## Required next steps

1. Finish the same resumable archive transfer; verify exact byte size, MD5,
   SHA-256, and gzip/tar integrity.
2. Run the pinned CPU extraction into a fresh feature directory and inspect
   unique records/groups, split sizes, identity overlaps, frame counts,
   status counts, and extraction receipts.
3. Train the single seed-17 Path A LSTM using only the internal train/validation
   features. Preserve best checkpoint, full history, and hashes.
4. Extract the fixed dev features once, inspect the sampled-media collision
   audit, then run the fixed evaluator once without changing the trained model
   or threshold from dev results.
5. Report the narrow task result and limitations; do not claim broad
   multimodal quality or a 95–98% general capability from this task.
