# MiniWoB frozen-checkpoint prospective generalization protocol (2026-11-14)

## Purpose and scope

This is one inference-only generalization check for the already selected DINOv2-small + MiniWoB readout on a new simulator-seed cohort. The model, preprocessing, readout, and temperature remain fixed. The 512-row development set that selected epoch 99 is not loaded or rescored. No fitting, calibration, checkpoint choice, confidence filtering, or threshold tuning is allowed.

This is a narrow synthetic screen-action M1 result. It cannot establish natural-image or video quality, temporal reasoning, audio/text/sensor support, multimodal fusion, or release quality. The cohort is permanently treated as observed after this one evaluation and cannot be used for further tuning.

## Frozen setup

- Source: MiniWoB++ `33c3b4ddef8c6eb67c57a29663d844b1eda7e614`, MIT; local `LICENSE` SHA-256 `24c56d96c8a2e4e20c20867b14d177b801027b603c45622218b34c841ad4972a`.
- Task: `miniwob/click-color-v1:screen-only`; the rendered visible swatch must uniquely match one option's center pixel. Text, DOM, color names, and target fields remain outside model-visible rows.
- Fresh cohort: first 1,024 eligible episodes from seed window `[1,008,192, 1,016,384)`. All attempted seeds, including ineligible episodes, are logged. Eligibility is fixed by the existing screen-only collector. Exact seed reuse and decoded-RGB pixel reuse against all enumerated prior MiniWoB artifacts fail closed.
- Frozen encoder: DINOv2-small revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, 22,056,576 parameters, weight SHA-256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`, CPU FP32.
- Frozen readout: `checkpoint-best.safetensors`, the already selected epoch-99 checkpoint, SHA-256 `983366413dc2afd9f21a5411dd8793525f68342c44936d3bb2ccffe123f5c3fb`. No optimizer, training, or calibration state is loaded. Temperature is 1.0.
- Preprocessing: whole 160x210 RGB screen resized to 224x224 bicubic, ImageNet normalization, no crop.
- Controls fixed before cohort generation: position-frequency prior from legacy TRAIN only; black screen; Sattolo screen permutation seed 1701; same-example option permutation `[2,0,3,1]` mapped back by opaque option ID.
- Metrics: Accuracy, NLL, multiclass Brier, 15-bin top-label ECE, mean confidence, target-color strata, and paired episode-group bootstrap intervals (10,000 draws; seeds 17/1701).
- Device: approved local CPU Python/runtime only. No CUDA initialization, model training, cloud compute, or dependency upgrades.

## Prior-use inventory required before generation

The inventory covers both invalidated and authoritative screen corpora, all three stored smoke cohorts, the historical seed-900000 browser inspection, and the retained failed first collection attempt. Exact roots and per-file receipts are encoded in the experiment config. The two full corpus attempts contain the same train/development seed IDs; attempt-01 remains invalidated due to JSONL newline-hash mismatch and is still included in the exclusion inventory. Runtime smokes reused old training rows and are also excluded.

The first collection attempt for this protocol generated 1,024 screenshots from `[1,000,000, 1,008,192)` but failed before writing model rows or a corpus manifest: its final validator unnecessarily imported Torch, absent from the collector environment. No model was loaded or scored. The incomplete output is retained and added to the historical-use inventory; the repaired run uses the next disjoint seed window and a new output directory.

The seed-window freshness and decoded RGB overlap checks must pass before inference. A partial corpus, failed integrity check, or incomplete environment blocks scoring; no alternate seed window may be chosen after viewing any predictions.

## Interpretation rule

Report the fixed checkpoint's result and paired control differences with uncertainty. This is prospective corroboration for a synthetic screen-only task, not a blind final product audit. A weak result is retained unchanged. No future model selection may use this cohort.
