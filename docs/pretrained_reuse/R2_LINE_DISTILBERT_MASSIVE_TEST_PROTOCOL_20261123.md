# Frozen Japanese DistilBERT MASSIVE test protocol — 2026-11-23

## Purpose and status

This freezes a one-time text-only evaluation of a four-layer LINE Japanese
DistilBERT candidate on the publisher's previously unopened Japanese MASSIVE
test partition. The model must emit probabilities over the fixed 60-intent
label list. This is a bounded intent-classification Decision task, not general
arbitrary-option reasoning and not evidence of multimodal fusion.

At protocol freeze, only publisher `train` and `validation` had been loaded.
The test Parquet was identified from pinned repository metadata only; its
payload and labels have not been downloaded or read. No fitting, validation
selection, calibration, or architecture change is allowed after the test
metrics are observed. If this candidate underperforms, preserve that result
and select the next candidate using a new untouched evaluation cohort.

## Frozen source and integrity gates

- Dataset: `AmazonScience/massive`, Japanese `ja-JP` 60-intent classification.
- Dataset revision: `5092332a2835c4d29c617ef6d118123a1abecd19`.
- Declared dataset license: CC BY 4.0, per the existing pinned source receipt.
- Training input: the existing frozen `train.jsonl`, 11,228 rows, SHA-256
  `ab55ddd41e4cf0f07f65bddc84432bea40ceb36caaa94c6edf15499ddf6a1270`;
  source split manifest SHA-256
  `d9fdcc0235b3caecac462574e6111a6c1bf59102eab3ce39361c7fb42af9ed1d`.
  It excludes normalized-utterance overlap with publisher development. The
  exposed development file will not be used in this run.
- Test input identified from the pinned repository tree only:
  `ja-JP/test-00000-of-00001.parquet`, 220,619 bytes, expected LFS SHA-256
  `985d95a0ec20724bc8f07e17e375f73ced6572f9670a3a525a19065736cde012`.
- Before fitting, the runner may read only test `id` and `utt` columns to
  confirm the pinned file and remove any test rows whose exact ID or normalized
  utterance occurs in train. It must not read test intent labels, predictions,
  or metrics during this preflight. The surviving ordered test-row IDs,
  normalized-content group IDs, count, and hash are frozen in the run receipt.
- After the fixed readout checkpoint is saved, the runner may load test
  `intent` labels, perform one scoring pass, and save all per-row predictions.
  The project sealed audit remains unopened.

## Frozen candidate and fit

- Encoder: `line-corporation/line-distilbert-base-japanese`, revision
  `c32d05753d0d4be0f60482291a9452ec7e6f7d68`; source weight SHA-256
  `9d8b678bd9ce14363d5e6e3dc53c7dd7c76f8ad11394ccffd200f15e9f6819eb`.
- Use the first four pretrained transformer layers from the six-layer source
  checkpoint. Keep the encoder frozen. The prior CPU inheritance smoke
  verified 53,912,064 encoder parameters, zero trainable encoder parameters,
  and finite output; its receipt is recorded in
  `R1_LINE_DISTILBERT_RUNTIME_20261123.md`.
- Tokenizer: the same pinned local LINE tokenizer implementation, offline,
  `use_fast=False`, maximum length 128. Use only `utt`; do not use annotated
  utterance, worker fields, or test metadata as model input.
- Readout: `[CLS] -> Linear(768,128) -> tanh -> Dropout(0.1) -> Linear(128,60)`.
  The 106,172 readout parameters are the only trainable parameters.
- Fit on the frozen 11,228-row training JSONL only; no publisher validation
  data. Seed 17; float32 cached features; batch size 64; shuffle each epoch
  with a seed-17 Python RNG; AdamW with learning rate `0.002`, weight decay
  `0.0001`; exactly eight epochs. Use the final epoch-8 state. There is no
  early stopping, checkpoint selection, class weighting, temperature
  calibration, or test-guided change.
- Run CPU-only in a fresh process with the locally pinned environment. No GPU
  or cloud compute.
- New external output directory:
  `C:\CodexArtifacts\pretrained-reuse-r2-local\massive-ja-line-distilbert4-official-test-20261123-attempt-01\`.
  Refuse to overwrite an existing directory. Keep model weights, media, and
  predictions outside Git.

## Required report

Record test row and normalized-text group counts, excluded overlap counts,
per-intent counts, Accuracy, macro Accuracy over represented intents, NLL,
multiclass Brier, ECE-15, mean confidence, and per-intent Accuracy. Preserve
the complete test predictions with ordered option probabilities and all input,
model, tokenizer, script, checkpoint, prediction, and environment hashes.
Record readout train CE by epoch, extraction/fitting/evaluation time, peak
process memory where available, parameter count, and package bytes. Reload the
saved checkpoint before the one test scoring pass. State explicitly that the
publisher test is now exposed to this project and must not be used for later
model selection or repeated evaluation.

## Interpretation boundary

This single-task test can measure the fixed Japanese intent Decision path. It
cannot prove 95–98% general text reasoning, variable-option calibration across
tasks, any image/audio/video capability, or multimodal benefit. The 4-layer
encoder is only one text component candidate; the broader 100–200M Omni model
and all four modality quality gates remain separate and open.
