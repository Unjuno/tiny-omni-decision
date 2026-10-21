# PCH participant holdout comparison protocol — 2026-10-21

## Purpose and fixed candidates

This is a one-time inference comparison for the optional sensor path. It asks
whether the selected compact window LSTM transfers to previously unopened PCH
participants, compared with the already trained frozen MOMENT-1-small readout.
It does not test multimodal fusion or qualify the four-modality product.

The only candidates are frozen before access to the holdout:

- Window LSTM: selected epoch 31 from
  `pch-lstm-sensor-attempt-04/lstm_best.pt`, 18,834 trainable parameters.
- MOMENT-1-small plus linear readout: selected epoch 12 from
  `pch-moment-activity-probe-attempt-01/moment_frozen-best.pt`, 35,346,642
  total parameters; the encoder is frozen.

The exact PCH archive is 98,272,925 bytes with SHA-256
`7814f871c834725d0108192cc9265b1abd2db8ead96815d82c828202f1e55ba9`. The
heldout participant ID set is fixed by the seed-17 source split and has SHA-256
`4dc5a3e76d3fcb19ef2cff19a18c92db88269a077ca6b180866e99c4be2bd901` (28
participants). The source inventory has 423 CSV recordings; 84 belong to those
participants. The participant assignment is preserved after deterministic
fit-only validation coverage swaps.

Before opening heldout labels or signal payloads, the CPU preflight verifies the
archive and both model result/checkpoint hashes, freezes the exact member-name
and participant hashes, and hashes raw plus normalized fit/validation inputs.
It records the complete command and an immutable preflight receipt. The one-time
run then checks that heldout raw and normalized signal hashes do not overlap
those fit/validation hashes or duplicate within the holdout. Every recording
from a participant stays in the same split.

## Evaluation rules

- No training, calibration, thresholding, checkpoint selection or other model
  choice is allowed after reading holdout labels.
- Use original preprocessing and checkpoint normalization. The LSTM averages
  logits over each recording's 2.5-second windows. MOMENT averages its frozen
  window embeddings, then applies its fixed selected linear head.
- Evaluate all 18 fixed activity classes. Report record-level Accuracy,
  present-class macro Accuracy, NLL, multiclass Brier, ECE-15, mean confidence,
  per-class support/accuracy and every recording's target, prediction and
  probabilities.
- Compare LSTM minus MOMENT using 10,000 paired participant-cluster percentile
  bootstrap draws with seed 17. Report intervals for Accuracy, NLL and Brier.
- Execute through the reviewed HotUGPU cooperative reservation path. It waits
  for voluntary GPU yield and holds the reservation so Orbitune's free-memory
  launch gate cannot start another child. It does not stop or signal Orbitune.
  If admission, integrity, finite-value, or resource checks fail, stop and mark
  the attempt incomplete; do not adapt settings or retry against the same
  holdout.
- Outputs go to a new external attempt directory. No model weights, PCH media or
  feature cache are added to Git. Cloud use and spend are zero.

This holdout is specific to PCH Data 2017 and remains a narrow optional-sensor
result. It does not resolve possible PCH overlap with MOMENT pretraining, qualify
other sensor populations, or count toward the required text/image/audio/video
quality objective. Once evaluated, these participants are considered observed
development evidence for this project and must not be reused for tuning or
checkpoint selection.

## Reproduction

The runner is `scripts/evaluate_pch_holdout_once.py`. First create a fresh
preflight directory without opening holdout labels/payloads; then commit the
preflight hash and invoke the runner once through `scripts/run_with_hotgpu_guard.py`.
The exact run receipt, prediction hash and result will be appended below after
the frozen execution.
