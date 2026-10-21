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

The runner is `scripts/evaluate_pch_holdout_once.py`. The completed CPU
preflight did not open holdout labels/payloads; it pinned the inputs before the
single guarded run described below.

## Attempt 01 — blocked before inference

The preregistered run was launched once through the cooperative HotUGPU guard
using the immutable preflight receipt with SHA-256
`43603d836f03ff66d3956605b52742d86edf35d6f03a965d788ed56ea633da74`. The
guard waited for Orbitune PID 10912 to yield, reserved 5,120 MiB after GPU
utilization reached 0%, and observed 11,095 MiB free for the child. It did not
stop or signal Orbitune. The reservation was released when the child exited.

Before loading either model or running a forward pass, the decoder rejected
the first heldout CSV because a row had fewer than seven columns (timestamp and
calibrated axes were missing). One heldout recording payload was opened and its
filename label parsed. No model was loaded, no inference forward or optimizer
update occurred, and no prediction or quality metric was produced. The
external failure receipt is at
`C:\\CodexArtifacts\\pretrained-reuse-r2\\pch-holdout-once-failure-01\\failure.json`.

Per the preregistered stop rule, this attempt is not retried and the decoder is
not adapted to this cohort. These participants are now considered observed
development evidence, not an untouched blind holdout. Consequently this
protocol produced no model comparison and cannot support a transfer-quality
claim. A future attempt would require a different untouched cohort and a
separately reviewed protocol.
