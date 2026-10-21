# Speaker-held-out calibration check for text-free audio

Date: 2026-10-21  
Scope: CPU-only development comparison; no encoder inference, GPU use, or sealed-audit access.

## Question

The existing frozen-feature, 10-way Speech Commands readout reached 96.56%
Accuracy on its previously inspected 640-example validation split, but had ECE
0.1515. This check asked whether a single temperature fitted on a disjoint set
of training speakers could improve the probabilities without changing the
classifier's predictions.

## Protocol

- Reused the frozen Whisper feature cache and source manifests without modifying
  them. Source revision: `google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da`;
  encoder revision: `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`.
- Deterministically ranked unique training speaker-group hashes by
  `sha256("17:calibration:" + group_hash)`. The first 20% (166 speakers, 264
  examples) were held out for scalar-temperature fitting. The remaining 663
  speakers (1,016 examples) trained a 3,850-parameter linear readout for eight
  epochs with the existing seed-17 AdamW recipe.
- Used the last epoch; no validation checkpoint selection. Fitted one positive
  temperature on calibration logits using CPU float64 LBFGS. Then evaluated
  both unscaled and scaled logits once on the same 640 existing validation IDs.
- Fit, calibration, and validation speaker groups were disjoint. The validation
  split has already been inspected in prior work, so it is development evidence,
  not blind evidence. No test or sealed-audit data was loaded.

## Result

| Same 640 validation examples | Accuracy | NLL | Brier | ECE (15 bins) | Mean confidence |
|---|---:|---:|---:|---:|---:|
| New readout, uncalibrated | 96.56% | 0.3346 | 0.1170 | 0.1885 | 0.7771 |
| Same readout, temperature 0.4270 | 96.56% | 0.1435 | 0.0559 | 0.0245 | 0.9536 |
| Prior 1,280-example readout (reference only) | 96.56% | 0.2834 | 0.0986 | 0.1515 | 0.8141 |

A paired bootstrap over 206 validation speaker groups (10,000 resamples, seed
17) gives calibrated-minus-raw differences of `0.000` Accuracy (95% percentile
CI `[0, 0]`), `-0.1912` NLL (CI `[-0.2182, -0.1621]`), `-0.0611` Brier (CI
`[-0.0722, -0.0499]`), and `-0.1640` ECE (CI `[-0.1788, -0.1418]`). The zero
Accuracy interval is expected because dividing logits by a positive scalar
preserves every argmax; it is not evidence that true task accuracy is 96.56%.
These are retrospective intervals on a validation split already inspected in
prior work. The reproducible bootstrap uses only the saved prediction rows and
does not reload the model or tune the temperature.

Temperature scaling preserved Accuracy and substantially improved all three
probability metrics over the new model's raw scores. Compared with the prior
readout, the new uncalibrated model is worse on NLL, Brier, and ECE; after
calibration it is better on these validation estimates. The two trained
readouts are not a matched model comparison: this check trained on 1,016 rather
than 1,280 examples, and the same previously inspected validation set is used
for both reference values. Calibration's gain is therefore promising but not
independent confirmation or a product-quality claim.

## Reproduction and artifacts

Run with Python 3.11 and `PYTHONPATH=src`:

```powershell
py -3.11 scripts/train_textfree_audio_calibrated_probe.py `
  --feature-cache C:\CodexArtifacts\pretrained-reuse-r2\audio-probe-v0-retry3\audio-features.pt `
  --sample-dir C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-sample-seed17-128-64-retry2 `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\textfree-audio-calibration-03
```

The machine-readable report, prediction rows, and weights are in that external
artifact directory. The report pins source, feature cache, manifests, config,
code, checkpoint, and prediction hashes. Model fitting took 0.15 CPU seconds;
the speaker bootstrap took 8.7 CPU seconds; no GPU was used. The bootstrap
receipt is
`C:\CodexArtifacts\pretrained-reuse-r2\textfree-audio-calibration-bootstrap-01\bootstrap-report.json`.
Its source prediction SHA-256 is
`9930919d76e39dee7490eaa312733a2fdbfec7ca7e091628eea66f8538370d26`; report
SHA-256 is
`6be746d8198cdfce067776a31f1008d14e2bd5421208c2b9203852741cf9b36a`.

Recompute only from the saved predictions and their run report:

```powershell
py -3.11 scripts/bootstrap_textfree_audio_calibration.py `
  --predictions C:\CodexArtifacts\pretrained-reuse-r2\textfree-audio-calibration-03\validation-predictions.jsonl `
  --run-report C:\CodexArtifacts\pretrained-reuse-r2\textfree-audio-calibration-03\run-report.json `
  --output C:\CodexArtifacts\pretrained-reuse-r2\textfree-audio-calibration-bootstrap-02\bootstrap-report.json
```

The new tests cover speaker-resampling determinism and probability validation.

This is a narrow confidence-calibration result for closed-set Speech Commands.
It does not test general speech understanding, open-set rejection, arbitrary
candidate JSON, or the broader multimodal model.
