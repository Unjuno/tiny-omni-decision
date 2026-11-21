# UCI HAR selected-validation error analysis

## Scope

This is a one-time descriptive breakdown of the already exposed validation
predictions from the seed-17, 100-epoch static/LSTM comparison. It did not
load a model, checkpoint tensor, signal archive, or official test record. It
does not change checkpoint selection or establish new-person performance.
The fixed protocol is in
[`R2_UCI_HAR_VALIDATION_ERROR_ANALYSIS_PROTOCOL_20261121.md`](R2_UCI_HAR_VALIDATION_ERROR_ANALYSIS_PROTOCOL_20261121.md).

## Inputs and integrity

- Source run: `C:\CodexArtifacts\pretrained-reuse-r2\uci-har-sensor-v1\attempt-02-long100`.
- Selected checkpoints: static epoch 84 and LSTM epoch 15, as frozen before
  public-test access.
- Both prediction files and both checkpoint files matched `artifact-index.json`
  by byte size and SHA-256; checkpoint hashes also matched
  `candidate-freeze.json`.
- The two ordered prediction sets match on all 1,324 sample IDs, targets and
  participant IDs. Their ordered sample-ID hash is
  `b62970cc13c6eb3566bce2fa79968c30d54ccc58ced3efc150969c74ac2b9190`.
- Participants: 5 (302 windows), 11 (316), 14 (323), and 30 (383).
- Machine-readable output is external to Git at
  `C:\CodexArtifacts\pretrained-reuse-r2\uci-har-sensor-v1\attempt-03-validation-error-analysis\validation-error-analysis.json`.
  Its SHA-256 is
  `26c8ac7f74a11311c46b9dc06831ddb2a1740a83cce587db099db16812acec9e`.

## Fixed validation results

| Candidate | Accuracy | Macro accuracy | NLL | Brier | ECE-10 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Static summaries, epoch 84 | 76.06% | 76.39% | 0.6270 | 0.3464 | 0.0521 | 0.7673 |
| Causal LSTM stem, epoch 15 | 62.01% | 62.39% | 0.7278 | 0.4232 | 0.0294 | 0.6495 |

The LSTM recognized walking classes comparatively well, but collapsed on
stationary classes. It predicted `STANDING` for all 213 `SITTING` windows and
220 of 230 `LAYING` windows: recalls were 0% and 3.91%, respectively. It had
99.55% recall on `STANDING`, 100% on `WALKING_UPSTAIRS`, and 96.0% on
`WALKING_DOWNSTAIRS`. The static scorer had 85.45% `SITTING` recall and
56.52% `LAYING` recall.

This pattern appeared across every held-out participant: LSTM `SITTING` recall
was zero for all four; `LAYING` recall ranged from 0% to 7.02%. Errors were
not simply due to one participant. The walking result was less consistent:
participant 14 had only 3.39% LSTM recall on `WALKING` (57/59 were called
`WALKING_UPSTAIRS`), while the other three had 94.64%–100% walking recall.
Per-person total Accuracy also varied: static was better for participants 11,
14 and 30; LSTM was slightly better only for participant 5. Full subject ×
class paired counts and confusion matrices are preserved in the external JSON.

## Interpretation and next step

For this exact prewindowed six-activity setup, the useful next diagnosis is
representation/training behavior for stationary-class separation and
participant variation, not a blanket increase in sequence-model capacity.
This result does not show that LSTMs or xLSTMs generally fail, nor that static
features are generally superior. The validation windows overlap and have
already influenced model selection; four participants are too few for
population inference. No window bootstrap was used. These results must not be
presented as a blind estimate or used to revise the broader 95–98% multimodal
product target.

## Reproduction

The analysis script reads only the frozen artifact index, candidate freeze,
and saved prediction JSONL files; it hashes checkpoint files but does not
deserialize them. Reproduce with:

```powershell
$env:PYTHONPATH = "$PWD;$PWD\src"
python scripts/analyze_uci_har_validation_errors.py `
  --run-dir C:\CodexArtifacts\pretrained-reuse-r2\uci-har-sensor-v1\attempt-02-long100 `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\uci-har-sensor-v1\attempt-03-validation-error-analysis
```

The output command refuses to overwrite a nonempty directory. Official test,
source signals, fitting, calibration, and checkpoint selection are outside
its code path.
