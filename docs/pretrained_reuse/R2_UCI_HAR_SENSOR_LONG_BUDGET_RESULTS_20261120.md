# UCI HAR sensor-only 100-epoch validation follow-up

## Scope and execution

This was a preregistered, validation-only extension of the fixed seed-17
static-summary versus causal LSTM comparison. Both models were freshly
initialized from the same deterministic seed as the prior run. This was not a
resume, warm start, pretrained-representation reuse, multimodal result, or
product candidate. The public UCI test was already consumed once by the earlier
run and was not opened here (`official_test_loaded=false`).

- Source commit: `a43662b0af0d685d02aa4460ad259664d401bc2f`.
- Runner SHA-256: `7d5bd694feedd7cb1928d59aae3cced13e02df4426d9f277275cca199d18d6a1`.
- Output: `C:\CodexArtifacts\pretrained-reuse-r2\uci-har-sensor-v1\attempt-02-long100`.
- Command: `PYTHONPATH=src python -u scripts/uci_har_sensor_experiment.py --phase fit --epochs 100 --archive C:\CodexArtifacts\pretrained-reuse-r2\uci-har-sensor-v1\attempt-01\uci-har-v1.zip --output-dir C:\CodexArtifacts\pretrained-reuse-r2\uci-har-sensor-v1\attempt-02-long100`.
- Started `2026-11-20T07:38:05Z`; finished `2026-11-20T08:22:12Z`; exit status 0.
- Windows 10, Intel Core i7-12700H, Python 3.11.9, PyTorch 2.6.0+cu124,
  NumPy 2.4.6. CPU only; CUDA unused. Cloud cost: `$0`.
- Fit time: static `257.13 s`, LSTM `2379.40 s`; wall time about `44m07s`.
- Peak process memory was not instrumented, so RAM peak is unknown.
- Both runs processed 6,028 unique train windows per epoch, 48 optimizer
  updates/epoch, 602,800 example presentations and 4,800 updates per model.
  They used the exact same 17 train / 4 validation participants and 1,324
  validation windows as the 12-epoch run. No repeated window occurs within an
  epoch; the same 6,028 training examples are revisited across epochs.
- The first 12 epochs have exactly matching train and validation metrics to
  the earlier run for both models (elapsed-time fields differ, as expected).
  Split-manifest SHA-256:
  `3e70b00bf16bd5eed7d3cfb04dfe66f7de78029f0c600d34c034584c5f45884c`.
- Effective-config SHA-256: static
  `fe45e153149140563ca391be5cacc9fdcfc2c2ded0bca6df74ba05e0d5f1a085`, LSTM
  `7301543db2bb7dff385b9eb27daca1e6a95c8d3e4e3a9d5bf77d232365b8bdcc`.
- Artifact index SHA-256:
  `ec3cf7e471989e48af498644592ec4b3315257a5495721038c5e6c98ae787caa`;
  17 indexed files, 2,430,373 bytes. Dataset and checkpoints remain outside Git.
- `candidate-freeze.json` was written after training and verifies the selected
  configs/checkpoints. Best-checkpoint reload predictions are byte-identical
  to the saved best validation predictions for both candidates.

## Selected validation checkpoints

| Candidate | Parameters | Best epoch | Accuracy | Macro accuracy | NLL | Brier | ECE-10 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Static summaries | 4,097 | 84 | 76.06% | 76.39% | 0.6270 | 0.3464 | 0.0521 | 0.7673 |
| Causal LSTM stem | 27,905 | 15 | 62.01% | 62.39% | 0.7278 | 0.4232 | 0.0294 | 0.6495 |

Both selected checkpoints were determined only by the preregistered earliest
minimum validation NLL. Static best weights SHA-256:
`bbe3dc24c6a4d88384683c2acccb492d4b13a4ea16b9f66b054e05fa343de480`; LSTM:
`88214f5c4f3d01a7c099abbf5226c3fa3b9ef0899f38ca1e667db326a9d6d36e`.

## Learning curve snapshots

| Epoch | Static val Acc | Static val NLL | LSTM val Acc | LSTM val NLL |
|---:|---:|---:|---:|---:|
| 12 | 66.16% | 0.8443 | 61.10% | 0.7374 |
| 15 | 69.11% | 0.8137 | 62.01% | 0.7278 |
| 24 | 68.43% | 0.7735 | 62.16% | 0.7810 |
| 48 | 73.26% | 0.7102 | 63.07% | 0.9359 |
| 72 | 75.98% | 0.6403 | 61.40% | 0.9346 |
| 84 | 76.06% | 0.6270 | 63.07% | 0.9547 |
| 100 (terminal) | 75.30% | 0.6459 | 65.33% | 0.8760 |

Train metrics are in the raw run history and are not represented as a held-out
generalization-gap estimate. At the terminal epoch, static train Accuracy/NLL
were 84.26% / 0.3989 and validation Accuracy/NLL were 75.30% / 0.6459. LSTM
train metrics were 65.20% / 0.6027 and validation metrics were 65.33% / 0.8760.

## Interpretation

- **Static sensor scorer:** the longer budget materially improved validation
  Accuracy from 66.16% at epoch 12 to 76.06% at the validation-selected epoch
  84, with NLL from 0.8443 to 0.6270. The terminal epoch was slightly worse
  than the best epoch. On this fixed participant split and one seed, this
  supports the narrow claim that 12 epochs were insufficient for the static
  candidate. It does not explain why and does not establish product quality.
- **LSTM sensor scorer:** the selected NLL improved only from 0.7374 at epoch
  12 to 0.7278 at epoch 15; the selected Accuracy improved from 61.10% to
  62.01%. Later Accuracy fluctuated and reached 68.43% at epoch 98, but that
  checkpoint was not selected because NLL was the frozen criterion. The
  terminal NLL was worse than the best. Longer training alone did not produce
  a clear, stable LSTM Accuracy gain under this protocol. These observations
  do not prove an architecture ceiling.
- The static scorer's 100-epoch validation result exceeded the LSTM, despite
  having fewer parameters. This is evidence against assuming that a recurrent
  stem helps this particular prewindowed HAR task; it is not evidence against
  sequence models generally.
- The earlier, one-time public-test comparison remains closed and is not
  repeated or used to select this follow-up. Its values are documented only in
  the first result report.

## Next engineering step

Keep this result as a narrow sensor M1 diagnostic. Do not add an xLSTM, increase
capacity, or change training settings based on the public test. The next useful
sensor experiment should diagnose the LSTM stationary-activity confusion on
validation with a predeclared error breakdown and compare the static/LSTM
validation predictions by participant. In parallel, continue the roadmap's
cross-modal M-2 evidence rather than presenting this sensor task as progress
toward the four-modality 95–98% target.

## Integrity notes

The active run's source script/configuration were not changed while it ran.
No official test data was loaded (`official_test_loaded=false`). No model,
dataset archive, or large checkpoint was added to Git. The test-split audit in
the experiment script was previously guarded only by per-output-directory
receipts; this follow-up must add a source-hash-level closed-audit regression
before another audit path can be used.
