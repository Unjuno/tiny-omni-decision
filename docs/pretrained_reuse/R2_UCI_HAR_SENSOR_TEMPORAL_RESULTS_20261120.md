# UCI HAR sensor-only temporal comparison — first result

## Scope

This is a six-class, single-sensor activity classification probe, not a
four-modality or multi-input Decision result. It uses scratch models and six
preprocessed body-acceleration/gyroscope channels. The source is UCI HAR dataset
240, CC BY 4.0; cite Reyes-Ortiz et al., DOI
[`10.24432/C54S4K`](https://doi.org/10.24432/C54S4K). The [official UCI
record](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
reports 30 volunteers, six activities, 50 Hz, 128-sample windows, and 50%
overlap. The accompanying experiment video was not used.

The goal was to compare a causal LSTM sensor stem against an order-insensitive
static-summary scorer under identical training and subject splits. This does
not close the separate deterministic-rule baseline, R2 M-2, or any product
quality gate.

## Data and integrity

- Downloaded official outer archive: 61,005,872 bytes,
  SHA-256 `C00B803081A5C797CD5E4B83700A9810B38D53D9D84E01917E090E1FDBC81031`.
- Nested `UCI HAR Dataset.zip`: 60,999,314 bytes,
  SHA-256 `2045E435C955214B38145FB5FA00776C72814F01B203FEC405152DAC7D5BFEB0`.
- Fit code extracted only the official `train` signal/label/subject members.
  Preflight found 7,352 windows across 21 official train participants.
- Seed-17 participant partition: 6,028 train windows / 17 participants and
  1,324 validation windows / 4 participants. Train and validation participants
  are disjoint. Normalization was fitted on train-participant windows only.
- Both candidates selected epoch 12, the minimum validation NLL through the
  predeclared 12-epoch budget. Their best and terminal checkpoint hashes are
  equal because the selected epoch was the terminal epoch.
- Only after `candidate-freeze.json` was written and its archive, config and
  checkpoint hashes verified, the official public test was parsed once. It
  contains 2,947 windows over participants `2, 4, 9, 10, 12, 13, 18, 20, 24`;
  overlap with all 21 fit participants is zero.
- Windows overlap 50% within each participant, so windows are not independent.
  The paired uncertainty estimate resamples participants, not windows. The
  public test is not a blind global audit and is now closed to further
  evaluation or selection.

## Fixed experiment

Both arms used seed 17, six one-hot typed activity options, batch size 128,
AdamW, learning rate `1e-3`, weight decay `0.01`, deterministic CPU execution,
and the same train/validation windows and normalization.

| Arm | Model | Parameters | Best epoch | Train accuracy / NLL | Validation accuracy / macro accuracy | Validation NLL / Brier / ECE | Fit seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| Static | mean, std, min, max + candidate scorer | 4,097 | 12 | 71.35% / 0.7707 | 66.16% / 65.74% | 0.8443 / 0.4961 / 0.1136 | 10.03 |
| LSTM | causal 64-state stem + same option scorer | 27,905 | 12 | 59.87% / 0.7489 | 61.10% / 61.41% | 0.7374 / 0.4326 / 0.0615 | 282.50 |

## Learning curves

The values below are epoch-level train and validation Accuracy / NLL. The
validation NLL decreased overall through epoch 12 for both arms; neither curve
establishes a plateau or clean overfit at this budget.

| Epoch | Static train Acc | Static val Acc | Static train NLL | Static val NLL | LSTM train Acc | LSTM val Acc | LSTM train NLL | LSTM val NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39.7% | 39.4% | 1.660 | 1.659 | 19.8% | 19.3% | 1.676 | 1.695 |
| 2 | 37.1% | 35.7% | 1.406 | 1.412 | 35.0% | 33.2% | 1.271 | 1.299 |
| 3 | 39.9% | 37.0% | 1.201 | 1.207 | 33.8% | 31.8% | 1.262 | 1.278 |
| 4 | 48.8% | 45.7% | 1.104 | 1.108 | 33.5% | 31.8% | 1.260 | 1.258 |
| 5 | 55.0% | 48.3% | 1.039 | 1.053 | 33.6% | 31.4% | 1.242 | 1.259 |
| 6 | 59.9% | 55.7% | 0.976 | 0.991 | 38.8% | 34.2% | 1.265 | 1.290 |
| 7 | 60.5% | 55.7% | 0.926 | 0.967 | 39.0% | 36.9% | 1.134 | 1.149 |
| 8 | 59.6% | 58.5% | 0.882 | 0.924 | 47.0% | 48.2% | 1.036 | 1.040 |
| 9 | 59.5% | 54.4% | 0.848 | 0.907 | 48.6% | 50.4% | 0.928 | 0.905 |
| 10 | 68.4% | 62.2% | 0.820 | 0.903 | 50.1% | 52.0% | 0.933 | 0.913 |
| 11 | 68.6% | 63.9% | 0.793 | 0.859 | 55.7% | 58.2% | 0.833 | 0.804 |
| 12 | 71.4% | 66.2% | 0.771 | 0.844 | 59.9% | 61.1% | 0.749 | 0.737 |

## One-time official public-test comparison

| Arm | Accuracy | Macro accuracy | NLL | Brier | ECE (10-bin) | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Static | 70.44% | 70.49% | 0.8043 | 0.4709 | 0.1633 | 0.5412 |
| LSTM | 59.08% | 60.38% | 0.7687 | 0.4501 | 0.0502 | 0.5577 |

Paired participant-cluster bootstrap (2,000 resamples, seed 17, nine
participants) gives LSTM-minus-static Accuracy `-11.32 pp` (95% percentile CI
`[-15.55, -7.10]`) and macro Accuracy `-10.06 pp` (`[-13.64, -6.67]`). The
LSTM is worse on Accuracy while its NLL, Brier and ECE are lower. Do not treat
the calibration changes as a compensating quality pass.

| Activity | Static recall (n) | LSTM recall (n) |
|---|---:|---:|
| WALKING | 82.06% (496) | 88.31% (496) |
| WALKING_UPSTAIRS | 62.85% (471) | 79.41% (471) |
| WALKING_DOWNSTAIRS | 76.43% (420) | 94.76% (420) |
| SITTING | 58.66% (491) | 0.00% (491) |
| STANDING | 75.94% (532) | 99.62% (532) |
| LAYING | 67.04% (537) | 0.19% (537) |

The LSTM's severe stationary-class collapse is the clearest observed failure
mode. This comparison does not identify whether it comes from training budget,
feature representation, optimization, or architecture. No capacity or
hyperparameter search was done.

## Reproduction and artifacts

- Repository: `Unjuno/tiny-omni-decision`; branch
  `codex/pretrained-reuse-next-20261115`.
- Source commit: `a80f567de7b328820c748289b0db5f506303cf68`.
- Runner SHA-256: `c2db654a01811d088fed5a8564e9587a097341e4fb724cdde4588d21fba71899`.
- Effective split-manifest SHA-256:
  `3e70b00bf16bd5eed7d3cfb04dfe66f7de78029f0c600d34c034584c5f45884c`.
- Candidate-freeze SHA-256:
  `741DEB41349A68563BA156EE72A1E77BEA72700151AAC4CE7DE3397E58681992`.
- Static best weights SHA-256:
  `1A362DB8E315EDC91DD2BF5089008392E54F53CD6DDD88F4715935F1CE81E7CA`.
- LSTM best weights SHA-256:
  `0B7F73F9C55D7426096DDC9F778AC6EE42AA407382C323F547FD05B6F7175F65`.
- Output directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\uci-har-sensor-v1\attempt-01\fit`;
  22 files / 4,145,883 bytes including the artifact hash index. The index
  SHA-256 is
  `D1EF37856D51819A18281398D246E20FC258BC4D1186778125FAA0BE70BD98E2`.
  Dataset remains outside Git.
- Fit command: `PYTHONPATH=src python scripts/uci_har_sensor_experiment.py --phase fit --archive <archive> --output-dir <new-fit-dir>`.
- One-time audit command: same runner with `--phase audit` and the frozen fit
  directory. Audit started `2026-11-20T07:28:08Z` and finished at
  `2026-11-20 16:28:12 JST`.
- Environment: Windows 10, Intel Core i7-12700H (14 cores / 20 logical),
  Python 3.11.9, PyTorch 2.6.0+cu124, NumPy 2.4.6. CPU only; CUDA was
  available but unused. Fit elapsed 4m55s including process startup and
  artifact writes. Peak RAM was not instrumented. Cloud spend: $0.
- Regression suite before the fit: `ruff check .` passed; `pytest -q` reported
  2,942 passed / 11 skipped; `git diff --check` passed.

The official test is closed. A separate 100-epoch validation-only budget
follow-up is registered in
[`R2_UCI_HAR_SENSOR_LONG_BUDGET_PROTOCOL_20261120.md`](R2_UCI_HAR_SENSOR_LONG_BUDGET_PROTOCOL_20261120.md).
It will not reload the public test or use its metrics for selection.
