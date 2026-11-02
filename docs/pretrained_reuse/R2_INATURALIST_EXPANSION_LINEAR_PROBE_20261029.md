# iNaturalist CC0 expansion linear probe (2026-10-29)

## Setup

One CPU-only seed-17 training run used the frozen VideoPrism features from the
pages 11–30 metadata expansion. It fit one affine 768-to-8 classifier (6,152
trainable parameters); the VideoPrism encoder stayed frozen. Feature and
metadata hashes/order were checked before fitting. Normalization used only the
921-row training split. The 446-row observer-disjoint validation split was
used for checkpoint selection by NLL (`min_delta=1e-4`, patience 25, maximum
200 epochs). No other seed or hyperparameter run was attempted. No calibration
or post-hoc adjustment was applied.

## Result

| Split / checkpoint | Accuracy | Macro Accuracy | NLL | Brier | ECE (15 bins) | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Train, selected epoch 4 | 81.32% | 77.18% | 0.6793 | 0.2699 | 0.0715 | 0.8832 |
| Validation, selected epoch 4 | 71.08% | 66.24% | 1.4023 | 0.4689 | 0.1745 | 0.8803 |

Chance accuracy is 12.5%; the training-majority prior reaches 16.59% on this
validation split. Per-taxon validation metrics:

| Taxon ID | Count | Accuracy | NLL |
|---:|---:|---:|---:|
| 3017 | 80 | 83.75% | 1.1268 |
| 4956 | 51 | 64.71% | 1.6870 |
| 6930 | 39 | 66.67% | 1.9675 |
| 7089 | 56 | 80.36% | 1.0852 |
| 9083 | 25 | 36.00% | 3.5473 |
| 12727 | 41 | 43.90% | 1.6868 |
| 13858 | 80 | 77.50% | 0.9926 |
| 144455 | 74 | 77.03% | 1.0063 |

Training stopped after epoch 29 with epoch 4 retained as best by validation
NLL. Train NLL then fell to 0.0095 and train accuracy reached 100%, while
validation NLL rose to 2.0876; final validation accuracy was 69.96% and ECE
0.2368. This is clear overfitting after the selected epoch. Validation accuracy
peaked at 71.30% at epoch 24, but the fixed NLL selector was not changed to
choose it. The result does not approach the 95–98% aspiration even on this
narrow eight-taxon image task; it is not a multimodal Decision evaluation.

## Reproduction

- Source commit before adding this run script: `546a5be9d26ea851588b5930432de306973fd165`.
- Training script SHA-256:
  `5191e2179da11ff984bddbc9804629f4dd410c695a911a51c89faf41873a73a9`.
- Feature cache SHA-256:
  `f198750082a71c92f939ddfc03f8df1f2fc362a59757d50e8dfb8f01ba065ecc`.
- Feature-row SHA-256:
  `d772a67381f37dbda25c77d7a2ee4e841773573023dfa920082e4f4be1d309ae`.
- Training / validation metadata SHA-256 are recorded in the frozen feature
  result and run config.
- Command:
  `python scripts/train_inaturalist_videoprism_linear_probe.py --feature-dir C:\CodexArtifacts\inat-birds-cc0-expansion-videoprism-features-20261029-attempt-01 --metadata-dir C:\CodexArtifacts\inat-birds-cc0-expansion-filtered-pages-11-30-20261029-attempt-02 --output-dir C:\CodexArtifacts\inat-birds-cc0-expansion-videoprism-linear-readout-20261029-attempt-01 --seed 17 --epochs 200 --patience 25 --min-delta 0.0001 --learning-rate 0.03 --weight-decay 0.0001`.
- Environment: Python 3.11, PyTorch 2.6.0+cu124; computation explicitly used
  CPU. Fit took 1.37 seconds. Trainable parameter count: 6,152.
- Output directory:
  `C:\CodexArtifacts\inat-birds-cc0-expansion-videoprism-linear-readout-20261029-attempt-01`.
- Selected checkpoint SHA-256:
  `1d70389ccc50cf8208d88f1cd4b3b03565d2f5c28ee27417f65d6effb9ae31b1`.
- Full per-epoch train/validation history SHA-256:
  `1fb404ea96fa1033906e580623a4b39e3f2a23f722db5caebf44e818f713832a`.
- Validation predictions SHA-256:
  `0fc854c25a8d9887674c12fb1b6af50a97992baf886d9bbdfd2f3b89b490fbeb`.
- Result SHA-256:
  `bbc5f4d03b0c6fc0c5242a6f7e1f5cb324fe08e84522eec631b0555def49a56f`.

The prior 176-row observer-transfer cohort and all Teacher checkpoints were
left untouched. This is one development experiment; do not treat it as a blind
test or product-quality claim.
