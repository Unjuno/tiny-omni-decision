# EgoViz expanded-train calibration transfer (2026-11-05)

## Result

A single positive inverse-temperature scalar fitted only from five-fold,
group-held-out logits on the existing 773-example / 48-session expanded train
corpus improved probability scores on the stored 2,394-example / 69-session
EgoViz fresh-group evaluation. Accuracy stayed exactly **55.22%**, as expected
for positive logit scaling. NLL changed **1.1981 → 1.0407**, Brier
**0.6508 → 0.5644**, and same-baseline 10-bin ECE **0.2105 → 0.0307**.

This is a **post-hoc analysis of an already exposed development set**: the
unscaled metrics for these 69 sessions had been reported before this analysis.
The scalar was fitted without their labels, but this result is not blind, does
not create a new independent test, and is not evidence of general four-modality
quality or product readiness. No checkpoint, model weight, corpus, or sealed
audit was changed or accessed.

## Frozen protocol and inputs

- Protocol: [fixed protocol](R2_EGOVIZ_EXPANDED_OOF_CALIBRATION_TRANSFER_PROTOCOL_20261105.md), SHA-256
  `04ad0032e5cd1970131f55fe9d0af5e7269449fc1d71247ef09237814bdbc833`.
- Code commit used: `3595bbc97d87d5e4f79234db77e29b141db1cb57`.
- Runner SHA-256: `8587d684e8c5bee58f5315d17eb68fac9b978bccec7b1a35c7b7229cf1bd98af`.
- Seed 17; CPU-only; five deterministic group folds. Every train example
  received one OOF score from a model that neither fit nor early-stopped on its
  session. The scalar was frozen before reading the stored fresh-group
  predictions and labels for metric reporting.
- Scorer: existing frozen-feature `JointTemporalScorer`, 81,920 parameters;
  fixed expanded-training checkpoint at step 616 was evaluated without
  modification. The temperature scalar is post-processing and adds no model
  weight parameters.
- Expanded train receipt SHA-256: `e8a451de798f468af65dddb037e12ef132a45062df3dc4f882aa2addb5550635`.
  Inputs were rehashed against the receipt:
  - train samples: `9e108f5d5ba2c756321c96b4ab47a0a387b146883367066499db7eea214d8245`
  - development samples: `55fe808c824825762da63d2c0f376dde30d61986a8fb6481c0768f661aa9d232`
  - IMU windows: `26e31282046ec47ff358ba0c18d1e46650fb78d03868186ff239b43f8f206554`
  - video features: `17b14afdfef6255a11662cd5989732ef4a15a11bdea737f4b8582f233dc9b28c`
  - sensor features: `4c2201b13403ce5f395591b876fdd78ae9b3a99cff7898556eaa3694f28af431`
- Fixed step-616 checkpoint SHA-256:
  `ac399c77e9d3d6dd647d6ae8a924068cd9dff20ec00b90b82f69d34733f01885`.
- Stored 69-session prediction SHA-256:
  `eacad54b8e49003d3d9fd265f0ce8397d5daef99e5293bddcbf4e141e4d9ce08`.
- Sealed audit was not loaded. Fresh-set labels were not used to fit the
  scalar, select a checkpoint, or select a model.

The fitted inverse temperature is **3.3731755665** (temperature **0.2964565527**).
Because it is positive, all per-example option rankings and Accuracy are
unchanged by construction.

## Metrics on the stored 69-session set

| Metric | Raw | Fixed train-only scalar | Difference (raw − scaled) |
|---|---:|---:|---:|
| Accuracy | 0.5522 | 0.5522 | 0.0000 |
| NLL | 1.1981 | 1.0407 | +0.1574 |
| Brier | 0.6508 | 0.5644 | +0.0864 |
| 10-bin ECE | 0.2105 | 0.0307 | +0.1798 |
| Mean confidence | 0.3417 | 0.5385 | — |

Paired group bootstrap used 2,000 resamples of the 69 `group_id` sessions,
seed 17, with the scalar held fixed. Percentile 95% intervals for raw-minus-
scaled differences were NLL **[0.1253, 0.1849]**, Brier **[0.0696, 0.1008]**,
and ECE **[0.1432, 0.1967]**. These intervals quantify group-resampling
variability conditional on this fitted scalar; they do not undo prior exposure
of this development result.

All nine task categories had unchanged Accuracy and lower NLL, Brier, and ECE
after scaling. Per-category sample counts and metrics are in
`category-metrics.json` in the artifact directory. Category sizes range from
40 to 827, so small-category values are descriptive.

## Fold and execution record

| Fold | Held-out examples | Best epoch | Stop epoch | Fit seconds |
|---:|---:|---:|---:|---:|
| 0 | 175 | 27 | 39 | 4.87 |
| 1 | 160 | 20 | 32 | 4.14 |
| 2 | 153 | 14 | 26 | 3.43 |
| 3 | 154 | 28 | 40 | 5.40 |
| 4 | 131 | 16 | 28 | 3.83 |
| **Total** | **773** | — | — | **21.67** |

The full attempt-02 runner elapsed **45.46 seconds**, including fitting,
prediction checks, and bootstrap calculations; the independent reload verifier
was separately rerun once after its first child process returned non-zero, so
that verifier time is outside this elapsed figure. It then strictly reloaded
all five saved states and replayed every held-out logit with maximum absolute
error **0.0**. The first failed analysis attempt is preserved as
`attempt-01/failure.json`; it completed the five folds but stopped because the
runner addressed `metrics.<model>` instead of `metrics.<model>.overall`. It was
an implementation error, not a model or data failure; its exact runtime was
not recorded.

Environment: Python 3.11.9, PyTorch 2.6.0+cu124 (CUDA build 12.4), NumPy 2.4.6,
Windows, 14 PyTorch CPU threads. All experiment and replay computation ran on
CPU. The RTX 3080 remained occupied by unrelated live PID 30764; it was neither
used nor touched. No cloud or paid compute was used. Attempt-02 outputs total
2,514,334 bytes.

## Reproducible artifacts

External artifact directory:
`C:\CodexArtifacts\egoviz-expanded-oof-calibration-transfer-20261105-attempt-02`

- `results.json` SHA-256: `76bc7f31e1e64eaf93dc8743749ffa91006fa3531c9d8079524bf23e49b4dab6`
- `training-oof.jsonl` SHA-256: `20b2a5bbdfcb991f47a94358f4779ff3819361b15fe5f52b7abfe8f923264b89`
- `fresh-predictions.jsonl` SHA-256: `af417a29913dfa989636584b7702c2169bc02b9b672bb9ad889368128dd44ba1`
- `category-metrics.json` SHA-256: `657dc6aec1d4538baa33c22f679b498553f5d7c7f6b29c053af7bb26fce76131`
- `fold-reload-receipt.json` SHA-256: `6755bae864ffba8cb52b6c503753faa2cc52ea25f5da66a20085b860104156e6`

Five fold state files are included. The prior attempt-01 partial artifacts are
also retained separately and were not reused or overwritten.

## Interpretation and next gate

This supports a narrow claim: a single scalar fitted on cross-fitted train
groups substantially improved calibration scores for the same fixed
video/IMU timestamp-correspondence scorer on the 69 stored sessions, with no
Accuracy change. It does **not** show that the 81,920-parameter scorer learned
better world representations, solve the low 55% task accuracy, or establish
95–98% performance. Do not promote this scalar as universally calibrated.

Next evidence should come from a separately pre-registered, untouched
session/task cohort (or another modality's task) using this scalar frozen in
advance. The 69-session set must not be used for further calibration or model
selection.
