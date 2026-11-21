# R2-M train-only participant-grouped CV (2026-11-21)

## Question and decision

Does the current time-aware audio+vibration Decision readout improve supplied
activity selection on participants that were excluded from all fitting and
checkpoint selection, compared with an audio-only readout of the same frozen
audio representation?

**It did not in this experiment.** Across seven outer folds over the original
training participants, Audio-only was better on pooled out-of-fold Accuracy,
macro Accuracy, NLL, Brier and ECE. The paired fusion model improved Accuracy
on two of seven participants and regressed on five. Do not retain this fusion
candidate for this task on the evidence here. This is not evidence that sensor
fusion is generally unhelpful.

## Data boundary and fold design

The run used only the already frozen, rights-cleared Zenodo audio/vibration
feature cache (`10.5281/zenodo.7937591`, CC BY 4.0), and only the seven
participants assigned to the source split's training pool: 2, 4, 7, 8, 10, 13
and 14. The cache loader verified the frozen metadata, pair index, sidecars and
feature hashes. It loaded 5,449 available train features across 24 activity
labels. Fifteen additional train metadata rows had no frozen feature entry and
were not fabricated or repeated.

Each outer fold held out one entire training participant. From the other six,
one whole participant was assigned to calibration and the remaining five to
fit. The deterministic calibration rule balanced reuse across participants
while requiring all 24 choices in both fit and calibration. A fold's held-out
participant was never used for fitting, normalization, model selection or
calibration. Every one of the 5,449 sample IDs received exactly one paired
out-of-fold prediction from each arm. The two model arms used the same folds,
fixed 24-choice vocabulary, seed, fit/calibration/held-out IDs and evaluation
order.

The original exposed calibration participants (12, 17) and validation
participants (9, 15) were not used in this experiment. The sealed audit was not
loaded or evaluated. The frozen feature cache reports `sealed_audit_loaded:
false`; the CV loader rejects sealed-split feature rows. This was a train-only
cross-validation estimate, not an independent final test.

## Fixed training conditions

| Setting | Value |
| --- | --- |
| Seed | 17 |
| Arms | Audio-only, audio+vibration time-aware gated fusion |
| Epochs | 80 per arm and outer fold; no early stopping |
| Batch size | 32 |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.1 |
| Schedule | CosineAnnealingLR over the full 80-epoch budget |
| Loss | Train-only inverse-frequency weighted cross entropy |
| Candidate order | Randomized for each training batch; canonical at evaluation |
| Input normalization | Vibration per-axis mean/std fit on that fold's fit rows only |
| Selection | Lowest calibration macro NLL; outer participant not consulted |
| Initialization | Fresh readout weights in every fold; no checkpoint warm start |
| Encoder/features | Frozen pretrained EfficientAT audio tokens and frozen vibration features |
| Trainable parameters | Audio-only 133,826; audio+vibration 180,675 |

The fusion readout adds 46,849 parameters over Audio-only (+35%). Each arm
processed every fit row once per epoch; rows reappear in distinct CV folds by
design, not within an epoch. Across all 14 fits this was 4,065,760 fit-row
presentations and 127,360 optimizer updates. The run lasted 3,728.4 seconds
(62.1 minutes); maximum PyTorch CUDA allocation was 174.8 MiB on the local RTX
3080 Laptop GPU (16 GiB). No cloud compute or compute charges were used.
The run started at `2026-11-20T22:32:09Z` and completed at
`2026-11-20T23:34:17Z`.

## Pooled out-of-fold results

All metrics below use the same 5,449 paired out-of-fold predictions and all 24
supplied activity choices. Macro metrics first average over the 24 activities.

| Arm | Accuracy | Macro Accuracy | NLL | Macro NLL | Brier | Macro Brier | ECE-15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Audio-only | 79.43% | 80.12% | 0.7053 | 0.7044 | 0.3060 | 0.3020 | 0.0851 |
| Audio+vibration | 72.29% | 73.27% | 1.2269 | 1.2633 | 0.4237 | 0.4227 | 0.1178 |

Fusion minus Audio-only was −7.14 percentage points in pooled Accuracy and
−6.85 points in macro Accuracy; macro NLL worsened by 0.5589 and macro Brier
by 0.1207. ECE worsened by 0.0327. Per-activity metrics and full probabilities
are retained in the result artifact.

| Held-out participant | Examples | Audio-only Accuracy | Audio+vibration Accuracy | Difference |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 727 | 86.24% | 82.39% | −3.85 pp |
| 4 | 1,867 | 77.13% | 69.90% | −7.23 pp |
| 7 | 494 | 83.20% | 76.52% | −6.68 pp |
| 8 | 392 | 67.86% | 78.83% | +10.97 pp |
| 10 | 392 | 70.41% | 77.55% | +7.14 pp |
| 13 | 1,182 | 83.50% | 69.04% | −14.47 pp |
| 14 | 395 | 81.27% | 57.72% | −23.54 pp |

An exploratory paired participant-cluster bootstrap used 10,000 resamples of
the seven participants with replacement (PCG64, seed 17). For equal-weighted
participant mean deltas, fusion minus Audio-only Accuracy was −5.38 pp (95%
percentile interval −13.48 to +2.64 pp), NLL was +0.431 (interval +0.108 to
+0.733), and Brier was +0.085 (interval −0.024 to +0.192). These intervals are
fragile with only seven clusters and do not capture uncertainty from the
overlapping fit participants across folds. Treat them as a diagnostic, not a
population-level confidence statement.

## Learning and selection behavior

All 14 fits completed all 80 epochs and retained separate best-calibration and
final checkpoints. Best epochs ranged from 4 to 23. Several runs reached
100% rolling train Accuracy while calibration NLL rose late in training; this
is direct evidence of overfitting on these small participant groups. The
calibration selector chose the checkpoint for each arm independently without
examining outer-fold results. Reloaded best checkpoints were evaluated by the
training pipeline before OOF predictions were written.

| Held-out participant | Audio-only best epoch | Audio+vibration best epoch |
| ---: | ---: | ---: |
| 2 | 14 | 6 |
| 4 | 7 | 23 |
| 7 | 9 | 9 |
| 8 | 4 | 20 |
| 10 | 5 | 8 |
| 13 | 4 | 5 |
| 14 | 9 | 8 |

## Interpretation and next work

The current audio+vibration gated readout does not transfer consistently to
held-out participants in this supplied-activity task. Its calibration and
proper scores are substantially worse overall, despite two participants
benefiting in Accuracy. The result argues against adding this particular
fusion head to the task model without redesign. It does not distinguish whether
the limiting factor is participant variation, vibration normalization,
training data volume, the supplied-label task, or the fusion architecture.

Do not tune this result on the previously exposed validation participants or
inspect the sealed audit. For the broader pretrained-world-representation
goal, next select a fresh rights-cleared task where both inputs are necessary,
keep participants/assets grouped, and pre-register a compact audio-only vs
multimodal comparison. A model result on this activity benchmark alone cannot
establish the four-modality 95–98% product objective.

## Reproduction and artifacts

Source commit before this experiment: `d075267a4264186234038cce45d6295725440b4a`.
The exact executed source file hashes are in the run's `source-files.json`;
the effective config and its hash are in `effective-config.json` and
`effective-config.sha256`. The code added the explicit fold-missing-class
metric path while preserving strict default evaluation behavior.

Command:

```powershell
$env:PYTHONPATH = "src;scripts"
python scripts/cv_r2m_train_participants.py `
  --feature-cache "D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-frozen-feature-cache-v1-attempt-02" `
  --pair-index-dir "D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-full-pair-index-v1-attempt-01" `
  --output-dir "C:\CodexArtifacts\pretrained-reuse-r2m\participant-cv-v1-attempt-01" `
  --device cuda
```

The run directory contains per-arm histories, best and final checkpoints,
per-fold summaries, paired OOF prediction JSONL, effective configuration,
run status and SHA-256 hashes for 78 artifacts. It remains outside Git. No
legacy corpus, checkpoint or artifact was overwritten. No quantization,
Recovery LoRA or product promotion was performed.

## Implementation verification

- Focused activity/CV/training-pipeline tests: 13 passed.
- Full test suite: 3,039 passed, 11 skipped.
- `ruff check .`: passed.
- The manifest validation commands from `.github/workflows/ci.yml`: passed.
- OOF integrity audit: 5,449 unique train IDs per arm, identical paired ID
  sets, finite normalized 24-choice probabilities, target/participant identity
  matched to frozen training features, and all 14 best/final checkpoint hashes
  verified.
- Run artifact audit: all 78 listed file hashes verified; no missing files or
  mismatches; complete external output directory is 57.9 MiB.
