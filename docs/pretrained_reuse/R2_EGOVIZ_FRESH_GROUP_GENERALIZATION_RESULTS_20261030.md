# EgoViz unseen-group fixed-checkpoint evaluation (2026-10-30)

## Result and scope

Completed the frozen one-time comparison on **2,394 four-choice video-to-IMU
timestamp-correspondence samples from 69 previously unused EgoViz groups**.
The feature corpus contains no training rows. This is a development-set
generalization diagnostic for one narrow correspondence task; it is neither a
sealed audit nor evidence of general-purpose multimodal Decision quality.
No training, checkpoint selection, tuning, GPU work, paid compute, or sealed
media access occurred.

The GRU fixed endpoint improved Accuracy over the matched-budget linear fixed
endpoint by 3.84 percentage points (group-bootstrap 95% CI +1.66 to +6.22 pp),
but its probability quality was substantially worse: NLL 4.796, Brier 0.786,
ECE 0.378, and mean confidence 0.949. Accuracy alone therefore makes the GRU
look better than its badly overconfident predictions warrant. The historical
linear step-616 checkpoint was better calibrated and had lower NLL/Brier than
the linear 5,632-step endpoint; this comparison reports the existing selector,
not a new selection on the fresh set.

| Fixed checkpoint | Parameters | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Linear, step 616 (historically selected) | 81,920 | 0.5522 | 1.1981 | 0.6508 | 0.2105 | 0.3417 |
| Linear, step 5,632 endpoint | 81,920 | 0.5326 | 1.2254 | 0.6655 | 0.1984 | 0.3342 |
| Ordered cross-modal GRU, step 5,632 endpoint | 123,521 | 0.5710 | 4.7958 | 0.7865 | 0.3783 | 0.9492 |

All rows have the same 2,394 examples and four choices per example. Models are
the already-trained fixed checkpoints; the GRU and linear endpoints each used
5,632 optimizer updates on their existing 773-example, 48-group training set.
The saved run metadata confirms identical SHA-256 hashes for all five training
and feature inputs, seed 17, and sample-order hash
`64104236342d53194075291203bbd66827e29ed85822eb73a6bacd53aa77ad8f`. The fresh
set was not used to pick any of them.

## Paired group uncertainty

Bootstrap resampled the 69 `group_id` clusters, with replacement, for 2,000
draws using seed 17; intervals are percentile 95% intervals. For GRU minus
linear endpoint, the deltas are:

| Metric | Difference | 95% interval |
|---|---:|---:|
| Accuracy | +0.0384 | +0.0166 to +0.0622 |
| NLL | +3.5703 | +3.1068 to +4.0567 |
| Brier | +0.1210 | +0.0655 to +0.1772 |
| ECE | +0.1798 | +0.1318 to +0.2321 |

For historical linear step 616 minus linear endpoint, Accuracy is +0.0196
(95% interval +0.0009 to +0.0388), NLL is -0.0273 (-0.0357 to -0.0188),
Brier is -0.0147 (-0.0192 to -0.0100), and ECE is +0.0121
(-0.0065 to +0.0313). These intervals describe resampling variability across
the selected groups; checkpoint selection and prior exposure of the older
development set remain relevant limitations.

## Activity coverage

| Activity | Samples | Groups | Linear endpoint Accuracy | Historical linear step 616 Accuracy | GRU endpoint Accuracy |
|---|---:|---:|---:|---:|---:|
| cleaning_kitchen | 40 | 1 | 0.650 | 0.575 | 0.575 |
| cleaning_restroom | 103 | 4 | 0.553 | 0.573 | 0.573 |
| cleaning_windows | 67 | 2 | 0.552 | 0.537 | 0.597 |
| managing_trash | 240 | 8 | 0.546 | 0.525 | 0.567 |
| mopping_floor | 827 | 21 | 0.469 | 0.470 | 0.503 |
| organizing_items | 181 | 6 | 0.541 | 0.608 | 0.724 |
| preparing_cleaning_supplies | 488 | 17 | 0.582 | 0.641 | 0.613 |
| sweeping_floor | 82 | 2 | 0.476 | 0.537 | 0.561 |
| wiping_surfaces | 366 | 8 | 0.587 | 0.607 | 0.593 |

This covers 9 of the publisher's 12 activity categories. Per-category scores
are descriptive; some strata have only one or two groups and should not be
treated as reliable task-level estimates.

## Integrity, runtime, and artifacts

- Frozen source index SHA-256: `c83126637cc24a7ac2b3600724b50cf9c53966b9b098acc57d5534e14ade1074`.
- Frozen selection plan SHA-256: `9cdbf6039afcb01299cede6849afe97524f96cbbedcfeaa59e5cdb4d973338979`.
- Feature aggregate SHA-256: `8aa69a2009f3fdf8532d11a029173d4e10e982fe4056d682d176f4aeca3743d1`.
- Frozen VideoPrism feature checkpoint SHA-256: `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`; extractor source commit: `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`.
- Frozen MOMENT sensor feature checkpoint SHA-256: `785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877`; extractor source commit: `38f7310ad594100747ca2a8357e9c7ca7d323e0e`.
- Compared readout checkpoint hashes: linear step 616 `ac399c77e9d3d6dd647d6ae8a924068cd9dff20ec00b90b82f69d34733f01885`; linear step 5,632 `4936aa2f696d41ede8419067af37daf251292afdb22578a766327bbc5de23bde`; GRU step 5,632 `ee54065f5161bfb1ead7356ad9d838d2f28cc1d4294597087986ff3d8d8505ec`.
- Evaluation report SHA-256: `f1ba9b0ee37b2113f69b414f36fb056bfbf539dfed02b4c38e6c49519564b1e8`.
- Per-sample predictions SHA-256: `eacad54b8e49003d3d9fd265f0ce8397d5daef99e5293bdcdbf4e141e4d9ce08`.
- The held-out set has 69 groups, zero train rows, and zero overlap with the
  frozen excluded group set. Publisher `/action` labels were not read. Sealed
  group IDs were used only as exclusion metadata; sealed media was not fetched
  or opened.
- The 69 selected MCAPs total 38,394,025,914 bytes cumulatively. They were
  processed one at a time, source size/hash checked, with a 3,000,000,000-byte
  reserve gate; only each newly acquired raw file was released after feature
  validation. The end-to-end feature acquisition/extraction/aggregation ran
  CPU-only from 19:50 to 22:45 JST (about 2 h 55 min). The readout comparison
  also ran CPU-only and completed in about 14 seconds. No training checkpoint
  was created or changed.
- The existing comparison checkpoints are linear step 616 and step 5,632
  (`C:\CodexArtifacts\egoviz-train-coverage-budget-20261030-attempt-01`) and
  GRU step 5,632
  (`C:\CodexArtifacts\egoviz-gru-temporal-budget-20261030-attempt-01`). Their
  hashes are captured in the external `evaluation/results.json`; they remain
  untouched.
- Full feature and prediction artifacts remain outside Git at
  `C:\CodexArtifacts\egoviz-fresh-generalization-20261030-attempt-01\`.
  This repository contains only the small protocol, code, tests, and summary.

## Interpretation and next step

The evidence supports a limited conclusion: on this timestamp-correspondence
task and these groups, the GRU endpoint ranks the correct option more often
than the matched linear endpoint, but produces unusably overconfident
probabilities. The linear historical early checkpoint retains better Accuracy
and NLL than its fixed endpoint. Thus longer training and temporal recurrence
are not a quality win by themselves; calibration/overconfidence is the
immediate readout issue. This does not prove an architectural ceiling or
justify another GRU training run.

Group separation does not guarantee participant separation; the publisher's
action labels are automatically generated and unverified; timestamp-derived
correspondence is not a natural-world action decision; audio and text are not
part of this task; and the 9-category distribution is imbalanced. R2-M stays
open. Before any new fit, define a product-directed, sensor-inclusive task
with trusted labels and a separately frozen development design. Do not tune
on these 69 groups or call this result four-modality quality.

## Reproduction

From the `codex/r2-egoviz-fresh-session-validation-20261030` worktree, with the
existing Python 3.11 / PyTorch 2.6.0+cu124 environment and saved external
artifacts:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:JAX_PLATFORMS='cpu'
python -m scripts.evaluate_egoviz_fresh_group_generalization `
  --plan manifests/pretrained_reuse/egoviz-fresh-generalization-plan-20261030.json `
  --corpus C:\CodexArtifacts\egoviz-fresh-generalization-20261030-attempt-01\full-aggregate `
  --linear-run C:\CodexArtifacts\egoviz-train-coverage-budget-20261030-attempt-01 `
  --gru-run C:\CodexArtifacts\egoviz-gru-temporal-budget-20261030-attempt-01 `
  --output C:\CodexArtifacts\egoviz-fresh-generalization-20261030-attempt-01\evaluation-replay
```

The evaluator refuses to overwrite output, validates plan/corpus/checkpoint
hashes and provenance, and does not read sealed data or perform training.
