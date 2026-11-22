# Path A LibriSpeech 100-epoch learning curve — 2026-11-17

## Question and scope

This run extends the fixed-seed, frozen-feature readout experiment from four to 100 epochs to test whether the narrow supplied-transcript choice task continues improving after the original short budget. It is a development learning curve on the same exposed speaker-held-out validation set, not blind evidence. No model/data selection used sealed audit or official LibriSpeech test data. This task is not general audio understanding, ASR, multimodal fusion, or a product-quality result.

No ternary quantization or product promotion occurred. The existing source data, runtime bundle, prepared corpus, feature cache, and earlier attempts were treated as read-only. The new run wrote only to:

`C:\CodexArtifacts\pretrained-reuse-r2-cdrive-20261117\librispeech-train-clean-100-long-curve-v1-attempt-01`

## What changed and what did not

The run used a fresh readout initialization (seed 17), not the previous four-epoch checkpoint. The only intended experiment change was the fixed budget: 100 epochs, with no performance-based early stop. The model, optimizer, LR, weight decay, query batch size, examples, split, validation selector, and candidate construction were unchanged. It reused the exact immutable audio feature cache (`ac5e3638174f4707ea4d3b9b075425ef7bdfb8453319b6e6186debc3bce51d82`) and loaded 17,369 cached feature rows in prepared corpus order.

The implementation received two narrowly scoped correctness changes before the run:

1. Per-condition validation now evaluates only that condition's validation rows. The previous code passed the full validation set to each condition report, which could make absent-condition metrics and selection invalid.
2. Checkpoint selection now uses the arithmetic mean of random- and hard-condition NLL, so neither condition dominates by count. Missing/non-finite condition metrics fail closed.

The new optional cache loader checks exact IDs and order, tensor shape `[N,384]`, floating type, and finite values. New tests cover these gates and equal-condition selection. The run itself completed with a distinct output directory; best and final readouts are separate files.

The older four-epoch run used the same seed and corpus but ran on CUDA. This 100-epoch run used CPU because the local GPU was occupied. Small numeric differences are therefore expected, and the comparison is not a perfectly controlled hardware-only or budget-only counterfactual. The long-run curve below is internally consistent on one CPU run.

## Frozen data and model identity

Source is OpenSLR LibriSpeech `train-clean-100` (CC BY 4.0), with the same archived source, CMUdict, runtime bundle, prepared train/validation split, and candidate policy as the four-epoch report. The split remains 201 train speakers / 50 held-out validation speakers, 13,902 paired train targets / 3,467 paired validation targets, and 13,902 / 3,467 unique audio assets. Speaker, audio asset, and normalized-text overlap are all zero. Each target has both random and phoneme/lexical/length-matched hard-distractor versions; paired conditions are intentional.

- Source commit before the run: `8baf812247c6df4c1c6c955d96d64eb796890578`.
- Execution script SHA-256: `b39e3d4bce930269b31c7afa3c9babb5e5a6b355e1f69fdbf9bec4bb8de21b5b`.
- Task-adaptation module SHA-256: `b54438825d00df0d69b1846e529021b4c8ee0e2a29a7d6c86a66489dfca298a6`.
- Focused test module SHA-256: `c90e7cf21a493c1326196e19d9902edeef1b68241ce0b7982e0e900c6eeb8174`.
- Effective config SHA-256: `8af006225b017ab65e492a1f91755752b0c1690653c155a99d27edf8d35f65a8`.
- Split manifest SHA-256: `3667525f7c1c2bf746d6728604a9f9609260a334c4fc14030ccc1c2bd5617b50`.
- Training feature cache SHA-256: `ac5e3638174f4707ea4d3b9b075425ef7bdfb8453319b6e6186debc3bce51d82`.
- Train examples SHA-256: `5065c1f1c10c188be389d77b0a25daeb6f3d00f4ad27f8890a417ea9f78366e6`.
- Validation examples SHA-256: `0aa0a7a792f224f092197a4ade89e8636fd35b6afed22f697c231b9b0748cb16`.

Pinned frozen backbones: `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af` (37,760,640 parameters) and `sentence-transformers/paraphrase-MiniLM-L3-v2@4ca70771034acceecb2e72475f72050fcdde4ddc` (17,389,824). Only the 344,321-parameter `FrozenFeatureCandidateScorer` was trainable; total component count is 55,494,785 before quantization. Backbone weights did not change.

## Run and compute

Command:

```powershell
$env:PYTHONPATH = 'src;.'
python -u scripts/train_librispeech_frozen_readout.py `
  --dataset-root C:\CodexArtifacts\pretrained-reuse-r2-cdrive-20261117\source-attempt-01\extracted `
  --archive C:\CodexArtifacts\pretrained-reuse-r2-cdrive-20261117\source-attempt-01\train-clean-100.tar.gz `
  --cmudict C:\CodexArtifacts\pretrained-reuse-r2-cdrive-20261117\cmudict.dict `
  --cmudict-license C:\CodexArtifacts\pretrained-reuse-r2-cdrive-20261117\cmudict-LICENSE `
  --bundle C:\CodexArtifacts\pretrained-reuse-r2-cdrive-20261117\runtime-bundle-v2 `
  --prepared-data-dir C:\CodexArtifacts\pretrained-reuse-r2-cdrive-20261117\prepared-v1 `
  --audio-feature-cache C:\CodexArtifacts\pretrained-reuse-r2-cdrive-20261117\librispeech-train-clean-100-readout-v1-attempt-01\audio-features.pt `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2-cdrive-20261117\librispeech-train-clean-100-long-curve-v1-attempt-01 `
  --device cpu --experiment-id path-a-audio-librispeech-train-clean-100-long-curve-v1 `
  --seed 17 --epochs 100
```

Started `2026-11-16T20:06:35Z`, completed `2026-11-16T20:22:16Z` (15m 41s total). The training/validation loop took 866.59 seconds; baseline and selected-checkpoint validation each took about 1.05 seconds. There were 86,900 optimizer updates (869 per epoch). Each epoch traversed 13,902 unique paired targets under both conditions (27,804 query presentations); total was 2,780,400 query presentations. At selected epoch 40, cumulative consumption was 34,760 updates / 1,112,160 query presentations. These are presentations, not unique examples: the run used the same 13,902 unique train targets throughout. Device was CPU; no GPU/VRAM was used, no cloud compute or cost was incurred. Environment: Python 3.11.9, Windows 10 build 26300, PyTorch 2.6.0+cu124, CUDA runtime 12.4 available but unused, Transformers 5.6.2, PyAV 18.1.0, NumPy 2.4.6. CPU model and system RAM were not captured.

The script reloaded `best-readout.pt` after training and recomputed selected validation metrics/predictions. `final-readout.pt` remains the separate epoch-100 state; it did not overwrite the selected best.

## Learning curve

All rows are on the same 3,467 held-speaker validation targets per condition. `Selector NLL` is the equal-weight mean of random and hard NLL. Training CE is a rolling epoch aggregate and should not be interpreted as a fixed-example train/validation gap.

| Epoch | Train CE | Selector NLL | Random acc. | Random NLL | Hard acc. | Hard NLL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7453 | 0.4206 | 87.45% | 0.3245 | 80.24% | 0.5167 |
| 4 | 0.0812 | 0.2336 | 94.40% | 0.1646 | 88.40% | 0.3025 |
| 8 | 0.0359 | 0.2380 | 94.26% | 0.1661 | 89.36% | 0.3100 |
| 16 | 0.0057 | 0.2446 | 94.72% | 0.1754 | 90.40% | 0.3139 |
| 32 | 0.0273 | 0.2614 | 95.04% | 0.1704 | 90.57% | 0.3523 |
| 40 (selected) | 0.0001 | **0.2073** | **96.02%** | **0.1312** | **92.90%** | **0.2834** |
| 48 | 0.0174 | 0.3437 | 93.39% | 0.2445 | 88.52% | 0.4430 |
| 64 | 0.0081 | 0.3127 | 94.78% | 0.2057 | 89.30% | 0.4197 |
| 80 | 0.0003 | 0.2974 | 95.07% | 0.2157 | 91.52% | 0.3790 |
| 100 (final) | 0.0001 | 0.3686 | 94.61% | 0.2592 | 90.25% | 0.4779 |

### Selected checkpoint metrics (epoch 40)

| Condition | N | Accuracy | NLL | Brier | ECE (15 bins) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random distractors | 3,467 | 96.02% | 0.1312 | 0.0635 | 0.0235 |
| Hard distractors | 3,467 | 92.90% | 0.2834 | 0.1146 | 0.0432 |
| Equal-condition macro | 6,934 | 94.46% | 0.2073 | 0.0891 | 0.0333* |

`*` Macro ECE is the arithmetic mean of condition ECE. Pooled ECE is 0.0331. The separate Speech Commands readout baseline on these same validation decisions remains 25.58% random / 25.90% hard accuracy (macro NLL 2.2038); it is a transfer baseline, not an appropriate quality target for this task.

At epoch 100, random/hard Brier were 0.0914 / 0.1634 and ECE were 0.0372 / 0.0706. At selected epoch 40 they were 0.0635 / 0.1146 and 0.0235 / 0.0432. The longer run improved the best hard-condition Accuracy from the prior four-epoch endpoint's 88.38% to 92.90% on this same development validation, while also lowering hard NLL from 0.3079 to 0.2834. The prior four-epoch state had no recoverable historical best, so only its final surviving checkpoint is comparable. Because that run used CUDA and the long run used CPU, treat the numerical difference as supportive development evidence, not a pure compute-budget causal estimate.

## Interpretation

The result supports the limited claim that the four-epoch run had not exhausted useful optimization for this specific task: the 100-epoch run reached a better selected validation checkpoint at epoch 40, with both random and hard conditions improving over the prior final checkpoint. It does not establish that longer training is the only cause; initialization, CPU/GPU numerics, and selection repair are limitations. Nor does it support endless training: selector NLL worsened after epoch 40, and by epoch 100 hard NLL had risen to 0.4779 from 0.2834 while train CE remained very low. This is direct evidence of overfitting/calibration degradation on the exposed validation set. The model's Accuracy can remain high while NLL/ECE worsen, so selecting on the fixed predeclared equal-condition NLL was appropriate.

The run is complete as an execution. The task remains narrow English audiobook transcript-choice and repeatedly observed development validation. No claim is made about 95–98% performance across audio tasks, general multimodal decision ability, quantized footprint, or product readiness. Do not tune repeatedly on these validation IDs; the next quality claim needs a distinct, predeclared task and untouched source groups.

## Artifacts and preservation

All bulky artifacts remain local and outside Git. The existing four-epoch run, prepared corpus, feature cache, runtime bundle, Teacher v1/v2, sealed audit, and GPU process were not modified. No sealed audit or official test data was loaded.

| Artifact | SHA-256 |
| --- | --- |
| `best-readout.pt` (1,379,448 bytes) | `f7ce2d1b04f6dde98e44cc1343916d594e2c095ad1d70f335323d0dbb5bc1f3a` |
| `final-readout.pt` (1,379,456 bytes) | `2d775cb64cfac729ee5afc2fe9337246994338c05d3fe0785c6fec9f30281e98` |
| `training-history.json` | `68292d94a69175f860c8a26d2c44dc5f05b7f682bbcc167f2a19f0f5acc2e04b` |
| `selected-validation-predictions.jsonl` | `35864035d7cec565153518e9c1fa8fd30254b04e480052ca5883631eaf8942aa` |
| `baseline-validation-predictions.jsonl` | `54546394a7cd1acd3a9b879ceab6769ce0ecc082cd8d55302d4e8cda53897384` |
| `run-report.json` | `aa2af92f6408e8d05bab432974d0d9d6b89310eb704c4c0e0a06cfffacaadb1bb` |

## Next step

Keep epoch-40 as a local development checkpoint only. Do not promote or quantize it. Stop expanding this exposed audiobook validation task. Continue the broader objective by selecting a different decision task with a real modality-dependent label, source rights and group splits pinned before training, then run an independent evaluation only after freezing the protocol.
