# R2-M parameter-matched sensor-absent control (2026-11-21)

## Question and result

Does the selected audio+vibration result improve merely because its gated
readout has more parameters than the earlier audio-only model, or does paired
vibration content contribute on the observed development participants?

One fresh control run used the exact same two-stream model, initialization
seed, train/calibration records, optimizer, schedule and total parameter count
as the selected attempt-03 audio+sensor arm, while masking the sensor for every
training, calibration and validation example. The control had the same
180,387 total/trainable parameters. Checkpoints were selected by the same
minimum calibration macro-NLL rule. Validation was not used for selection.

On the same 762 validation IDs, paired fusion and the sensor-absent control
were close on sample-weighted Accuracy, while fusion was better on macro
Accuracy and both macro proper scores. The result was not consistent by
participant: fusion improved on participant 9 and regressed on participant 15.
This supports a sensor-content contribution to some class-balanced decisions
on these two already-observed participants, but does not establish a reliable
new-person benefit or justify product adoption.

| Selected checkpoint | Total parameters | Sample Acc. | Macro Acc. | Macro NLL | Macro Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|---:|
| Attempt 04, sensor absent, epoch 4 | 180,387 | 75.72% | 67.81% | 1.0630 | 0.4957 | 0.1318 |
| Attempt 03, paired audio + sensor, epoch 9 | 180,387 | 75.85% | 72.46% | 0.8908 | 0.3870 | 0.1006 |

Fusion minus sensor-absent point differences were +0.13 percentage points in
sample Accuracy, +4.65 pp in macro Accuracy, −0.1724 macro NLL, and −0.1088
macro Brier. Paired trial bootstrap (10,000 replicates, seed 17, resampling
within participant × target-activity strata) gave:

| Metric delta, fusion minus control | Mean delta | Percentile 95% interval |
|---|---:|---:|
| Sample Accuracy | +0.11 pp | −2.36 to +2.62 pp |
| Macro Accuracy | +4.63 pp | +1.66 to +7.55 pp |
| Macro NLL | −0.1724 | −0.2566 to −0.0882 |
| Macro Brier | −0.1088 | −0.1385 to −0.0781 |

The participant-specific directions conflict:

| Validation participant | Examples | Sensor absent Acc. / macro Acc. / macro NLL | Paired fusion Acc. / macro Acc. / macro NLL |
|---:|---:|---:|---:|
| 9 | 376 | 64.10% / 66.10% / 1.1659 | 71.54% / 72.79% / 0.8973 |
| 15 | 386 | 87.05% / 87.15% / 0.4641 | 80.05% / 80.13% / 0.7055 |

The intervals resample trials conditional on these two participants; they do
not measure participant-population uncertainty. Validation has already been
observed in prior development runs and is not a sealed or blind evaluation.
The conventional attempt-03 audio-only model is also retained for context; the
primary capacity control is the same-topology, equal-total-parameter model
trained without sensor input.

## Controlled run

- Source/task: the admitted Zenodo `10.5281/zenodo.7937591` `multisensor-1`
  frozen-feature corpus and the same supplied-activity, 21-choice task as
  attempt 03. The three unsupported labels (`11_typing`, `14_sweep`,
  `15_vacuum-cleaning`) remain excluded by the pre-existing split-coverage
  rule; no examples or labels were changed.
- Split: train participants 2, 4, 7, 8, 10, 13, 14; calibration 12, 17;
  validation 9, 15. The split manifest SHA-256 is
  `0f1539f3ca4fa59fb854f62def1c538e1f5851576444b678af4aa050a393a6f0`.
- Data: 5,424 supported train examples, 760 calibration examples, and 762
  validation examples. Train-only per-axis sensor normalization and all frozen
  feature arrays were reused read-only. Sealed-audit participants/signals were
  not loaded.
- Architecture: `TimeAwareGatedFusion(audio=384,sensor=3,query=21,width=96,
  heads=4)` plus scalar option scorer. Total/trainable parameters: 180,387.
  The control masks sensor presence and all sensor tokens; it keeps the same
  model topology and state dimensions as the paired-input model.
- Initialization: fresh weights, seed 17, matching attempt-03 seed and model
  construction. `train_arm` resets Python, NumPy and PyTorch seeds before model
  construction. No prior checkpoint was warm-started.
- Training: 80 epochs; batch 32; AdamW LR `1e-3`, weight decay `0.1`; cosine
  schedule; train-only inverse-frequency weighted cross entropy; random option
  order per training batch; no early stopping. The exact effective configs
  differed only in declared arm list, modality condition, and source
  provenance.
- Selector: minimum calibration macro NLL. Best epoch 4; the best checkpoint
  was kept separately from final epoch 80. The best checkpoint corresponds to
  21,696 cumulative training example presentations and all 5,424 unique train
  examples. The completed run used 13,600 optimizer updates / 433,920
  presentations.
- Learning curve: train Accuracy rose to 100% and weighted CE fell to 0.0001
  by epoch 80, while calibration macro NLL rose from 0.8171 at selected epoch
  4 to 3.6057 at epoch 80. This is clear late overfitting; the final checkpoint
  was not used for the reported comparison.

| Epoch | Train Accuracy | Train weighted CE | Calibration macro Acc. | Calibration macro NLL | Calibration macro Brier | LR |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48.53% | 1.6069 | 63.78% | 1.1297 | 0.4597 | 0.001000 |
| 2 | 81.47% | 0.5227 | 71.98% | 0.9007 | 0.3724 | 0.0009996 |
| **4 selected** | 89.60% | 0.2817 | 76.06% | **0.8171** | 0.3286 | 0.0009965 |
| 8 | 93.01% | 0.1865 | 77.85% | 0.9911 | 0.3516 | 0.0009812 |
| 16 | 96.92% | 0.0926 | 79.14% | 1.1786 | 0.3443 | 0.0009157 |
| 32 | 98.51% | 0.0408 | 80.16% | 1.4667 | 0.3339 | 0.0006731 |
| 64 | 99.98% | 0.0011 | 77.79% | 3.2959 | 0.4086 | 0.0001073 |
| 80 final | 100.00% | 0.0001 | 77.53% | 3.6057 | 0.4201 | 0.0000004 |

## Execution provenance

- Repository branch: `codex/pretrained-reuse-next-20261115`; source commit at
  run time: `6ed081774727aae6e702bf80cb056edb9d70c9ac`. The worktree had
  uncommitted changes for this controlled experiment. The effective config
  records SHA-256 for the runner, feature/label helpers and fusion module.
- New output directory:
  `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-param-matched-audio-v1-attempt-04`.
  Existing attempts 01–03, feature cache, pair index, and checkpoints were
  read-only and unchanged.
- Effective config exact-file SHA-256:
  `e45e97b9ab19a4ee64eee0c52f639d97658ce7a6da9fe6a1d3103ce4cdee72bd`;
  internal config payload SHA-256:
  `a8498a50cf345cd9b89c9bb4a77480aa3fbc5927fc9c4d2dd8107285feb64e69`.
- Training runner SHA-256 at execution:
  `5701563080ba5a0964d2662497bb22b08d3bb3a6f47ce5ef30c965d54a616a6a`;
  R2-M activity helper SHA-256:
  `5f074042a4600bdc152a1537ebe89facc0196b8d880d8483311647b9ecf9ee73`;
  fusion module SHA-256:
  `5c9c46a0b9292cc3ed96a210abb22c6976aabe2835165abef1cdde6981a8eef8`.
- Feature-cache index SHA-256:
  `61758bebaf53ad810737d31e7955820bf7094e2166460c5ce595ea0c475c6498`;
  allowed-trial metadata SHA-256:
  `faee790bfaa53016e7c92f0ac545f0a7dfe23cd126fa7e1287b10d8d8b0282c0`;
  pair-index internal SHA-256:
  `d1e1c74a1114fc0d071f71a50876b852e8d3fce8919fde9696368d4ebf365276`.
- Runtime: Python 3.11.9, PyTorch 2.6.0+cu124, CUDA runtime 12.4, NumPy
  2.4.6; NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB. Peak PyTorch allocation
  was 72,146,432 bytes. The training arm took 535.85 seconds; total run
  process time was 538.75 seconds. No cloud compute was used.
- Best checkpoint SHA-256:
  `f61f91a2b7cc9142858332a42d2099e261088dfdb9f68dd55a5cc545f392714d`;
  final checkpoint SHA-256:
  `3f4c21458fdc6db69938b4b8dac621bda275623a77f92dc31e1f35c52d4a5657`;
  training-history SHA-256:
  `04daef757c8cc0e1369b30adaa574ab23adcfa827754785adc4339bab02e8675`.
- Paired comparison JSON SHA-256:
  `da407d030db2e4675fa41333932340a818d81211c67c52db69465f2a0a4c2569`.
- The paired comparison JSON, exact sample IDs, per-example probabilities and
  both configs/results are retained in the new attempt directory. Comparison
  bootstrap: 10,000 paired resamples, seed 17, percentile intervals, strata
  participant × target activity.
- Sealed audit was not loaded. No model was promoted, quantized or merged.

## Decision and next work

The equal-total-parameter control weakens the “fusion only wins because it is
larger” explanation for macro Accuracy, NLL and Brier on this validation set.
The near-zero sample Accuracy difference and the opposing participant-level
effects keep the evidence mixed. The controlled comparison does **not** show
that sensor fusion reliably helps unseen people, and the 80-epoch curve
confirms this readout can overfit badly. Do not expand this specific run or
promote it. Preserve it as a mechanism diagnostic and prioritize a new
participant-grouped, rights-cleared task/source before drawing a population
claim.

Attempt 03's selected checkpoint, the frozen feature cache and the source
corpus were not modified. No audit data or signals were inspected.
