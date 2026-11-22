# R2-M sensor dependence diagnostic — 2026-11-21

## Scope

This was a fixed-checkpoint input-dependence check on the already observed
development validation split for the supplied-activity R2-M experiment. It
performed no training, checkpoint selection, or sealed-audit access. It does
not establish general sensor quality or new-participant performance.

The selected attempt-3 audio+sensor gated checkpoint was held fixed:

- Attempt: `r2m-supplied-activity-decision-v1-attempt-03`
- Selected checkpoint: epoch 9
- Checkpoint SHA-256: `a9ee14679af167e1461d1758c7c7e69fbbf380fc6f98799d411588baa6637347`
- Effective config SHA-256: `89aeb7cede1edece23cd6a80039aeff8580b49793b8b82fd7c6e11671b45c2fe`
- Validation: 762 records, participants 9 and 15
- Validation ID-order SHA-256: `5afc8fda15f22ec3f48e1d557cbd2b66c172f0d3c692256b2c821e5de0484b4e`
- Feature-cache index SHA-256: `61758bebaf53ad810737d31e7955820bf7094e2166460c5ce595ea0c475c6498`
- Allowed metadata SHA-256: `faee790bfaa53016e7c92f0ac545f0a7dfe23cd126fa7e1287b10d8d8b0282c0`

The original checkpoint reload reproduced saved predictions exactly (maximum
absolute probability delta `0.0`). The frozen feature cache, checkpoint,
training histories, and earlier one-row-control outputs were left unchanged.

## Why the old shuffle was inadequate

The training runner sorts validation rows by `sample_id`; this ID begins with
the activity label. Its original `val_rows[1:] + val_rows[:1]` sensor rotation
therefore paired nearly every record with a sensor from the same activity.
The old control's small metric change cannot be interpreted as evidence that
the fitted readout ignores sensor content. The original outputs are preserved
with that limitation recorded.

## Corrected intervention

For each participant separately, validation rows were sorted into contiguous
activity blocks. The donor sensors were shuffled within each activity block,
then the donor order was circularly shifted by half the participant's rows.
The function fails closed if any activity occupies more than half of a
participant's rows. This creates a one-to-one donor permutation satisfying:

- every donor is from the same participant as its recipient;
- every donor has a different activity label;
- every validation sensor is used exactly once.

All 762 rows met those checks. This preserves the participant and marginal
sensor distribution while intentionally making sensor evidence conflict with
the audio/target. It is a sensitivity intervention, not a naturally occurring
input condition.

## Results

| Condition | Sample Accuracy | Macro Accuracy | Sample NLL | Macro NLL | Sample Brier | Macro Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paired audio + sensor | 75.85% | 72.46% | 0.8249 | 0.8908 | 0.3478 | 0.3870 | 0.1006 |
| Same-participant, different-activity sensor | 68.64% | 62.88% | 1.3298 | 1.5259 | 0.4519 | 0.5141 | 0.1505 |

The option argmax changed on 140/762 records. A paired bootstrap over records,
stratified by participant × target activity, used 10,000 replicates and seed
17:

| Mismatched minus paired | Mean difference | 95% percentile interval |
|---|---:|---:|
| Sample Accuracy | −7.21 pp | −9.32 to −5.12 pp |
| Macro Accuracy | −9.56 pp | −12.29 to −6.89 pp |
| Macro NLL | +0.6344 | +0.5303 to +0.7405 |
| Macro Brier | +0.1268 | +0.0986 to +0.1563 |

Both observed participants declined: participant 9 macro Accuracy was 72.79%
with paired inputs and 64.55% with mismatched sensor; participant 15 was
80.13% and 73.99%, respectively. These intervals resample trials within only
two participants, so they do not estimate participant-population uncertainty.
The validation set was already used for development reporting, and the
selected checkpoint itself was chosen on separate calibration records.

This confirms that the selected fusion's outputs respond to activity-
inconsistent sensor content on these two participants. It does not prove the
same gain on unseen participants, nor isolate the sensor's contribution from
the larger gated architecture. Attempt 3's fused-vs-audio-only model comparison
also changes trainable parameter count; a parameter-matched baseline and a
fresh participant-grouped evaluation are still needed.

## Reproducibility artifacts

The fixed-checkpoint runner is
[`evaluate_r2m_activity_sensor_dependence.py`](../../scripts/evaluate_r2m_activity_sensor_dependence.py).
Future training-run sensor controls use the same-participant,
different-activity donor permutation, with CPU tests for determinism, group
preservation, mismatched labels, and fail-closed class support.

CUDA RTX 3080 Laptop execution reported 35,268,608 bytes peak allocated
PyTorch memory. Diagnostic wall time was not measured. No cloud compute was
used. The full report and per-example predictions are stored outside Git at:

`D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-supplied-activity-decision-v1-attempt-03\sensor-dependence-corrected-replay`

The diagnostic JSON records exact config/checkpoint/data hashes and source
hashes for the diagnostic runner, donor helper, and model runner. Sealed audit
was not loaded; no training or checkpoint changes occurred.
