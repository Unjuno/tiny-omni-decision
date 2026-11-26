# Physical mirror: training-only frozen diagnostic

First64 pinned training images;256 questions, unchanged exact labels/options.
Original re-extraction matches cached spatial features exactly (max abs0).
No validation or sealed audit inference, training, selection, calibration or ensemble.

| Condition | Task | Accuracy | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| identity | all | 0.929688 | 0.185345 | 0.102399 | 0.041879 |
| identity | color | 1.000000 | 0.004917 | 0.000605 | 0.004752 |
| identity | texture | 0.968750 | 0.070485 | 0.041253 | 0.028513 |
| identity | count | 0.750000 | 0.665236 | 0.367730 | 0.159718 |
| identity | shape | 1.000000 | 0.000743 | 0.000008 | 0.000741 |
| physical_mirror | all | 0.878906 | 0.268335 | 0.154546 | 0.039386 |
| physical_mirror | color | 1.000000 | 0.018785 | 0.008818 | 0.015558 |
| physical_mirror | texture | 0.906250 | 0.271372 | 0.143662 | 0.070582 |
| physical_mirror | count | 0.609375 | 0.781185 | 0.465587 | 0.139683 |
| physical_mirror | shape | 1.000000 | 0.002000 | 0.000119 | 0.001959 |
| cell_reversal | all | 0.890625 | 0.229501 | 0.132128 | 0.042143 |
| cell_reversal | color | 1.000000 | 0.027133 | 0.015767 | 0.021397 |
| cell_reversal | texture | 0.984375 | 0.070528 | 0.027675 | 0.017149 |
| cell_reversal | count | 0.578125 | 0.819023 | 0.485014 | 0.164046 |
| cell_reversal | shape | 1.000000 | 0.001320 | 0.000056 | 0.001302 |

## Interpretation

Physical mirroring reduces overall training-subset Accuracy92.97% to87.89%
and exact counting75% to60.94%. Cell reversal gives89.06% overall and57.81%
counting. The two interventions are therefore not interchangeable.

The actual image transformation also exposes fragility, so the previous effect
is not only an artifact of swapping contextual cells. However,64 training images
cannot establish generalization or distinguish encoder versus readout responsibility.
Do not select a model using these results or claim the95% product objective.

A bounded training-only mirror-augmentation experiment is now justified: retain
the frozen encoders and compare a matched original-only head against a head
trained with physically mirrored training observations, with equal update budgets.
Predefine the comparison before further validation use; no feature-permutation
augmentation or extra inference averaging should be silently substituted.

## Paired descriptive uncertainty

- physical_mirror minus original: -5.078pp;95%[-8.594,-1.953]pp.
- cell_reversal minus original: -3.906pp;95%[-7.422,-0.391]pp.

64-image paired percentile bootstrap,10,000 draws,seed17; convenience training
subset, not a blind population estimate. No multiple-comparison correction.

Local diagnostic2.343s, peak allocated VRAM292,862,464bytes, cloud$0.
Frozen DINO/MiniLM and40,331,392-parameter stack unchanged. Saved features and
all predictions under `C:/CodexArtifacts/pretrained-reuse-r2/dinov2-physical-mirror-01`.
Companion result/analysis JSON preserves metrics, checkpoint/cache/script hashes
and feature/prediction hashes. No existing artifact overwritten.
