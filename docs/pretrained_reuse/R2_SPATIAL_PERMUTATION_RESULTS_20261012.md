# Fixed spatial permutation diagnostic

Execution PASS on original CUDA backend; identity probabilities match exactly.
No training, selection or calibration. Same512 development images/2048 questions.
Frozen spatial epoch16 checkpoint retained. CLS unchanged in every condition.

| Intervention | Task | Accuracy | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| identity | all | 0.900391 | 0.253458 | 0.140201 | 0.022108 |
| identity | color | 0.974609 | 0.059791 | 0.035664 | 0.012743 |
| identity | texture | 0.923828 | 0.228276 | 0.107183 | 0.043987 |
| identity | count | 0.705078 | 0.722229 | 0.416147 | 0.059495 |
| identity | shape | 0.998047 | 0.003538 | 0.001810 | 0.002769 |
| horizontal | all | 0.878906 | 0.316368 | 0.168910 | 0.032892 |
| horizontal | color | 0.957031 | 0.158349 | 0.068658 | 0.024350 |
| horizontal | texture | 0.914062 | 0.267784 | 0.128866 | 0.054920 |
| horizontal | count | 0.646484 | 0.835001 | 0.475814 | 0.059433 |
| horizontal | shape | 0.998047 | 0.004339 | 0.002300 | 0.002964 |
| seed17 | all | 0.762695 | 1.286294 | 0.350677 | 0.138639 |
| seed17 | color | 0.931641 | 0.250739 | 0.110162 | 0.032009 |
| seed17 | texture | 0.882812 | 0.531134 | 0.193493 | 0.080262 |
| seed17 | count | 0.236328 | 4.349337 | 1.095183 | 0.460125 |
| seed17 | shape | 1.000000 | 0.013966 | 0.003870 | 0.012583 |

Horizontal cell reversal lowers overall Accuracy by2.15pp (paired512-image
bootstrap95%[-3.32,-0.98]pp), and count70.51% to64.65%. One seed17 cell
permutation lowers overall Accuracy by13.77pp ([-15.28,-12.26]pp), count23.63%.
Bootstrap10,000 draws,seed17; development selection bias remains.

This demonstrates dependence on cell arrangement. It does not establish whether
that dependence is useful geometry or a shortcut: contextual encoder tokens retain
their original positional information, and shuffled features may be off-distribution.
These are not flipped images. Do not infer physical mirror invariance or a causal
object-counting mechanism. No model adoption or permutation tuning follows.

The next useful spatial experiment should operate on actual transformed training
images with unchanged labels before evaluating generalization; feature shuffling
alone is insufficient evidence to impose invariance on this encoder.

Attempt01 CPU identity difference1.359e-5 failed the fixed1e-5 gate. Attempt02
stopped before metrics due to CPU option/GPU index mismatch; corrected in03.
Both failed output directories retained. No threshold relaxation.

Measured inference diagnostic time1.009s, local GPU, cloud cost$0.
Artifact: `C:/CodexArtifacts/pretrained-reuse-r2/dinov2-spatial-permutation-03`.
Companion JSON contains exact permutations, checkpoint/cache/script hashes and metrics.
The unchanged baseline remains90.04% on this narrow task, below the product target.
