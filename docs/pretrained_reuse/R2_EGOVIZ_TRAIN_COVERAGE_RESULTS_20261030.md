# EgoViz expanded training-session comparison

The fixed-budget CPU experiment completed 5,632 updates. The preregistered
[protocol](R2_EGOVIZ_TRAIN_COVERAGE_PROTOCOL_20261030.md) and training code were
committed before fitting; the run records source commit `ac997cc692657a40583df426e4bc151106ab8025`.
The same frozen VideoPrism/MOMENT encoders and 81,920-parameter uniform readout
were used. Training grew from 332 examples in 24 sessions to 773 examples in
48 sessions; the original 307 development examples from 20 sessions retained
identical IDs, order, video features and sensor features. Group, chunk, sample
and raw-media SHA overlap checks passed. No sealed media were read.

| Updates | Baseline dev accuracy | Expanded dev accuracy | Baseline dev NLL | Expanded dev NLL | Expanded train accuracy |
|---:|---:|---:|---:|---:|---:|
| 352 | 53.75% | 56.68% | 1.2258 | 1.1790 | 81.24% |
| 704 | 53.09% | 57.65% | 1.2410 | 1.1783 | 88.10% |
| 1,408 | 50.49% | 56.68% | 1.2610 | 1.1808 | 92.37% |
| 2,816 | 47.88% | 55.37% | 1.2854 | 1.1901 | 96.25% |
| 4,224 | 43.97% | 55.05% | 1.3054 | 1.1976 | 97.41% |
| 5,632 | 41.37% | 55.05% | 1.3110 | 1.2001 | 96.38% |

The selected expanded checkpoint at update 616 achieved development Accuracy
56.68%, NLL 1.1746, Brier 0.6383 and ECE 0.2215. The prior baseline selected
at update 176 achieved 56.03%, 1.2147, 0.6582 and 0.2260 respectively. The
paired 20-session bootstrap (2,000 draws, seed 17) gives the selected Accuracy
difference as +0.65 percentage points, 95% percentile interval [-3.90, +5.02];
baseline-minus-expanded NLL is 0.0401 [0.0195, 0.0601], and Brier is 0.0199
[0.0082, 0.0306]. These intervals do not correct for checkpoint selection.
The expanded selector also included update 1 while the old selector began at
update 11; fixed-update comparisons do not have that selection mismatch.

At update 616 expanded train Accuracy was 87.32%, versus 56.68% development;
at 5,632 it was 96.38% versus 55.05%. Development NLL rose from its selected
1.1746 to 1.2001 as training continued: more diverse training data reduced
the late degradation relative to the old run but did not eliminate overfitting.
The extra sessions were chosen from a different deterministic group pool; both
training-set composition and shuffled batch sequence changed. This is an
exploratory comparison on repeatedly observed development data, not a blind
estimate or proof of product-level performance. Participant independence is
unknown. The task is video-to-IMU window matching, not general four-modality
decision accuracy. No general conclusion about grokking is established.

The full run consumed 174,149 sample exposures (74,798 original and 99,351
new), including 225 complete passes and seven batches of a partial pass. The
selected checkpoint had consumed 19,064 exposures. All 773 examples appeared.
The expanded fitting took 58.29 seconds on four CPU threads; data extraction
was CPU-only. No paid compute was used. The first acquisition stopped safely
after 23 sessions on insufficient disk space; after space was made available,
the 23 packages were copied and individually revalidated in a new output root
before the remaining session was acquired. The original run was preserved.

Input hashes: training JSONL `9e108f5d5ba2c756321c96b4ab47a0a387b146883367066499db7eea214d8245`;
development JSONL `55fe808c824825762da63d2c0f376dde30d61986a8fb6481c0768f661aa9d232`;
video features `17b14afdfef6255a11662cd5989732ef4a15a11bdea737f4b8582f233dc9b28c`;
sensor features `4c2201b13403ce5f395591b876fdd78ae9b3a99cff7898556eaa3694f28af431`.
Selected and final checkpoint SHA-256 are `ac399c77e9d3d6dd647d6ae8a924068cd9dff20ec00b90b82f69d34733f01885`
and `4936aa2f696d41ede8419067af37daf251292afdb22578a766327bbc5de23bde`.
All six matched-update checkpoints and both selected checkpoints were reloaded
in a fresh process and their saved logits reproduced exactly; saved probability
vectors and option permutations passed validation. Large model and media files
remain outside Git. Full history, exposure accounting, predictions and paired
comparisons are in `C:\CodexArtifacts\egoviz-train-coverage-budget-20261030-attempt-01`.
