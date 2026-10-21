# Training-only candidate permutation: rejected candidate

## Verified execution

The precommitted protocol (`e220583`) completed 1,024 optimizer updates with
seed 17, implementation `a8aa20e`. The only intervention relative to the joint
readout control is training-time option permutation. All 8,192 realized option
lists were reconstructed: 7,223 changed order and 969 retained their original
order. Parent order, validation order, tokenization statistics, initialization,
optimizer settings and step-zero metrics match the control. There are 5,120
unique parent examples and 3,072 repeated encounters; permutations are not new
independent examples.

Selected step = final step = 1,024; best reload is exact. Execution passed,
quality adoption failed. Resident parameters: 22,713,601; trainable: 10,647,169.
The timed training/validation loop took 223.76 seconds and peak allocated VRAM
was 964,418,048 bytes. The fixed order diagnostic took another 33.98 seconds,
including tokenization. These timings exclude initial model loading. Local
RTX 3080 Laptop GPU only; no cloud expenditure.

## Fixed validation results

| Selected model / source | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Independent matched / original | 0.63543 | 0.83661 | 0.47112 | 0.01932 |
| Joint control / original | 0.58307 | 0.91801 | 0.52414 | 0.03970 |
| Joint shuffled / original | 0.54803 | 0.96949 | 0.55120 | 0.02097 |
| Independent matched / curriculum | 0.84082 | 0.37537 | 0.21432 | 0.01682 |
| Joint control / curriculum | 0.40088 | 1.28778 | 0.69604 | 0.08032 |
| Joint shuffled / curriculum | 0.25293 | 1.43707 | 0.76025 | 0.00437 |

Original validation has 2,540 questions / 732 states; curriculum has 2,048
questions / 256 states. State-paired bootstrap (10,000 draws, seed 17,
percentile 95% intervals) estimates shuffled-minus-control Accuracy differences
of -3.50 percentage points [-5.61, -1.40] and -14.79 [-17.29, -12.30], respectively.
NLL and Brier also worsen on both sources. Low curriculum ECE accompanies low
Accuracy and does not establish useful decision quality. These are repeatedly
observed development sets with checkpoint selection, not blind audit evidence.

| Step | Equal-source mean validation NLL |
|---:|---:|
| 0 | 1.317061 |
| 128 | 1.312852 |
| 256 | 1.312087 |
| 384 | 1.285498 |
| 512 | 1.264852 |
| 640 | 1.251749 |
| 768 | 1.230252 |
| 896 | 1.208396 |
| 1024 | 1.203282 |

The endpoint is still improving. This experiment does not establish a capacity
ceiling or rule out longer training, but supplies no quality justification to
adopt this candidate or automatically extend its budget.

## Candidate-order diagnostic

Natural-order predictions replay exactly. Reversing the same options gives
original Accuracy 0.54213 / NLL 0.99354 and curriculum Accuracy 0.25732 /
NLL 1.43843. The aggregate order gap shrinks, but answer identities change for
897/2,540 original and 1,741/2,048 curriculum questions. Maximum probability
changes are 0.86520 and 0.46449. Thus permutation stability is not solved;
similar aggregate Accuracy conceals many changed answers. No permutation
averaging, calibration fitting or diagnostic-based checkpoint selection occurred.

## Artifacts and reproduction

- Run: `C:/CodexArtifacts/pretrained-reuse-r2/joint-shuffled-training-01`
- Diagnostic: `C:/CodexArtifacts/pretrained-reuse-r2/joint-shuffled-order-diagnostic-01`
- Training: `python -m scripts.train_joint_shuffled --output C:/CodexArtifacts/pretrained-reuse-r2/joint-shuffled-training-01` (original invocation; existing outputs are protected, use a new directory for any authorized reproduction).
- Diagnostic: `python -m scripts.evaluate_joint_order --candidate shuffled`
- Report verification: `python -m scripts.report_joint_shuffled` (refuses existing report).
- Best SHA-256: `3a93ce9af6db773182294ebd31589f46ba93ba1abf3bbde0f4b63df4a5f1dadc`
- Final SHA-256: `9bc939417b4557a84274690ebc712c5af349c8313c88b0ef5aba71eac7f46d54`

[Machine-readable evidence](R2_JOINT_SHUFFLED_RESULTS_20261012.json) includes all
curves, per-task metrics, effective configuration, input/output hashes and paired
intervals. Best/final files are separate; serialization hashes alone do not imply
different tensor values at the same step. This is optimizer-reset continuation,
not exact resume. Legacy artifacts, multimodal towers and sealed audit are
unchanged. No product promotion. Retain the independent candidate baseline;
text compositional transfer and video reasoning remain major unmet quality gates.
