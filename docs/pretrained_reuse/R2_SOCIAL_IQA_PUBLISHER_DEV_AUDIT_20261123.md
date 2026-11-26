# Social IQa official-dev audit — 2026-11-23

## Decision

The selected frozen MiniLM-L6 supplied-option scorer transferred to Social
IQa's previously unopened official public-dev partition at **43.40% accuracy**
(1,954 three-choice questions), compared with 53.81% on the internal
train-only validation used for checkpoint selection. This is 10.41 percentage
points lower on a different, publisher-held-out cohort. The result is above
three-choice uniform chance but remains far below the project's 95–98%
aspiration. Treat this as a one-shot public-dev generalization audit, not a
blind test or a clean estimate of a model-family ceiling.

Do not tune, select another checkpoint, or repeat evaluation on these dev
labels. The model, readout, config, and training run were not changed.

## Data and split-integrity checks

- Dataset: [AI2 Social IQa](https://allenai.org/data/socialiqa), pinned source
  revision `8835ceb9141d7896d9d968634a9b21ae440e3ec5`; dataset card body
  declares CC BY 4.0.
- Source archive SHA-256:
  `ee073914a0fc33265cfbcfc50ec20df9b2e07809c3f420f599138d1f394ef5c3`.
- Full publisher train contains 33,410 rows / 33,000 normalized context
  groups. The fitted readout used 26,751 rows from that publisher train split;
  its internal validation used 6,659 rows and selected epoch 8.
- Official dev contains 1,954 rows / 1,930 normalized context groups. Target
  counts are 643 / 654 / 657 for answer positions 1 / 2 / 3.
- **Before opening dev labels**, the evaluator compared the full official
  training records against dev text. Normalized context overlap,
  normalized full-example overlap, and normalized question/options overlap
  were all zero. The official dev question/options were also checked against
  the frozen 256-token input limit: 5,862 pairs, maximum 71 tokens, zero
  truncations. Only after both checks passed did the script read
  `dev-labels.lst`.
- Social IQa provides no stable ATOMIC source-event identity in these records.
  Exact content is disjoint, but paraphrased shared seed-event overlap remains
  unknown. The benchmark is public and may already have been examined outside
  this project.
- The project sealed audit was not opened.

## Frozen model and procedure

- Encoder: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, weight SHA-256
  `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.
- Existing selected readout SHA-256:
  `066e943de2dbe493f6257c32c24b11c2566cd36507326c2d5bef7b868db33fa3`.
- Parameters before quantization: 22,713,216 encoder + 49,409 readout =
  **22,762,625 total**. No quantization or export was performed.
- Existing training report SHA-256:
  `870bd0c41e45c5b51978cc5ac8c02ca69b3f2535928cd7c218243a0ead2bb072`;
  config SHA-256:
  `737e7c062f633c82a76022980bf03c5a282abc9e76772a56b486568f7dfdf5cc`.
- The selected epoch-8 checkpoint was loaded as-is. There was no fitting,
  validation, calibration, prompt change, or selection on official dev.
  Save/reload reproduced logits and probabilities exactly (maximum delta 0).
- CPU-only, Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2,
  two Torch threads. Feature extraction took 39.45 seconds. No GPU or cloud
  compute was used.

## Results

| Metric | MiniLM | Uniform-choice reference |
|---|---:|---:|
| Accuracy | **0.4340** | 0.3333 expected |
| Macro Accuracy by correct option position | 0.4340 | 0.3333 expected |
| NLL | **1.0739** | 1.0986 |
| Brier | **0.6485** | 0.6667 |
| ECE-15 | 0.0763 | — |
| Mean confidence | 0.5101 | 0.3333 |

A 10,000-replicate PCG64 bootstrap (seed 17) resampled all 1,930 normalized
context groups with replacement and retained every question from each sampled
context. Percentile 95% intervals were Accuracy `[0.4116, 0.4560]`, NLL
`[1.0518, 1.0970]`, and Brier `[0.6347, 0.6624]`. Intervals for differences
from uniform were Accuracy `[+0.0782, +0.1227]`, NLL `[-0.0468, -0.0017]`,
and Brier `[-0.0320, -0.0043]`. These describe this public-dev cohort only.

Compared with the selected internal validation, dev Accuracy decreased from
0.5381 to 0.4340, NLL worsened from 0.9506 to 1.0739, Brier from 0.5661 to
0.6485, and ECE from 0.0154 to 0.0763. Because the cohorts differ and internal
validation selected the checkpoint, this gap is descriptive rather than a
causal estimate of selection optimism. It does show that the internal
validation score overstated performance on this source-held-out split.

## Reproduction artifacts

The pinned archive, saved predictions, extracted frozen features, and reports
remain outside Git under
`C:\CodexArtifacts\pretrained-reuse-r2-local\socialiqa-official-dev-audit-20261123-attempt-01\`.

| Artifact | SHA-256 |
|---|---|
| Official-dev audit report | `d090b04a4d3b8d551ffab6de10372499e30446f441a9e168c6d25d4bf444fa98` |
| Per-example predictions | `2807c65801841ff58dd4bcf734831309599146fc2349f29c86c31e9e76c83e95` |
| Frozen official-dev features | `5515203904597096497778fa7464450ed179b453a0d65269e9122b77d1d4f993` |
| Context bootstrap JSON | `ae503b11fa3f04d5b4fff66a12e66418fa6da2f6c30b86b1e1a6fb2136d2eab1` |
| Evaluation script | `1adbc46445102689dcf330ca0daec9c34e6a7b3f36d518c605e21159c1d3b96e` |
| Bootstrap script | `8e1b3d4e529bef3d72de262c293af2768b9c3daeadf8b631b902b2af51f6ef26` |

The result supports a limited claim: this compact BERT-family encoder can use
supplied options better than uniform choice for English social commonsense,
but the single internal validation score did not carry over unchanged to the
publisher dev cohort. This does not establish high-quality general text
Decision, multilingual capability, or any image/audio/video/sensor quality.
Keep the official dev labels closed to future tuning and use a newly admitted
task for the next candidate comparison.
