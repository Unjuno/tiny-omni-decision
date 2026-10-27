# Train-only matched audiovisual source baselines

The fixed [protocol](R2_SOURCE_UNIMODAL_CV_PROTOCOL_20261027.md) ran three
fresh readouts on the same cached Perception Test official-train features:
audio only, video only, and joint. Its three video-disjoint folds cover all
2,593 records from 320 video groups. The prior observed development labels,
official-validation outputs, official test and sealed audit were not read for
this run. This is internal cross-validation on a previously used source; it is
not a fresh blind benchmark or a product-quality evaluation.

Each readout has 344,834 parameters and was trained for seven fixed epochs,
seed 17, CPU AdamW, with identical per-fold permutations. The seven-epoch
budget was motivated by the historical readout's selected epoch, so this is
not a new prespecified independent test. Frozen VideoPrism and EfficientAT
features were reused; their 116,326,423 source parameters plus the readout do
not include a text encoder, a complete product stack, memory, or package cost.
The three folds use 213/213/214 training video groups and 107/107/106 held-out
video groups. No source model or original checkpoint was modified.

| Held-out predictions pooled over 3 folds | Accuracy | NLL | Brier | ECE-10 | Visible-source accuracy | None-visible accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Audio only | 0.2669 | 2.0982 | 0.6211 | 0.0263 | 0.0005 | 0.9986 |
| Video only | 0.5445 | 1.4945 | 0.6018 | 0.1120 | 0.5287 | 0.5882 |
| Joint | 0.5515 | 1.4441 | 0.5946 | 0.1240 | 0.5066 | 0.6749 |

Audio alone lacks candidate-specific visual inputs and scores visible candidates
symmetrically. Unlike the older zero-video intervention, its separate learned
none-visible score is allowed to differ. Its high none-visible accuracy and
near-zero visible-source accuracy show that this restricted model mostly chooses
none; they do not measure general audio understanding. The joint model improves
none-visible accuracy over the video-only fit but its visible-source point
accuracy is lower. Lower aggregate ECE for audio does not compensate for its
low accuracy.

Joint minus video Accuracy is +0.00694, with a paired 320-video percentile
bootstrap interval of −0.01087 to +0.02494 (10,000 draws, seed 17). Joint
minus audio is +0.28461 (interval +0.25399 to +0.31469). Intervals are
descriptive of these out-of-fold predictions: models share training examples
across folds, there is only one seed, video groups may share participants or
similar content, and subgroup/annotation validity is not independently
established. They do not establish that audio/video fusion beats video-only or
that this oracle-track source requires both inputs. This is not a 95% result.

Training elapsed 42.47 seconds, CPU only; CUDA was not initialized. A later
independent-process replay strictly reloaded all nine weights and reproduced
all 7,779 saved held-out predictions, with maximum probability difference 0,
matching labels, decisions, NLL and Brier. It reused the same pinned feature
loader and code, so it checks reproducibility, not independent correctness.
Its result at
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-unimodal-cv-replay-20261027-attempt-01/result.json`
has SHA-256 `b79a02bea9893ab838e09d5483c6a0cdccda23d2b9d5dbf17266e6ce1d5f402f`.
Replay used no GPU, training, or sealed audit. Targeted tests passed (5), Ruff passed, and the full CPU
suite passed 2,670 tests with 11 skipped. An initial shell-only checkpoint reload command
failed due to quoting; the corrected check loaded all nine without modifying
outputs.

External outputs:
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-unimodal-cv-20261027-attempt-01/`
contains the nine separate checkpoints, out-of-fold predictions, training
losses and run receipt, SHA-256
`cadda73e4f396a50b376a2e8f5e59c7321531560464e6ed3d6c7e082757cc8a5`.
The [summary](C:/CodexArtifacts/pretrained-reuse-r2/perception-source-unimodal-cv-summary-20261027-attempt-01/result.json)
SHA-256 is `402dc766ec066aafaac09c2b539efedb7a03f5b3d099eae80fe5390ae093cb0f`.
The run records all checkpoint and prediction hashes. Executed trainer SHA-256
is `3f018f47f48cb0062521edf3d970041799e83a2de2f5836f1273eee07c583592`;
summary script SHA-256 is
`d4512dd11e534420e905f15ef2b53ac33daa9024ab6b8eda7c144bfec77960fb`.
Continue toward a rights-cleared independent task with matched baselines and
validated source labels; do not promote this candidate on these scores.
