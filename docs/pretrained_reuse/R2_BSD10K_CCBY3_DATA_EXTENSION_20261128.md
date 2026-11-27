# BSD10k CC BY 3.0 train extension — 2026-11-28

## Question and scope

Does adding a substantial number of source-disjoint, individually CC BY 3.0
BSD10k training clips help the frozen EfficientAT audio readout, when the
validation records, model, optimizer, seed, and checkpoint selector stay fixed?

This is one audio-taxonomy experiment under the R2 frozen-feature Path A. It is
not a multimodal quality result and does not validate the product's 95–98%
aspiration. The BSD10k test split, DCASE evaluation data, and sealed audit were
not opened. Teacher v1/v2 and their model artifacts were not changed. All new
corpus and run files are under the separate local attempt directory named in
the reproduction section; media, private attribution, and readout weights are
not checked into Git.

## Source, rights, and frozen split

- BSD10k v1.2 Zenodo record: `19868804`.
- Source metadata ZIP SHA-256:
  `c090b97e5cdac221f867d5a970b0a8326833e57cfe586c5886494170b25413f1`.
- Existing seed-17 split SHA-256:
  `e1ae3d4088b792619a08a87023e4aeb2b1b5215091ba9772c0dc47fe88842c33`.
- The extension considers exact per-file CC BY 3.0 records only. The official
  [CC BY 3.0 deed](https://creativecommons.org/licenses/by/3.0/) permits sharing
  and adaptation, including commercially, subject to attribution, license-link,
  and applicable change-notice terms; it also notes that publicity, privacy, or
  moral rights may require separate permissions. This is a dataset experiment,
  not a blanket rights determination for every downstream product use.
- A private local attribution CSV records sound ID, supplied title and uploader,
  source URL, license URL, and the extraction/resampling change. The CSV is not
  committed because it contains raw uploader names and titles.
- Of 2,156 CC BY 3.0 metadata rows, 8 were excluded because their uploader group
  already occurs in frozen validation or test. The 2,148 remaining records were
  added to train only. They have 431 uploader groups, including 405 new groups;
  106 added records belong to groups already in train. All 23 labels and all
  annotation-confidence values were retained. No example was filtered for
  confidence or model outcome.
- Frozen validation and test rows were copied unchanged. New sound IDs are
  disjoint from the original manifest. The combined manifest SHA-256 is
  `2822c14203b2d5b11e335db550dd2f07d75742c62e351060c5b606ba3de9b650`.
- The 2,148 files total 1,735,933,124 bytes. Their ZIP CRC-32, per-file SHA-256,
  sizes, PCM decoding, and manifest metadata were checked. An integrated audit
  over the expanded train and frozen validation found zero sound-ID, uploader
  group, exact-file-hash, or normalized PCM-hash overlap.
- The normalized PCM fingerprint converts channel count and PCM width to
  mono signed-16 samples but retains the source sample rate in the fingerprint.
  Therefore a re-encoded duplicate at a different sample rate could escape the
  exact PCM match; source/uploader grouping and exact-file hashes remain the
  other overlap gates. No claim is made that this catches perceptual duplicates.
- Four archived files are valid PCM WAVs at 48 kHz / 24-bit / mono, although
  the [upstream BSD10k README](https://github.com/allholy/BSD10k) describes the
  archive's standardized files as 44.1 kHz / 16-bit / mono. They were not
  silently discarded: the pinned extraction decoder supports those PCM
  variants and resamples through the same fixed 32 kHz feature path. This source
  format discrepancy is retained as a data-quality limitation for follow-up.

## Frozen representation and training protocol

- EfficientAT DyMN-0.4 AudioSet checkpoint: 1,961,239 parameters,
  SHA-256 `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`.
- Encoder source revision: `a425fdce92572e602a1d5634799bd9f1f2efa806`.
- Original train features (4,214 × 384), original validation features
  (1,436 × 384), and all previous run outputs were left unchanged.
- Added feature extraction processed 18,696 two-second windows on CPU in
  235.80 seconds. Feature matrix is 2,148 × 384, SHA-256
  `7a5b60b38581144982e28585dee67b4d27cea4724846d4ab6fcbbe29047896a9`.
  The frozen model-state hash before and after extraction was identical.
- A new combined feature set contains 6,362 train examples / 1,590 uploader
  groups. The original 1,436 validation IDs, order, and feature values are
  preserved exactly in a new receipt bound to the combined manifest.
- Both readouts use the same seed 17, linear 384→23 readout, AdamW, learning
  rate 0.001, weight decay 0.0001, batch size 128, maximum 100 epochs,
  validation macro-NLL selector, patience 12, min-delta 0.0001, and the same
  class order. Scalar temperature is fitted only on a train-group calibration
  split. Calibration groups are frozen from the original 4,214-row training
  set; new uploader groups go to fit. Thus model, validation, and calibration
  roles remain aligned across the data comparison.
- The code supports an explicit extra-license allowlist. Historical behavior
  remains CC0/CC BY 4.0 only; CC BY 3.0 is accepted only when explicitly passed
  as `--allow-license CC-BY-3.0`.

## Matched validation comparisons

Both runs used the exact same 1,436 validation IDs in the same order and the
same targets. Metrics below are uncalibrated macro averages over 23 classes;
temperature-scaled values are also listed because the API returns calibrated
probabilities. The evaluation set was used for checkpoint selection, so this
is exposed development evidence.

| Run | Train rows / groups | Optimizer updates | Train-example exposures | Selected epoch | Accuracy | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen baseline | 4,214 / 1,185 | 2,200 | 278,300 | 100 | 0.4326 | 1.8051 | 0.7140 | 0.2775 |
| Extension, 100 epochs | 6,362 / 1,590 | 3,900 | 490,900 | 100 | 0.4800 | 1.5635 | 0.6527 | 0.2417 |
| Extension, matched 2,200 updates | 6,362 / 1,590 | 2,200 | 276,952 | 57 | 0.4304 | 1.7742 | 0.7051 | 0.2789 |

Temperature-scaled validation macro metrics:

| Run | Accuracy | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|
| Frozen baseline | 0.4326 | 1.7906 | 0.7155 | 0.2745 |
| Extension, 100 epochs | 0.4800 | 1.5623 | 0.6534 | 0.2450 |
| Extension, matched 2,200 updates | 0.4304 | 1.7571 | 0.7072 | 0.2847 |

The equal-epoch run uses 1.77× as many optimizer updates and 1.76× as many
sample exposures as the baseline. Its apparent +4.74 percentage-point macro
Accuracy and lower loss therefore do **not** isolate the effect of data.
Uploader-group paired bootstrap over 2,000 resamples gives macro Accuracy
difference [0.0046, 0.0748] and macro NLL difference [−0.2686, −0.1534], but
the wider compute budget is a confound.

The matched-update comparison better isolates dataset coverage: update counts
are equal, and the expanded run presents 276,952 examples versus 278,300 for
baseline (−0.5%), while exposing the readout to 2,148 more distinct examples
and 405 additional uploader groups. Macro Accuracy changes by −0.23 points;
NLL improves by 0.0309 and Brier by 0.0090, while ECE worsens by 0.0014.
Paired uploader-group bootstrap (254 validation groups, 2,000 replicates,
seed 17) gives candidate-minus-baseline intervals:

- macro Accuracy: [−0.0318, 0.0204]
- macro NLL: [−0.0561, 0.0403]

Both intervals include zero. This single-seed result does not establish a
same-compute Accuracy gain; its calibration/loss movement is mixed and
uncertain. The validation-selected checkpoint was the final epoch in all
three runs, so none showed validation-based early stopping within its budget.

## Interpretation and next step

The larger run shows that added data plus more optimization budget can improve
this task. The update-matched run does not show a reliable Accuracy improvement
from added coverage alone, and does not establish that more data is useless.
Likely contributors remain class difficulty, annotation ambiguity, uploader
distribution, and the low-capacity fixed linear readout. The exposed validation
and single seed limit certainty. No conclusion generalizes to text, image,
video, sensors, or cross-modal fusion.

Keep the original run as baseline. Do not promote this readout or tune against
the opened validation. If more data work continues, first adjudicate the
48 kHz/24-bit archive discrepancy and inspect classes where extra examples
changed support without improvement. Keep any new source disjoint by asset and
uploader, and run the next comparison with a prespecified matched update and
sample budget. The BSD10k test and sealed audit stay unopened.

## Reproduction and local artifacts

Source code was based on commit `2b43dcbdb647c33da8a589d7ac8ac7a68ec1df7e`;
the implementation and report are in the current R2 source-audit branch.
Focused tests cover explicit license opt-in, frozen held-out assignments,
feature-receipt combination, unchanged validation features, valid PCM format
variants, and fixed-update training. The fixed dependency environment was
Python 3.11.9 / PyTorch 2.6.0+cu124 / torchaudio 2.6.0+cu124 on CPU. No GPU or
cloud compute was used.

Local attempt root (outside Git):

```text
C:\CodexArtifacts\pretrained-reuse-r2-local\bsd10k-v1.2-source-audit-20261128-attempt-01\ccby3-extension-attempt-01
```

It holds the private attribution CSV, new manifests, download receipts and
media audit, frozen feature receipts, both expanded readouts, and comparison
reports. The original corpus and readout directories were read-only inputs.

Key local artifact hashes:

| Artifact | SHA-256 |
|---|---|
| Addition download receipt | `f9ecd4bf31f1d72a2374132c78a45d3323a1f74a1ae5cbdbe6d3332e48b6446a` |
| Addition media audit | `b4b268bc4a1b801019db9e4da5c1dad951ad5ed719086d0a985484a765d9df7b` |
| Combined train-feature receipt | `c9d259d21e4e2a64702867d3790852f6a50454006163ebbecb51e457f66e46c7` |
| Matched-update run config | `42085e468336890d4f93319576c3d4a482f42a6f8d3ce5a868230b9ea2275b8b` |
| Matched-update readout weights | `122305cd9591eff1c77fc47f1a6fe460ab0d139c71d5f88c8a8b6c8efee61fe3` |
| Matched-update comparison | `522fbe9549da8163a4696721e8df78821f3851ed6a3a4819cd814bfc8ceb2ccc` |
| Equal-epoch comparison | `b25e3f5555a90f1c48e13cde7a4473608f607c5a2d2de043d848868096711020` |
