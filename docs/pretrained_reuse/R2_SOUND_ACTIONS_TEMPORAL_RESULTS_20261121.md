# SoundActions frozen temporal readout — 2026-11-21

## Result

This is a narrow, single-seed development-validation pilot for the admitted
SoundActions v2 task. It compares frozen EfficientAT audio features, frozen
VideoPrism video features, and a small learned gated-BiLSTM readout. It does not
measure general-purpose multimodal decision quality, new-person generalization,
or the product's full four-modality target.

On the same 48 recording-ID validation examples, audio-only was strongest by
validation NLL and Accuracy. The gated audio/video model did not improve on
audio-only. The learned gate and zero/permuted-input controls do not support a
useful-video-fusion claim for this task.

| Condition | Best epoch / stopped epoch | Trainable parameters | Accuracy | Macro Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Audio-only | 6 / 106 | 107,331 | 64.58% | 65.11% | 0.809 | 0.468 | 0.093 |
| Video-only | 3 / 103 | 205,635 | 43.75% | 44.94% | 1.123 | 0.678 | 0.164 |
| Gated A/V | 6 / 106 | 325,187 | 52.08% | 53.12% | 0.914 | 0.552 | 0.209 |

Each validation condition has 48 recordings: 13 impulsive, 16 sustained, and
19 iterative. Per-class Accuracy:

| Condition | Impulsive (n=13) | Sustained (n=16) | Iterative (n=19) |
|---|---:|---:|---:|
| Audio-only | 76.92% | 50.00% | 68.42% |
| Video-only | 76.92% | 0.00% | 57.89% |
| Gated A/V | 69.23% | 37.50% | 52.63% |

On gated A/V's selected checkpoint, zeroing video gave 64.58% Accuracy and
0.819 NLL, while permuting validation video identities gave 60.42% Accuracy
and 0.867 NLL. The unmodified gated A/V checkpoint gave 52.08% Accuracy and
0.914 NLL. Zeroing audio gave 43.75% Accuracy and 1.225 NLL. Thus, on this
validation set, removing or mismatching video did not hurt the A/V readout.
The gate's mean was 0.531 (standard deviation 0.135); a near-half gate value
does not establish useful information exchange.

Paired bootstrap used 10,000 resamples stratified by target class, seed 17,
over the same 48 selected validation recordings. Positive differences mean
the second condition is better. Gated A/V minus audio-only Accuracy was
−12.38 percentage points (95% percentile interval −22.92 to −2.08 pp), and
macro Accuracy was −11.89 pp (−23.04 to −1.43 pp). The NLL difference was
−0.105 in the “candidate improves” convention (interval −0.264 to +0.049),
which includes zero. Gated A/V versus video-only had a +8.31 pp Accuracy
difference (−6.25 to +22.92 pp), also uncertain. Because validation selected
the checkpoints, these are selection-affected development intervals, not
blind estimates.

## Learning and overfitting

The fixed 246-example training set was re-evaluated each epoch. All three
conditions overfit strongly. The selected-to-final fixed-train and validation
NLLs were:

| Condition | Selected train / validation NLL | Final train / validation NLL | Overfit signal |
|---|---:|---:|---|
| Audio-only | 0.437 / 0.809 | 0.00009 / 2.751 | Yes |
| Video-only | 0.844 / 1.123 | 0.00002 / 4.133 | Yes |
| Gated A/V | 0.305 / 0.914 | 0.00003 / 2.482 | Yes |

The best checkpoint was selected by lowest validation NLL; exact ties keep the
earlier epoch. Early stopping patience was 100 epochs, with 500 as the maximum.
Each condition saved both its selected checkpoint and its final checkpoint,
then reloaded the selected checkpoint and checked validation predictions.
Rolling training loss is not used as a generalization estimate.

| Condition | Updates | Best checkpoint SHA-256 (bytes) | Final checkpoint SHA-256 (bytes) |
|---|---:|---|---|
| Audio-only | 1,696 | `58f27a4a00d1df1e17bb11b513c641bac33da8cb8d5b3c229d210345670016a7` (433,282) | `cae0fde973c97f0b5183b8cb30fda400cb69ecc91f6b0b95d1183ac1a6a01b75` (432,442) |
| Video-only | 1,648 | `c4f294daab04ecce27b722debef221cc8809394c7e9614327efc75a35038747b` (826,498) | `c85f4850ef78113687dc08cc9fc233842a6e4200fa14606386260ba12dc6886f` (825,658) |
| Gated A/V | 1,696 | `a81e01367a19ec1de20c7b20e0290ce045f9642dfee775ca2243d1f9eb5212ed` (1,307,630) | `a783c23d3dd57629149ae75bbe1305d89e6285d3f89611c22ae8a7b5acc3b31f` (1,306,322) |

## Frozen data and encoders

- Source: Zenodo record 23971067, SoundActions v2, CC BY 4.0.
- Frozen recording-ID split: 266 train / 53 validation / 46 sealed audit.
  IDs 1–35 were assigned to train because their overview labels had appeared in
  the source preview before the split was frozen.
- Majority-label filtering left 246 train and 48 validation examples; 20
  train and 5 validation `unknown` labels were excluded. No sealed-audit row
  or media was opened.
- All 588 train/validation audio and video files matched the publisher's
  individual SHA-256 inventory before use. The 46 sealed recording IDs were
  not extracted or evaluated. One recordist and published prior cross-validation
  results limit the interpretation: this is a project-held-out recording
  split, not a blind benchmark or person-independent evaluation.
- EfficientAT DyMN-0.4: 1,961,239 frozen parameters; source commit
  `a425fdce92572e602a1d5634799bd9f1f2efa806`; checkpoint SHA-256
  `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`.
  Audio was represented as 2-second windows with 1-second stride, producing
  2,622 feature windows over 294 recordings.
- VideoPrism-B: 114,365,184 frozen parameters;
  `google/videoprism-base-f16r288@be719a406d563b66f0ac969e7c94bab8e997c81a`;
  checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
  Eight uniformly sampled 288×288 frames were encoded on CPU and the 256
  spatial tokens were averaged independently per frame.
- The paired frozen encoders total 116,326,423 parameters; the readout adds at
  most 325,187 parameters, for 116,651,610 total with the gated readout. The
  paired encoder checkpoints total about 465.6 MB (the VideoPrism checkpoint
  alone is about 457.5 MB), so this research setup does not meet the roadmap's
  200 MB package target. No quantization or product artifact was made.

## Integrity and implementation notes

The first ranged video-archive assembly failed the Zenodo whole-archive MD5.
Train/validation ZIP CRC checks localized six bad resumed ranges (24, 25, 27,
31, 34, 35). Those ranges were independently fetched from Zenodo's official
content endpoint; the original range files were retained separately. The
reassembled archive then matched the official 1,877,548,445-byte size, MD5
`300fd4c64d367482bbbff59ab6b1d1ca`, and SHA-256
`bc49282ca4e127e547069a1340c18e600b3edcd80ae93a0957733a9baf26533d`.
The audio archive matched its official MD5 and has SHA-256
`a682a9587be1bca8f06fca14bf09b84e42f3c0f2ccfcf3e550ddd2c4b44a321b`.

Media extraction once hit Windows `OSError: [Errno 22]` while writing one
video. Extraction was changed to stream through a temporary file, verify the
publisher SHA-256, flush, then atomically publish. Existing partial media is
reused only after its size and publisher hash match. Resumed archive ranges now
require a saved partial SHA-256, accept only complete verified HTTP ranges, and
merge atomically; unverified partials are preserved under an `.untrusted`
suffix. The publisher-hash parser now ignores still-image entries when
validating the required audio/video set.

## Reproduction and artifacts

Source repository commit before this change: `9d704ce795a5be8a5d42abbc4f735e67550dbdbf`.
No existing Teacher v1/v2 checkpoint or sealed audit was changed or read.

```powershell
python -m scripts.train_soundactions_temporal_readout `
  --audio-features C:\CodexArtifacts\pretrained-reuse-r2\soundactions-v2-20261121-attempt-02\audio-features-01 `
  --video-features C:\CodexArtifacts\pretrained-reuse-r2\soundactions-v2-20261121-attempt-02\video-features-01 `
  --output C:\CodexArtifacts\pretrained-reuse-r2\soundactions-v2-20261121-attempt-02\temporal-readout-01
```

- Split manifest SHA-256: `0404814579c67b766e068a43810d1d724b19afd46394a467584d9da98b1ed9dc`.
- Train/validation media index SHA-256: `e3bae83f95035579afa3b7aa86eb2778a5081080e6913ecbcec8a7145c5ffdb4`.
- Audio feature cache SHA-256: `ecb678bbfd97c3ad17becfbdd758a926b981a392815b050cbc7b247ba80fede5`.
- Video feature cache SHA-256: `012b1b0035eca23607292db899b6ccb18a7f06b55ff3e78b8c670419beff15ce`.
- Readout output contains `experiment-summary.json`, `comparisons.json`,
  per-condition `training-history.jsonl`, best/final checkpoints, and
  48-record selected-validation prediction files. Checkpoint hashes and sizes
  are in each condition's `result.json`.
- Local compute only: RTX 3080 Laptop for audio feature extraction and readout
  training; CPU for VideoPrism. Audio feature extraction used a peak 29.5 MB of
  PyTorch CUDA memory. Readout training peaked at 49.5 MB. VideoPrism averaged
  1.87 seconds per forward pass and took about 21.8 minutes wall time. Cloud
  compute and cloud cost: none. Audio feature extraction took 42.6 seconds;
  train/validation media verification and preparation took 193.5 seconds.
  Readout model fitting used 57.9 seconds in training and 30.1 seconds in
  evaluation across all three conditions. PyTorch reported 49.5 MB peak
  allocation for the readout process.

## Decision

This pilot does not support promoting the gated A/V readout. Audio-only is the
stronger validation result, and video identity controls do not show useful
video dependence. The clear overfit and small, single-recordist dataset also
mean that no claim about model-family capability or future fusion is justified.
Keep the sealed partition unused; next work should improve task diversity and
label coverage or choose a larger rights-cleared task where both modalities
are necessary. Text, image, sensor, 4-modality fusion, 95–98% quality, and the
final compact package remain unvalidated.
