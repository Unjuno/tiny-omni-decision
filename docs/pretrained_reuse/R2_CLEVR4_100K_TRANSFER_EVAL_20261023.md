# CLEVR-4 100k VideoPrism Fixed-Readout Transfer Evaluation

**Status:** full official-validation replay is running; no prediction metrics have
been inspected or used for selection. This is an image-only transfer check, not
a sealed audit or a general multimodal quality claim.

## Question and frozen candidate

Measure whether the existing fixed VideoPrism-B + candidate-MLP readout,
selected on the earlier CLEVR-4 release, transfers to the complete official
validation split of the 100k release. The checkpoint, option embeddings,
preprocessing, candidate vocabularies, and model parameters are unchanged.
There is no training, calibration, hyperparameter search, checkpoint selection,
or test-set evaluation in this experiment.

The candidate is scoped to four synthetic CLEVR-4 image taxonomies: color,
texture, count, and shape, with ten supplied options per taxonomy. Each image
is repeated over the frozen runtime's eight VideoPrism frame slots. This does
not test temporal video reasoning, arbitrary open-vocabulary choices, audio,
text, sensor input, or joint multimodal fusion.

## Data and split integrity

The source is the official CLEVR-4 100k archive, code revision
`sgvaze/clevr4@cddc78fb2a8359dc958987b2c750bfdd4bfd2c73`, under the recorded
CC-BY-4.0 license. The archive SHA-512 matches the official checksum recorded
in the immutable source preflight:

`7e8454f9f44d7b527eb20252a687b15b4cd28f3e9c79cf6147317971034c367b64bf6bba98d2a66f64b26f1e7eea2105a8acf221198342e89a844fd0007cc94c`

The archive has 80,012 train and 19,988 validation images, and no official test
partition. The annotation payload SHA-256 is
`1e51a1ea77da85c7c9203b422dcd3c337f053981f4bd6848be7c41f2784556cc`.
All four expected ten-class vocabularies are present. The validation set
therefore contains 79,952 image-taxonomy decisions.

The earlier fixed-readout corpus has 8,424 train and 2,107 validation images
(10,531 unique IDs and 10,531 unique media hashes). Its IDs overlap the new
archive's train split in 5,440 cases (4,339 earlier train and 1,101 earlier
validation), but overlap the new official validation in zero cases. The complete
new validation also has zero SHA-256 media overlap with either earlier split.
All 19,988 validation IDs and image SHA-256 values are unique within that
split. This is a clean held-out transfer evaluation at the image level, while
sharing the CLEVR-4 source family and synthetic task design.

The frozen, sorted validation ID order hash is
`4287f25d9bcdcf1333fe48159eeaf1c7fc212e17ca8e05e475c7c124bfbf4525`; the
per-image validation manifest hash is
`6afb7ca40309d7a107f277bc567f8c71c855ecdff2c30eb37e24cb2e45435a9a`.

## Candidate and execution

- Bundle: `path-a-image-videoprism-clevr4-runtime-v0`
- Bundle-manifest SHA-256: `958e1f8735c33237342d0533f7529a0514fc4f002ca812057b2eff28ac51012e`
- VideoPrism-B revision: `be719a406d563b66f0ac969e7c94bab8e997c81a`
- Encoder checkpoint SHA-256: `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`
- Frozen readout export SHA-256: `1b5311669c4945c7d5bd49fd16d7ee1994dd0e5b52704c134a7690ec77004335`
- Readout source SHA-256: `d8499747b8b94de20972c2c9ddf6c15911727a62e1aaeea393d76f6359bb40b1`
- Resident model parameters: 114,512,897; the prior bundle's 200 MB package
  target is not met (458,143,789 payload bytes).
- Runtime: Python 3.11.16, JAX 0.6.2, TensorFlow 2.18.1, NumPy 1.26.4;
  Windows CPU backend. GPU evaluation and cloud compute are disabled.
- Source runner: `scripts/evaluate_clevr4_100k_videoprism_transfer.py`.
- Output: `C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-videoprism-transfer-eval-attempt-02`.
- Run metadata and per-image predictions are written incrementally. The
  effective config and exact data/model hashes are saved before inference.
- Source base: `0bbb5d17cc9a2b7e5e0d9353e3e70d0c6c1ff17a`. The run started
  before its runner commit was created; the executed runner's SHA-256 matches
  the file blob later committed as `0794b5407006cc19ec5a386f9d689096453dc3c7`
  (and remains identical at `7dc8fe07bdb351604e345c74324f821895346b49`). The
  exact command was:

  ```powershell
  C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe -u scripts/evaluate_clevr4_100k_videoprism_transfer.py --archive C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-videoprism-v0\clevr_4_100k_v1.full.zip --source-preflight C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-videoprism-v0\preflight.json --old-manifest C:\CodexArtifacts\pretrained-reuse-r2\clevr4-full-coverage-seed17\train-image-manifest.jsonl --old-manifest C:\CodexArtifacts\pretrained-reuse-r2\clevr4-full-coverage-seed17\val-image-manifest.jsonl --bundle C:\CodexArtifacts\pretrained-reuse-r2\path-a-image-videoprism-clevr4-runtime-bundle-v2 --source-root C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source --output-dir C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-videoprism-transfer-eval-attempt-02
  ```

The first attempt completed archive, annotation, media-hash, vocabulary, and
runtime preflight, then was stopped before model inference; its predictions
file is empty. Its preflight artifacts are retained at
`C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-videoprism-transfer-eval-attempt-01`.
The second attempt reuses the previously verified full-archive SHA-512 record,
rechecks all selected validation media against the old media hashes, and runs
the full validation once. Neither attempt modifies the archive, readout,
encoder, or prior training outputs.

## Results

The one-shot CPU replay completed all 19,988 official validation images and
79,952 supplied-choice decisions from 2026-10-23 15:24:38 to 21:40:27 JST in
22,549 seconds (6 h 15 m 49 s). The prediction file contains exactly 19,988 unique IDs in the frozen manifest
order; all four taxonomies contain 19,988 rows each. Every row has ten options,
finite nonnegative probabilities summing to one within `1e-5`, and a prediction
matching the maximum-probability option. No ID, shape, label, or probability
validation errors were found. The prediction SHA-256 in run metadata matches
the complete file hash:

`fdd0ff5a7ff524af68ba8c92e800999956935d617a6314f6896c70b7e96826b0`

The run-metadata file SHA-256 is
`b805544d252b371e835b059ac4711e41542c6e8ee812347e2f31b6a271046bcb`, and the
effective-config file SHA-256 is
`cf3f322d637983fe563abb7752f00400aa8cf0a5c408124ca02f172a67dcf0cb`. Its
canonical effective-config hash recorded by the runner is
`4308ab93929af9874a3c6926cd5c80adbb501b2909ac0f922a171234a025998d`.

| CLEVR-4 taxonomy | N | Accuracy | NLL | Brier | ECE (15 equal-width bins) | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Color | 19,988 | 0.99485 | 0.02994 | 0.01034 | 0.01864 | 0.97624 |
| Texture | 19,988 | 0.90279 | 0.27437 | 0.13692 | 0.01092 | 0.89877 |
| Count | 19,988 | 0.60436 | 0.89736 | 0.49178 | 0.02179 | 0.60444 |
| Shape | 19,988 | 0.99450 | 0.02000 | 0.00926 | 0.00572 | 0.98888 |
| Macro across taxonomies | 79,952 decisions | 0.87412 | 0.30542 | 0.16207 | 0.01427 | 0.86708 |

The result is highly uneven: color and shape transfer strongly, texture is
moderate, and object count remains difficult. The model does not clear 90%
macro Accuracy, and this result does not establish generalized image reasoning.
It measures four fixed ten-way CLEVR-4 attribute readouts on repeated static
images only. It says nothing about arbitrary option wording, temporal video,
audio, text, sensors, or modality fusion. No quality metric was examined before
the full run completed, and the result did not select or tune a model.

For reproduction, the effective config SHA-256 is
`4308ab93929af9874a3c6926cd5c80adbb501b2909ac0f922a171234a025998d`; the
runner SHA-256 is
`2d0c4be6047baafe61dd743f1a85e4af065132b9252d972dbd8ae61b64e99008`. The
validation manifest SHA-256 is
`6afb7ca40309d7a107f277bc567f8c71c855ecdff2c30eb37e24cb2e45435a9a`. Run
metadata is at
`C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-videoprism-transfer-eval-attempt-02\run-metadata.json`;
the full per-image predictions are in the same output directory. All 19,988
validation images had unique media hashes, and their media hashes and IDs had
zero overlap with the prior train/validation manifests.

The runtime was Python 3.11.16, JAX 0.6.2, TensorFlow 2.18.1, and NumPy 1.26.4
on Windows CPU. No GPU, cloud compute, model training, calibration, or
checkpoint selection was used. CPU elapsed time was approximately 6.26
single-core hours; compute cost was $0. This was the official validation split,
not a blind or sealed audit, and it must not be reused to tune a candidate.
Prior release-holdout results use a different split and are not a matched
comparison; no claim of improvement over that result is made here.
