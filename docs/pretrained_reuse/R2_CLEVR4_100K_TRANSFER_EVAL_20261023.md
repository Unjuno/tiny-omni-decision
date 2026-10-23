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

The first attempt completed archive, annotation, media-hash, vocabulary, and
runtime preflight, then was stopped before model inference; its predictions
file is empty. Its preflight artifacts are retained at
`C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-videoprism-transfer-eval-attempt-01`.
The second attempt reuses the previously verified full-archive SHA-512 record,
rechecks all selected validation media against the old media hashes, and runs
the full validation once. Neither attempt modifies the archive, readout,
encoder, or prior training outputs.

## Results (pending)

The complete evaluation has not finished. Accuracy, NLL, multiclass Brier,
15-bin equal-width ECE, mean confidence, runtime, and prediction hashes will be
recorded here after the single full replay completes. No partial model-quality
metrics will be used for decisions.

This official validation is not a sealed or blind audit. After observing these
results, it must not be reused as a fresh holdout for another tuned candidate.
