# iNaturalist CC0 VideoPrism image readout (2026-10-26)

## Status and scope

This is a development-only, closed-set, eight-species image-classification
probe. It reuses a frozen VideoPrism image representation and fits a single
linear classifier. It is not an open-vocabulary decision model, a temporal
video result, a multimodal result, or evidence for the project's 95–98% target.
There is no sealed audit set in this cohort, and no sealed audit was loaded.

The experiment followed the roadmap's pretrained-reuse Path A: establish a
small, rights-cleared natural-image readout with strict image-dependence
controls before making any product-quality or compression claim. The result
supports proceeding with a longer fixed-feature experiment, but it does not
justify promotion to a product Teacher.

## Frozen data and image features

The metadata cohort and its train/validation identities are described in the
[CC0 metadata freeze](R2_INATURALIST_CC0_METADATA_FREEZE_20261026.md). It has
1,642 train and 683 observer-disjoint validation photos across Rock Pigeon,
Great Blue Heron, Mallard, Canada Goose, Northern Cardinal, American Robin,
House Sparrow, and Great Egret. The metadata was selected from the live
iNaturalist API using explicit CC0 observation and photo licenses. One
cross-taxon photo identity and one photo without an explicit CC0 license were
excluded. This is a live-source freeze, not an immutable source snapshot.

All 2,325 frozen train/validation photos were downloaded to external storage.
The original media were checked for host, byte limit, and hash; EXIF orientation
was applied and metadata removed from normalized PNG copies. Cross-split exact
photo/media/RGB identity overlaps were zero; no cross-split perceptual-hash
match at Hamming distance 4 or below was found. A 32-image validation contact
sheet received a visual spot check; it found no clear mismatch, but it is not a
full manual audit.

External artifacts, not committed to Git:

- Metadata: `C:\CodexArtifacts\inat-birds-cc0-v1-20261026\`
- Media: `C:\CodexArtifacts\inat-birds-cc0-v1-20261026-media-attempt-01\`
- Features: `C:\CodexArtifacts\inat-birds-videoprism-features-v1-20261026-attempt-01\`
- 512-epoch readout: `C:\CodexArtifacts\inat-birds-videoprism-readout-v1-20261026-attempt-01\`
- 2,048-epoch readout: `C:\CodexArtifacts\inat-birds-videoprism-readout-v1-20261026-attempt-02\`

The feature run completed 1,642 train and 683 validation examples in 3,761.16
seconds on JAX CPU (`TFRT_CPU_0`). VideoPrism base revision
`be719a406d563b66f0ac969e7c94bab8e997c81a` used checkpoint SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29` and
114,365,184 frozen parameters. Preprocessing EXIF-oriented RGB images to a
288-pixel square produced one 768-dimensional feature vector per image by
averaging output tokens. The image was repeated over eight identical frames
for the encoder interface; this does not test temporal information.

| Feature artifact | SHA-256 |
|---|---|
| Frozen train IDs/metadata | `2c2f914b8575111836ea53cd2143b8621703a410d67c7c39a8ebc34781c66b2a` |
| Frozen validation IDs/metadata | `9a40230bdfb2312701b62471adb999cb557d17c631a208f95d4d36c924ae0487` |
| Train features | `77f689f30a7fa1178b90f56b638d57c63517596b6d1cf3963947ed4bf668ee76` |
| Validation features | `24c9a598ef762fd626cdbaba02988f4726171182757680f7f74b3901b35a3e86` |
| Feature-run config | `e7013de590648259a1ab01fae4414a13125232bd22febde15fab7f4ce1379aaa` |
| Train media manifest | `2014b39eccf26c22ba962131ffd2e710823dcfba5f4e83d140489b45cbedcedd` |
| Validation media manifest | `c15faf6827f23a795ae7fd8bc0e6789d368d64257d0e8c2b923b794972ee3fec` |
| Combined media summary | `a478860413eb8ee0dcfa2830b553a7a1d5c6fe24f8f565d25d592600e87e3a88` |

## Readout protocol and execution note

Both attempts used seed 17, the same frozen features and splits, a 6,152-
parameter linear 8-class readout, full-batch inverse-frequency weighted CE,
AdamW (`lr=0.001`, `weight_decay=0.0001`), and earliest minimum macro validation
NLL selection. Every epoch is one optimizer update over all 1,642 train
examples. Attempt 01 ran 512 epochs; attempt 02 was a fresh initialization with
the same seed/configuration and a 2,048-epoch budget. This is a matched-budget
comparison, not a resumed optimizer run.

In both attempts the training loop reached its configured epoch count and
saved best/final weights, full history, and validation predictions. The CLI
then exited with code 1 because the installed `TextFreeImageReadout.decide`
implementation reduced float32 softmax values in caller option order. The
measured probability differences were `1.1921e-7` (attempt 01) and `2.3842e-7`
(attempt 02), just above the respective `1e-7` and `2e-7` assertions; best
checkpoint reload logits were identical. The run's editable package import
resolved to another worktree (`codex/r0-r2-continuation-audit-20261021`, commit
`4a9f856ab010254fee6f83f438cc52ef77c9a311`, module SHA-256
`43acd10175dc7ffce734e2bb7054e8487cd4b33a73856f4ac404338fd2140045`), rather
than this worktree. The two module versions have identical readout forward
logic; the only difference was option-order softmax reduction.

Attempt 01 used the committed trainer at source commit
`f1fabbd5179c18bea032ecc3288c0479c19d29c8`. Attempt 02 used that same trainer
with only the tolerance changed in the working tree; its runner SHA-256 was
`2d617ace6442a03acaa7920127a2061c4ccbb6cecaf52cc8424c65ef43e59411`. The
trainer now explicitly places this repository's `src/` first on `sys.path` so
future execution cannot silently import the package from another editable
worktree.

The readout now normalizes supplied options in canonical taxonomy order and
maps probabilities back to the caller's order. This removes the reduction
order variation and is covered by an exact-invariance regression test. A
postflight verifier, without retraining, checked history continuity and
checkpoint selection, best reload, validation prediction identities and
probabilities, controls, and hashes. Both artifacts passed. The postflight
files explicitly retain the original nonzero process exit and the reason; they
do not rewrite either training attempt as a clean original CLI exit.

Training duration was not serialized because the original runner failed before
writing run metadata. It is therefore UNKNOWN; no estimated duration is
presented as a measured value. The training process selected CPU execution;
the installed PyTorch build was `2.6.0+cu124`, but no CUDA/GPU training was
performed. The frozen encoder feature extraction separately used Python
3.11.16, JAX 0.6.2, TensorFlow 2.18.1, and NumPy 1.26.4.

## Learning curves

Validation values below are from the same development split. The selector
repeatedly observed this split, so these are model-selection diagnostics, not
blind estimates.

| Epoch | Attempt 01 train acc | Attempt 01 validation macro acc | Attempt 01 macro NLL | Attempt 02 train acc | Attempt 02 validation macro acc | Attempt 02 macro NLL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.55% | 12.55% | 2.0890 | 12.55% | 12.55% | 2.0890 |
| 16 | 57.55% | 57.26% | 1.7351 | 57.55% | 57.26% | 1.7351 |
| 64 | 65.77% | 64.59% | 1.1997 | 65.77% | 64.59% | 1.1997 |
| 128 | 73.20% | 71.45% | 0.9580 | 73.20% | 71.45% | 0.9580 |
| 256 | 80.94% | 74.87% | 0.7693 | 80.94% | 74.87% | 0.7693 |
| 384 | 84.35% | 77.60% | 0.6830 | 84.35% | 77.60% | 0.6830 |
| 512 | 86.91% | 79.01% | 0.6337 | 86.91% | 79.01% | 0.6337 |
| 1,024 | — | — | — | 94.46% | 80.95% | 0.5634 |
| 1,276 | — | — | — | 96.22% | **81.13%** | **0.5584** |
| 1,536 | — | — | — | 97.50% | 80.71% | 0.5622 |
| 2,048 | — | — | — | 98.78% | 80.99% | 0.5845 |

The 512-epoch result was still improving at its final checkpoint. Extending the
same setup improved best validation macro accuracy by 2.11 percentage points
and reduced macro NLL by 0.0753. The longer run's best was epoch 1,276; from
there to epoch 2,048 train accuracy rose from 96.22% to 98.78%, while validation
macro NLL worsened from 0.5584 to 0.5845 and macro accuracy slipped from 81.13%
to 80.99%. This is a clear overfitting signal after the selected point. The
longer run did not reach the target.

## Selected 2,048-budget checkpoint

Best epoch 1,276, selected using minimum macro validation NLL. Counts and
metrics are on the fixed 683-example validation split.

| Metric | Overall | Macro over 8 species |
|---|---:|---:|
| Accuracy | 81.55% | 81.13% |
| NLL | 0.5408 | 0.5584 |
| Brier | 0.2634 | 0.2696 |
| ECE (15 bins) | 0.0453 | 0.0955 |
| Mean confidence | 0.7890 | 0.7858 |

| Species | n | Accuracy | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|
| American Robin | 74 | 74.32% | 0.7554 | 0.3917 | 0.1180 |
| House Sparrow | 116 | 73.28% | 0.6631 | 0.3579 | 0.0633 |
| Great Egret | 104 | 89.42% | 0.3037 | 0.1528 | 0.0661 |
| Rock Pigeon | 86 | 84.88% | 0.4141 | 0.2156 | 0.0645 |
| Great Blue Heron | 75 | 80.00% | 0.7180 | 0.2738 | 0.1488 |
| Mallard | 85 | 82.35% | 0.5435 | 0.2535 | 0.1461 |
| Canada Goose | 100 | 88.00% | 0.4086 | 0.1908 | 0.0586 |
| Northern Cardinal | 43 | 76.74% | 0.6605 | 0.3204 | 0.0983 |

### Input dependence controls

On the selected checkpoint, validation-image permutation fell to 10.54%
accuracy (macro NLL 6.0611), with only 10.54% top-1 agreement with aligned
predictions. Zeroing the image gave 14.64% accuracy and 14.64% top-1 agreement.
The aligned input reached 81.55% accuracy. This is evidence the readout uses
image features, but the controls do not show broad visual reasoning.

## Size and artifact integrity

- Trainable linear readout: 6,152 parameters.
- Frozen VideoPrism encoder: 114,365,184 parameters.
- Total unquantized model: 114,371,336 parameters, below the provisional
  200-million parameter ceiling.
- Encoder checkpoint: 457,475,296 bytes; best readout is about 26 KB. The
  unquantized package is therefore over 200 MB; no ternary quantization or
  package compression was performed here.
- Best checkpoint reload maximum logit delta: 0.0.
- Post-fix option-order probability delta on the first 32 validation cases:
  0.0. Saved prediction maximum probability delta: 0.0.
- attempt-02 best readout SHA-256:
  `323d0954a83c037ba5c6b37da3997addba48a66787a072acd3bae5fdbe7d2b88`.
- attempt-02 final readout SHA-256:
  `ed41cec67c3461ae2f14e2ee729a75ae845644a75db88fa3590818b62f83aacf`.
- attempt-02 training history SHA-256:
  `485437a39316acf66ee182a7d48f151b0c5b190d68888f1ed7d450a066bfda53`.
- attempt-02 validation predictions SHA-256:
  `46a82cbfb5e94af5c2f671e5691126c0e85a5e96f37a0d362acb34f8403eac36`.
- attempt-01 best readout SHA-256:
  `7dc85da5cebf91696422d5b22aca30ff2c1d7fd1ed2fcea34e832c7cd7d8635b`.
- attempt-01 training history SHA-256:
  `a2873278bdf7cba55614ff92cd19422ff8bb2b890673b5ac0ee979f043ba9759`.
- attempt-01 validation predictions SHA-256:
  `ac60f0df6d6014bc33c217b484a28d8c7575039ce678656ba60870b64899a353`.
- attempt-02 postflight report:
  `C:\CodexArtifacts\inat-birds-videoprism-readout-v1-20261026-attempt-02\postflight-verification-v2.json`.
- attempt-02 postflight report SHA-256:
  `8247cd9d07d30925f9c7c08ad0159ca5ab4068bf2bbd8e5f7d08e1130c087971`.

## Decision and next gate

The longer budget materially improved the selected checkpoint over 512 epochs,
but the curve then overfit before 2,048, and all eight classes remain below
90%; none approaches 95–98%. The controls establish image dependence, not
decision competence. The appropriate next work is to diagnose the data/task
ceiling and improve coverage or task formulation before considering a larger
readout. This result does not justify training a multimodal gate, ternary QAT,
or promoting this image-only classifier. Keep this artifact as a narrow R2
development reference and preserve the product's four-modality objective.

The postflight reports, histories, checkpoints, media, and feature arrays are
outside Git. Only the small source/tests/docs changes for this diagnostic and
the artifact hashes are versioned here.
