# VideoMAE Clevr-4 count-only linear readout diagnostic — 2026-11-10

## Question and interpretation

The frozen VideoMAE Clevr-4 Path A result had only 42.60% selector Accuracy on
counting, much lower than color and shape. This one-run diagnostic asks whether a
plain affine readout can recover count from the already frozen 384-wide global
image feature, compared with the existing shared taxonomy-conditioned MLP. It
is a readout-capacity diagnostic on the same development selector, not a new
benchmark and not a model-selection sweep.

The linear readout scored 32.00% Accuracy versus 42.60% for the existing MLP on
the same 1,000 selector images and feature cache. Its NLL, Brier, and ECE were
also worse. This argues against the claim that a simple linear boundary is
sufficient for count in this representation. It does **not** show that count
information is absent from the frozen feature: the nonlinear MLP performs 10.6
percentage points better. Nor does it establish that the MLP or global feature
is the sole bottleneck.

| Readout (fixed epoch 8) | Parameters | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|
| Existing shared taxonomy MLP | 52,682 | 42.60% | 1.3405 | 0.6648 | 0.0354 |
| Count-only linear | 3,850 | 32.00% | 1.6720 | 0.7607 | 0.0534 |
| Linear minus MLP | — | -10.60 pp | +0.3315 | +0.0959 | +0.0180 |

The MLP was the previously selected epoch-8 checkpoint under equal-taxonomy
macro NLL. The linear model ran one fixed eight-epoch schedule and kept the
final epoch; selector metrics were recorded each epoch but were not used to
choose a checkpoint or change the recipe. Its selector NLL declined throughout
(1.9997 to 1.6720), while Accuracy rose from 22.90% to a maximum of 32.50% at
epoch 6 and ended at 32.00%. This short curve is diagnostic only.

## Fixed data and method

- Dataset: Clevr-4 100k v1, source license CC-BY-4.0, official TRAIN split only.
- Frozen Path A experiment: `path-a-image-videomae-clevr4-100k-seed17`.
- Train / selector: 72,010 / 1,000 images. The split manifests have unique
  media identities and zero train-selector overlap; the code rechecks IDs,
  manifest order, media hashes, cached feature order, and finite features.
- Unique training images/assets: 72,010; eight passes, 576,080 presentations.
  The selector consists of 1,000 images and 1,000 count decisions.
- The original 5,296-image one-time reporting labels were not read, and no
  reporting feature rows were materialized for scoring. The frozen compressed
  feature archive contains the full frozen split matrix; only train and
  selector feature rows were gathered into tensors. No sealed audit was loaded.
- Frozen encoder: OpenGVLab VideoMAE V2 ViT-Small K710, 22,153,286 parameters,
  16 identical copies of each static image, 224-pixel input, 384-wide feature.
  No encoder update or new feature extraction occurred.
- New readout: `Linear(384, 10)` (3,850 parameters), raw frozen features,
  AdamW, learning rate `0.001`, weight decay `0.0001`, constant schedule,
  batch 512, seed 17, eight epochs. No selector checkpoint selection.
- Total resident pre-quantization parameters for this task-scoped candidate:
  22,157,136. This is not a whole Omni model or quantized package.

## Curves and runtime

| Epoch | Rolling train CE | Rolling train Accuracy | Selector Accuracy | Selector NLL |
|---:|---:|---:|---:|---:|
| 1 | 2.0653 | 24.34% | 22.90% | 1.9997 |
| 2 | 1.8534 | 30.26% | 27.20% | 1.8826 |
| 3 | 1.7618 | 32.95% | 28.10% | 1.8200 |
| 4 | 1.7035 | 34.67% | 30.10% | 1.7705 |
| 5 | 1.6612 | 36.03% | 31.50% | 1.7355 |
| 6 | 1.6287 | 36.83% | 32.50% | 1.7113 |
| 7 | 1.6010 | 37.85% | 32.20% | 1.6898 |
| 8 | 1.5790 | 38.27% | 32.00% | 1.6720 |

Rolling train metrics are not a fixed-example generalization gap. Local compute
used the RTX 3080 Laptop GPU, PyTorch `2.6.0+cu124`, Python `3.11.9`; fit plus
eight selector evaluations took 2.745 seconds and peak allocated VRAM was
70,353,408 bytes. No paid or cloud compute was used. The final state reloaded
and produced bitwise-identical selector logits.

## Reproduction and integrity

- Parent source commit at execution: `7d65f11e18da2b2ab5fbd24eb77340308082e00e`.
- Executed script SHA-256:
  `a9a2267c1222da30e8104d739899fc9e491995efe164abb3cfae38e8474c5a97`.
- Frozen input receipt SHA-256:
  `350e7d4dec9f113d941f1415dc3bda4e556315ea57baffb89fe2efa62717a16b`.
- Frozen experiment config SHA-256:
  `97f32fd4a2172526c67265f7cd74a153c993fd2e05992ce0b4899de8a04f160c`.
- Feature cache SHA-256:
  `7b6cef6b6451eb7cafd8d918e34fe527fac20fb658ae79c692cac28c9130e4b1`.
- Feature manifest SHA-256:
  `019cb10f6943c1e6b58ff671215e657d310a65cd53162538c4e3c3789d0694f6`.
- Readout SHA-256:
  `d2e020531183d2a3ae4647e94fdf9734393fc1fa3e39f2b0589340d2a0be5633`.
- Effective config SHA-256:
  `f17b0f4ada71a927bf2f06cd4c12874e363e558e66d56d2e33ddecd8ce923d59`.
- History SHA-256:
  `9c292ace92d59da505c7aafef2435ffe30125d01dc5be0dbb7f148ebe28d608a`.
- Selector predictions SHA-256:
  `5279378222e8e947a5f1497b20af01a7e8da53c994db832ecbd4fbc711587838`.
- Result JSON SHA-256:
  `fdd9016d2437f909d2da128945209a4dcbfe8b9117d1026c7beb5bead7b6b6ff`.
- External artifacts: `C:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae-count-linear-diagnostic-20261110-attempt-01\`.

## Next bounded implication

Keep the existing MLP as the stronger count baseline. The linear result is not a
reason to enlarge the model or to reopen the one-time reporting pool. A spatial
or object-centric representation could be tested only as a new frozen-feature
candidate with its own predeclared recipe and the same selector used strictly
for development selection; it must be compared against this preserved MLP and
must not touch the one-time reporting labels. The current result remains
narrow synthetic image evidence and does not close the broader R2 multimodal,
quality, or deployment gates.
