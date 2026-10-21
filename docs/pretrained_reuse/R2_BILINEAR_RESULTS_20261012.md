# Capacity-matched candidate interaction results

Execution PASS. Concat reproduces all eight prior epoch records and the
saved best tensor-file SHA exactly. Frozen CLS+patch-mean768 and question/
candidate384 features; only the interaction readout differs. No feature
normalization, new data, encoder tuning, or temperature selection.
Concat147,713 parameters; bilinear147,712. Same seed17,8epochs,AdamW1e-3
wd1e-4,batch64, question order and validation-NLL selection. Initialization
parameterization differs; this is not an isolated language-removal experiment.

8,424 train images/33,696 unique questions;269,568 encounters per run.
512 reused selection images/2,048 questions. No reporting1595 images
or sealed audit. Synthetic taxonomies do not measure generic image reasoning.

| Policy | Taxonomy | Accuracy | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| concat | all | 0.850586 | 0.377859 | 0.193470 | 0.015542 |
| concat | color | 0.964844 | 0.109854 | 0.054958 | 0.013810 |
| concat | texture | 0.906250 | 0.290167 | 0.140665 | 0.025670 |
| concat | count | 0.537109 | 1.083845 | 0.566052 | 0.053907 |
| concat | shape | 0.994141 | 0.027569 | 0.012206 | 0.005465 |
| bilinear | all | 0.874023 | 0.289319 | 0.157210 | 0.011022 |
| bilinear | color | 0.978516 | 0.059637 | 0.029487 | 0.016945 |
| bilinear | texture | 0.921875 | 0.201637 | 0.109964 | 0.031818 |
| bilinear | count | 0.597656 | 0.889974 | 0.486729 | 0.042458 |
| bilinear | shape | 0.998047 | 0.006029 | 0.002658 | 0.003344 |

Bilinear-minus-concat +2.344pp; paired95% interval[+1.025,+3.662]pp.
512-image paired bootstrap,10,000 draws,seed17,percentile interval;
validation-selection bias remains. All probabilities retained locally.

## Learning curves

| Policy | Epoch | Train Acc | Validation Acc | Train NLL | Validation NLL |
|---|---:|---:|---:|---:|---:|
| concat | 1 | 0.712429 | 0.709961 | 0.844325 | 0.846183 |
| concat | 2 | 0.766055 | 0.769531 | 0.627966 | 0.640931 |
| concat | 3 | 0.819949 | 0.814453 | 0.485783 | 0.513627 |
| concat | 4 | 0.829950 | 0.821289 | 0.439285 | 0.457541 |
| concat | 5 | 0.840426 | 0.830078 | 0.412118 | 0.428126 |
| concat | 6 | 0.851080 | 0.829102 | 0.380870 | 0.415854 |
| concat | 7 | 0.870756 | 0.847168 | 0.334178 | 0.378065 |
| concat | 8 | 0.867254 | 0.850586 | 0.332495 | 0.377859 |
| bilinear | 1 | 0.814043 | 0.805664 | 0.509039 | 0.518170 |
| bilinear | 2 | 0.842177 | 0.837402 | 0.409898 | 0.429787 |
| bilinear | 3 | 0.859123 | 0.855469 | 0.369991 | 0.381043 |
| bilinear | 4 | 0.872952 | 0.869629 | 0.316525 | 0.343843 |
| bilinear | 5 | 0.877552 | 0.870117 | 0.306238 | 0.324418 |
| bilinear | 6 | 0.883755 | 0.873535 | 0.283585 | 0.309805 |
| bilinear | 7 | 0.880134 | 0.859375 | 0.298052 | 0.343881 |
| bilinear | 8 | 0.892302 | 0.874023 | 0.263574 | 0.289319 |

## Artifacts and compute

- concat: selected epoch8, selected consumption269,568; whole run269,568.
  Total28.809s, training/evaluation 26.781s, peak VRAM1,919,623,680bytes.
  Rotated-image Acc0.096680; best readout593,080bytes.
- bilinear: selected epoch8, selected consumption269,568; whole run269,568.
  Total26.429s, training/evaluation 24.619s, peak VRAM2,093,685,248bytes.
  Rotated-image Acc0.095703; best readout593,016bytes.

FP32, local RTX3080Laptop16GB; cloud cost$0. Two separate successful
eight-image smoke processes preceded full runs; smoke weights never reused.
Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/dinov2-interaction-{policy}-01`.
Exact source commit, commands, versions, base/cache hashes and result hashes
in companion JSON. Best/final separate and reload exact; no exact optimizer resume.
Old checkpoints, feature caches and corpora unchanged. Four-modality95–98%
remains unproven; do not extrapolate a taxonomy result to product readiness.

## Decision and limits

Retain bilinear as the stronger measured DINO readout candidate: all four
taxonomy Accuracies rise and aggregate NLL/Brier/ECE improve at essentially
equal parameter count. Color/texture ECE individually worsens, so aggregate
calibration improvement is not a claim that every task is better calibrated.
Train89.23% versus validation87.40% at selected epoch8; NLL0.2636 versus
0.2893. The endpoint is best NLL, so no clean plateau has been established.
No automatic extra epochs or product promotion. Count59.77% remains weak.

The existing larger VideoPrism reference remains87.89% on these IDs; the
new DINO result does not establish superiority or statistical noninferiority
to it. The full DINO/text/readout stack is39,594,112 parameters before
quantization. Its improvement supports a readout bottleneck in this fixed
image setup, not a universal frozen-feature claim or isolated causal proof
about one mathematical operation.

Next useful transfer test is the same interaction change on fixed video
features with text-only and shuffled-scene controls. Do not infer temporal
reasoning from these static image results or discard the controls.
