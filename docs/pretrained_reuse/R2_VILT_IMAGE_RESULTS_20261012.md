# Frozen ViLT conditioned image readout

Both use8424 training images/33696 questions and512 validation images/2048 questions.
All four10-option taxonomies retained, including9 options absent from ViLT's closed head.
Only147712-parameter bilinear readout trained; encoders frozen, seed17,8epochs,
AdamW1e-3/weight-decay1e-4,batch64,earliest minimum validation NLL selection.
This compares complete representations/preprocessing, not fusion alone:
ViLT uses question-conditioned published384×512 input; DINO uses224×224 image features.
No prompt search, new data, sealed audit or1595-image reporting set evaluation.

| Model | Taxonomy | Accuracy | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| vilt | all | 0.811035 | 0.471817 | 0.238904 | 0.018794 |
| vilt | color | 0.996094 | 0.010858 | 0.004123 | 0.009253 |
| vilt | texture | 0.771484 | 0.663302 | 0.323659 | 0.040122 |
| vilt | count | 0.488281 | 1.169200 | 0.608363 | 0.038564 |
| vilt | shape | 0.988281 | 0.043908 | 0.019473 | 0.009545 |
| dino | all | 0.874023 | 0.289319 | 0.157210 | 0.011022 |
| dino | color | 0.978516 | 0.059637 | 0.029487 | 0.016945 |
| dino | texture | 0.921875 | 0.201637 | 0.109964 | 0.031818 |
| dino | count | 0.597656 | 0.889974 | 0.486729 | 0.042458 |
| dino | shape | 0.998047 | 0.006029 | 0.002658 | 0.003344 |

ViLT-minus-DINO -6.299pp, 95% paired interval[-8.008,-4.688]pp.
512-image paired percentile bootstrap,10000 draws,seed17; selection bias remains.

## Learning curve

| Epoch | Train Acc | Validation Acc | Train NLL | Validation NLL |
|---:|---:|---:|---:|---:|
| 1 | 0.730057 | 0.725586 | 0.735882 | 0.733183 |
| 2 | 0.763206 | 0.761719 | 0.626032 | 0.623614 |
| 3 | 0.780894 | 0.776367 | 0.561257 | 0.569897 |
| 4 | 0.792972 | 0.790039 | 0.533196 | 0.549524 |
| 5 | 0.798463 | 0.790039 | 0.511047 | 0.512315 |
| 6 | 0.808108 | 0.789551 | 0.495448 | 0.519162 |
| 7 | 0.811788 | 0.798828 | 0.474238 | 0.501768 |
| 8 | 0.817872 | 0.811035 | 0.459255 | 0.471817 |

## Resource/integrity results

- Best epoch8; selected269568 encounters, whole run269568;33696 unique questions and8424 unique images.
- Rotated-image Accuracy0.100586, NLL11.514954.
- Total resident parameters135,126,073, including unused VQA classifier.
- Extraction554.701s; train/eval24.800s; total580.456s; peak VRAM2093685248bytes; cost$0.
- Best/final checkpoints separate; exact best reload; original weights unchanged.
- Artifacts `C:/CodexArtifacts/pretrained-reuse-r2/vilt-image-probe-01`.
  All probabilities, feature caches, configs and hashes retained outside Git.
- Execution smoke completed separately before the full run.
- Fixed taxonomy performance is not general image/video/omni decision quality.
  ViLT feature extraction requires a new joint forward for each new question.

## Decision and limits

Do not replace the current DINO candidate on these results. ViLT is6.299pp lower
overall with a paired interval excluding zero; NLL/Brier/ECE are also worse.
Color improves to99.61%, but texture77.15% and count48.83% dominate the failures.
Image rotation reduces Accuracy to10.06%, consistent with image dependence on
these taxonomy tasks. It does not demonstrate general question understanding.

Best is the final epoch8: train81.79%/NLL0.4593, validation81.10%/NLL0.4718.
Both continue improving; this is not evidence of a clean plateau or an absolute
ViLT capability ceiling. The small gap and incomplete train fit leave additional
optimization as a possible explanation. Any budget extension must retain the
fixed feature hashes and report its extra encounters; do not overwrite this run.

The135.13M measured stack includes the unused5.99M VQA classifier and17.39M
candidate encoder. DINO's corresponding stack is39.59M. No tested video path
or all-modality95% result exists for ViLT. Preserve all prior candidates.
