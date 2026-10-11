# DINO bilinear24-epoch learning curve

Fresh seed17; complete first8-epoch history matches the original exactly.
Same pinned features, all8424/512 images, four10-option taxonomies,147712-parameter head.
No encoder changes, new data, re-extraction, sealed audit or reporting-set inference.
This is reused model-selection validation, not blind or general-image quality.

| Budget | Task | Accuracy | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| 24epochs | all | 0.892578 | 0.264363 | 0.141047 | 0.017840 |
| 24epochs | color | 0.988281 | 0.048747 | 0.022570 | 0.012395 |
| 24epochs | texture | 0.941406 | 0.241759 | 0.099353 | 0.033729 |
| 24epochs | count | 0.640625 | 0.764562 | 0.441308 | 0.040045 |
| 24epochs | shape | 1.000000 | 0.002384 | 0.000958 | 0.002018 |
| 8epochs | all | 0.874023 | 0.289319 | 0.157210 | 0.011022 |
| 8epochs | color | 0.978516 | 0.059637 | 0.029487 | 0.016945 |
| 8epochs | texture | 0.921875 | 0.201637 | 0.109964 | 0.031818 |
| 8epochs | count | 0.597656 | 0.889974 | 0.486729 | 0.042458 |
| 8epochs | shape | 0.998047 | 0.006029 | 0.002658 | 0.003344 |

24-budget minus8-budget: +1.855pp; 95% interval[+0.586,+3.125]pp.
512-image paired percentile bootstrap,10000 draws,seed17; selection bias remains.

| Epoch | Train Acc | Val Acc | Train NLL | Val NLL |
|---:|---:|---:|---:|---:|
| 1 | 0.814043 | 0.805664 | 0.509039 | 0.518170 |
| 2 | 0.842177 | 0.837402 | 0.409898 | 0.429787 |
| 3 | 0.859123 | 0.855469 | 0.369991 | 0.381043 |
| 4 | 0.872952 | 0.869629 | 0.316525 | 0.343843 |
| 5 | 0.877552 | 0.870117 | 0.306238 | 0.324418 |
| 6 | 0.883755 | 0.873535 | 0.283585 | 0.309805 |
| 7 | 0.880134 | 0.859375 | 0.298052 | 0.343881 |
| 8 | 0.892302 | 0.874023 | 0.263574 | 0.289319 |
| 9 | 0.881737 | 0.871582 | 0.288257 | 0.331587 |
| 10 | 0.894468 | 0.890137 | 0.251203 | 0.289676 |
| 11 | 0.892361 | 0.875488 | 0.265384 | 0.298865 |
| 12 | 0.896575 | 0.880859 | 0.256035 | 0.285888 |
| 13 | 0.894112 | 0.875000 | 0.249170 | 0.293959 |
| 14 | 0.903757 | 0.885742 | 0.236024 | 0.290772 |
| 15 | 0.898415 | 0.875488 | 0.245401 | 0.295689 |
| 16 | 0.907585 | 0.879883 | 0.226875 | 0.286132 |
| 17 | 0.900938 | 0.878418 | 0.244667 | 0.310954 |
| 18 | 0.909663 | 0.890625 | 0.214841 | 0.273017 |
| 19 | 0.912542 | 0.890137 | 0.214556 | 0.293116 |
| 20 | 0.912571 | 0.880371 | 0.213004 | 0.289514 |
| 21 | 0.903193 | 0.879395 | 0.239726 | 0.308431 |
| 22 | 0.918299 | 0.889160 | 0.198127 | 0.272147 |
| 23 | 0.919812 | 0.892578 | 0.192118 | 0.264363 |
| 24 | 0.917765 | 0.888672 | 0.203681 | 0.279825 |

## Resource and artifact record

- Best epoch23; selected775008 encounters; whole run808704, unique33696 questions/8424 images.
- Rotated-image Accuracy0.096191.
- Parameters39,594,112; only147,712 trainable.
- Cache load0.056s; train/eval71.011s; total72.798s; peak VRAM2093685248 bytes; local cost$0.
- Best/final files separate; exact best reload; original weights/features unchanged.
- Artifacts `C:/CodexArtifacts/pretrained-reuse-r2/dinov2-bilinear-curve-01`; all predictions/hashes retained.
- Complete four-modality95% goal remains unmet.

## Interpretation

Retain this as a stronger narrow image candidate: Accuracy/NLL/Brier improve,
but aggregate ECE rises0.01102→0.01784. Texture Accuracy improves while its NLL
worsens0.20164→0.24176; calibration trade-offs must not be hidden.
Best23 train91.98% versus validation89.26% leaves a2.72pp gap; final24 worsens
both train and validation, so that step alone is not pure-overfitting evidence.
No clean plateau or universal capacity limit is established, but do not extend
epochs automatically. This candidate has not passed independent audit or calibration.

Saved predictions show220 errors:184 count,30 texture,6 color,0 shape. Thus count
accounts for83.64% of errors. Count absolute errors:328 exact,161 off-by-one,21
off-by-two,2 off-by-three. These are diagnostics; exact-count Accuracy remains
64.06%, and the benchmark tolerance is not relaxed. Shape100% is not image95%.
Next prioritize whether retained spatial information and candidate interaction
can improve exact counting, with unchanged options/splits, rather than declaring
the near-misses correct or adding epochs without a distinct hypothesis.
