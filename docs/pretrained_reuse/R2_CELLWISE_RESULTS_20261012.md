# Shared nonlinear cell readout: fixed24-epoch comparison

Execution PASS, exact best reload; all pinned feature/model hashes retained.
Source `359bfd9ac6cf9e3cdd628f1095f12408e77e86de`; exact argv/config/hashes in companion JSON.
Frozen DINO/MiniLM plus180736-parameter head =39,627,136 total parameters.
Train8424 unique images/33696 questions, validation512/2048, all options.
Best epoch15; selected encounters505440; total808704.
Final checkpoint saved separately. Repeated encounters are not independent data.
No augmentation, new validation, audit, calibration or ensemble.

| Task | Acc | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| all | 0.897949 | 0.253962 | 0.134592 | 0.022196 |
| color | 0.982422 | 0.055338 | 0.024479 | 0.007692 |
| texture | 0.945312 | 0.218400 | 0.081745 | 0.028910 |
| count | 0.664062 | 0.740191 | 0.431978 | 0.060901 |
| shape | 1.000000 | 0.001920 | 0.000164 | 0.001865 |

## Learning curve

| Epoch | Train Acc | Val Acc | Train NLL | Val NLL |
|---|---:|---:|---:|---:|
| 1 | 0.800214 | 0.779297 | 0.517069 | 0.538947 |
| 2 | 0.827517 | 0.815430 | 0.432530 | 0.466777 |
| 3 | 0.873516 | 0.876953 | 0.331988 | 0.342767 |
| 4 | 0.877404 | 0.870117 | 0.305914 | 0.346947 |
| 5 | 0.885535 | 0.875977 | 0.281115 | 0.312508 |
| 6 | 0.881262 | 0.880859 | 0.295906 | 0.335303 |
| 7 | 0.887405 | 0.862793 | 0.283828 | 0.340267 |
| 8 | 0.901175 | 0.888672 | 0.240034 | 0.271710 |
| 9 | 0.911859 | 0.890625 | 0.214155 | 0.282587 |
| 10 | 0.901442 | 0.882324 | 0.257318 | 0.323550 |
| 11 | 0.906369 | 0.895996 | 0.231273 | 0.280855 |
| 12 | 0.908149 | 0.889648 | 0.233959 | 0.283214 |
| 13 | 0.919753 | 0.901367 | 0.187784 | 0.257408 |
| 14 | 0.911414 | 0.880371 | 0.211070 | 0.298962 |
| 15 | 0.919605 | 0.897949 | 0.190821 | 0.253962 |
| 16 | 0.929101 | 0.897461 | 0.176514 | 0.264266 |
| 17 | 0.931802 | 0.899902 | 0.166883 | 0.267687 |
| 18 | 0.917112 | 0.894531 | 0.204873 | 0.290992 |
| 19 | 0.933286 | 0.905273 | 0.164553 | 0.260300 |
| 20 | 0.938568 | 0.901855 | 0.149798 | 0.271342 |
| 21 | 0.927350 | 0.886719 | 0.182566 | 0.322153 |
| 22 | 0.950617 | 0.910645 | 0.130008 | 0.264816 |
| 23 | 0.940913 | 0.897949 | 0.156515 | 0.313246 |
| 24 | 0.949252 | 0.899902 | 0.123950 | 0.286026 |

## Paired comparisons

- minus_spatial_bilinear: -0.244pp;95%[-1.611,+1.074]pp.
- minus_small_bilinear: +0.537pp;95%[-0.781,+1.855]pp.

Paired512-image percentile bootstrap,10000 draws,seed17. These reused
selector-data intervals do not remove checkpoint/model-selection bias.
This jointly changes nonlinearity, weight sharing and parameter count;
do not attribute results to only one factor or claim object-level reasoning.

Total96.417s, peak allocated VRAM346,956,800bytes, cloud$0.
Rotated-image Accuracy0.091797.
Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/dinov2-cellwise-01`.
No weights/features in Git; old artifacts preserved. General four-modality95% remains unmet.

## Interpretation and decision

The predeclared NLL selector chooses epoch15,89.79% Accuracy, not the91.06%
Accuracy maximum at22. Do not switch selection rules after viewing this curve.
Against spatial bilinear90.04%, the Accuracy difference is uncertain and NLL
is essentially similar but slightly worse (0.253962 vs0.253458). The head has
704256 fewer parameters, but exact counting drops70.51% to66.41%. Texture
improves92.38% to94.53%; this is a trade-off, not a counting fix.

Train NLL falls0.190821 at15 to0.123950 at24 while validation NLL rises
0.253962 to0.286026. This is evidence of overfitting after the selected point.
Retain the smaller-head artifact for cost comparison, without replacing the
counting reference or claiming robust superiority. No automatic training extension.
The result does not establish a frozen encoder ceiling: this head still averages
contextual cells rather than detecting or relating individual objects.

Ruff passes;396 CPU tests pass,1skip. Full best/final weights stay local.
