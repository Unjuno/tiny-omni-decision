# Matched physical-mirror augmentation comparison

Execution PASS: two fresh seed17 runs completed24epochs, with identical initial
weights and shuffled question order. Both best reloads match exactly. All
frozen weights and pinned cache hashes retained; global-feature parity passed.
Source commit `4db01f156d5a7ee18d0b577d8fa2ddf503fdcf04`; exact argv/config in companion JSON.

Each condition uses40,331,392 total parameters,884,992 trainable. Same width
and architecture; only training observation view policy differs.
Train8,424 unique images/33,696 questions; validation512 images/2,048 questions.
Per run808,704 encounters,775,008 repeats. No new data or sealed audit.
Best selected by earliest minimum validation NLL; final saved separately.

| Condition | Best epoch | Selected encounters | Acc | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| original | 16 | 539136 | 0.900391 | 0.253458 | 0.140201 | 0.022108 |
| augmented | 19 | 640224 | 0.901855 | 0.258652 | 0.134151 | 0.015088 |

## Per-task selected metrics

| Condition | Task | Acc | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| original | color | 0.974609 | 0.059791 | 0.035664 | 0.012743 |
| original | texture | 0.923828 | 0.228276 | 0.107183 | 0.043987 |
| original | count | 0.705078 | 0.722229 | 0.416147 | 0.059495 |
| original | shape | 0.998047 | 0.003538 | 0.001810 | 0.002769 |
| augmented | color | 0.982422 | 0.044976 | 0.026463 | 0.012436 |
| augmented | texture | 0.929688 | 0.283977 | 0.107620 | 0.038761 |
| augmented | count | 0.695312 | 0.702377 | 0.401010 | 0.027631 |
| augmented | shape | 1.000000 | 0.003277 | 0.001510 | 0.002715 |

## Complete learning curves

| Condition | Epoch | Train Acc | Val Acc | Train NLL | Val NLL |
|---|---:|---:|---:|---:|---:|
| original | 1 | 0.807158 | 0.796875 | 0.509604 | 0.515996 |
| original | 2 | 0.829980 | 0.813477 | 0.444109 | 0.474014 |
| original | 3 | 0.858114 | 0.852539 | 0.379498 | 0.403984 |
| original | 4 | 0.885862 | 0.866699 | 0.294624 | 0.334208 |
| original | 5 | 0.888948 | 0.881836 | 0.279491 | 0.311759 |
| original | 6 | 0.895240 | 0.878418 | 0.257927 | 0.292091 |
| original | 7 | 0.893756 | 0.865234 | 0.287687 | 0.365896 |
| original | 8 | 0.905330 | 0.886230 | 0.241485 | 0.311522 |
| original | 9 | 0.915153 | 0.894043 | 0.207222 | 0.272289 |
| original | 10 | 0.913669 | 0.895996 | 0.209626 | 0.289909 |
| original | 11 | 0.908001 | 0.889160 | 0.236226 | 0.328612 |
| original | 12 | 0.914975 | 0.891113 | 0.222760 | 0.298304 |
| original | 13 | 0.909782 | 0.889160 | 0.223066 | 0.304555 |
| original | 14 | 0.919605 | 0.889648 | 0.191472 | 0.278034 |
| original | 15 | 0.916370 | 0.890625 | 0.211785 | 0.289143 |
| original | 16 | 0.931119 | 0.900391 | 0.168861 | 0.253458 |
| original | 17 | 0.917409 | 0.888184 | 0.207047 | 0.301962 |
| original | 18 | 0.927054 | 0.894043 | 0.179394 | 0.291422 |
| original | 19 | 0.934414 | 0.899414 | 0.164273 | 0.284145 |
| original | 20 | 0.928508 | 0.886230 | 0.177556 | 0.320532 |
| original | 21 | 0.919723 | 0.888672 | 0.207918 | 0.349165 |
| original | 22 | 0.943910 | 0.896973 | 0.148432 | 0.280281 |
| original | 23 | 0.933642 | 0.885254 | 0.155897 | 0.302459 |
| original | 24 | 0.935185 | 0.885254 | 0.162839 | 0.338654 |
| augmented | 1 | 0.802499 | 0.796875 | 0.523799 | 0.526360 |
| augmented | 2 | 0.830306 | 0.816406 | 0.427497 | 0.450353 |
| augmented | 3 | 0.855829 | 0.854004 | 0.382832 | 0.392347 |
| augmented | 4 | 0.870489 | 0.868652 | 0.325778 | 0.357063 |
| augmented | 5 | 0.871201 | 0.854980 | 0.336904 | 0.355602 |
| augmented | 6 | 0.890491 | 0.882812 | 0.272913 | 0.305130 |
| augmented | 7 | 0.884675 | 0.869141 | 0.301804 | 0.354032 |
| augmented | 8 | 0.895833 | 0.882812 | 0.262294 | 0.316831 |
| augmented | 9 | 0.907882 | 0.895508 | 0.222221 | 0.259231 |
| augmented | 10 | 0.905983 | 0.895508 | 0.232498 | 0.272120 |
| augmented | 11 | 0.897792 | 0.895508 | 0.266203 | 0.297178 |
| augmented | 12 | 0.897881 | 0.890625 | 0.272685 | 0.308524 |
| augmented | 13 | 0.908535 | 0.889648 | 0.213869 | 0.259877 |
| augmented | 14 | 0.898089 | 0.879395 | 0.247126 | 0.314126 |
| augmented | 15 | 0.903223 | 0.888184 | 0.250739 | 0.290039 |
| augmented | 16 | 0.914382 | 0.889160 | 0.206410 | 0.267271 |
| augmented | 17 | 0.909396 | 0.886719 | 0.221236 | 0.290799 |
| augmented | 18 | 0.910731 | 0.893555 | 0.223840 | 0.282962 |
| augmented | 19 | 0.922305 | 0.901855 | 0.195320 | 0.258652 |
| augmented | 20 | 0.915361 | 0.892090 | 0.208224 | 0.308245 |
| augmented | 21 | 0.903312 | 0.885742 | 0.242620 | 0.328547 |
| augmented | 22 | 0.925956 | 0.901855 | 0.184230 | 0.267162 |
| augmented | 23 | 0.911265 | 0.884277 | 0.221365 | 0.320177 |
| augmented | 24 | 0.921979 | 0.891113 | 0.194063 | 0.318465 |

## Paired comparisons

- augmented_minus_original: +0.146pp,95% percentile interval[-0.928,+1.270]pp.
- original_minus_small: +0.781pp,95% percentile interval[-0.537,+2.051]pp.
- augmented_minus_small: +0.928pp,95% percentile interval[-0.391,+2.246]pp.

Paired512-image bootstrap,10,000 draws,seed17. Validation has been reused
for model selection; these intervals do not remove selection bias.

## Compute and controls

Total175.605s; peak allocated VRAM529,484,288bytes; NVIDIA GeForce RTX 3080 Laptop GPU; cloud cost$0.
No encoder extraction; existing feature cache only. Rotated-image Accuracy:
- original: 0.096191.
- augmented: 0.096191.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/dinov2-mirror-control-01`.
Weights stay outside Git. Full checkpoint hashes, gradients and task curves
are recorded in JSON. This narrow image diagnostic does not establish
general image reasoning or four-modality95–98% quality.

## Scope and safeguards

Original-only history matches all24 saved epochs exactly; verified in training.
Augmentation uses one independent seed17 mask per image/epoch, shared by4tasks.
Train metrics use original images for both models, not changing training views.
Realized mirrored counts are in JSON. Unique original assets remain8424.
No inference ensemble, calibration, new validation images or audit. Best and final
checkpoints are separate. This is one seed on reused selector data.

## Decision

No convincing Accuracy improvement:90.04% to90.19% is3 additional correct
answers out of2048, with paired interval spanning zero. Aggregate NLL is
worse (0.253458 to0.258652), although Brier/ECE improve. Counting Accuracy
decreases70.51% to69.53% (5/512 fewer), despite improved counting NLL/Brier/ECE.
Texture NLL worsens0.228276 to0.283977. Keep both records, retain the prior
lower-NLL reference; do not adopt mirror augmentation as a proven fix.

The small physical-mirror robustness diagnostic justified this controlled
experiment but did not predict a validation gain. This budget does not support
augmentation alone solving counting. Do not extend epochs or tune reflection
probabilities against this selector. Remaining spatial readout/object relation
limitations need a different, separately specified hypothesis.

Ruff passes;395 CPU tests pass,1skip. Existing checkpoints/corpora preserved.
