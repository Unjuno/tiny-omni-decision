# Shared VideoPrism spatial image comparison

Two fresh seed17 runs completed24epochs. Encoder and text weights frozen;
same initialization/question order verified, best reload exact, best/final
checkpoint and cache hashes verified. No reporting or sealed audit records.
Source commit `de5e538a47b91b1f88aeedc646054fbf1c9c1951`; exact argv and hashes in companion JSON.

Train8,424 images/33,696 questions; validation512 images/2,048 questions.
Each run808,704 encounters,775,008 repeats; these are not new examples.
Earliest minimum validation NLL selects best; final weights retained separately.
1,720,576 trained parameters;133,475,584 total image stack including full
VideoPrism114,365,184 and MiniLM17,389,824. This is not a complete Omni runtime.

| Condition | Best epoch | Selected encounters | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| global | 24 | 808704 | 0.907715 | 0.224677 | 0.124737 | 0.012183 |
| spatial | 22 | 741312 | 0.909180 | 0.227901 | 0.122006 | 0.017593 |
| reference: dino_spatial | — | — | 0.900391 | 0.253458 | 0.140201 | 0.022108 |
| reference: videoprism_pooled | — | — | 0.878906 | 0.287809 | 0.154222 | 0.021868 |

## Task metrics

| Condition | Task | Accuracy | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| global | color | 0.996094 | 0.006115 | 0.003003 | 0.004867 |
| global | texture | 0.972656 | 0.102003 | 0.048089 | 0.015795 |
| global | count | 0.664062 | 0.785480 | 0.445594 | 0.052186 |
| global | shape | 0.998047 | 0.005112 | 0.002263 | 0.003388 |
| spatial | color | 0.996094 | 0.012599 | 0.006732 | 0.005183 |
| spatial | texture | 0.955078 | 0.178506 | 0.068951 | 0.025028 |
| spatial | count | 0.687500 | 0.709433 | 0.408363 | 0.049541 |
| spatial | shape | 0.998047 | 0.011064 | 0.003980 | 0.002149 |

## Complete fixed-budget curves

| Condition | Epoch | Train Acc | Val Acc | Train NLL | Val NLL |
|---|---:|---:|---:|---:|---:|
| global | 1 | 0.799531 | 0.798340 | 0.561567 | 0.567464 |
| global | 2 | 0.835559 | 0.836426 | 0.437251 | 0.441371 |
| global | 3 | 0.869985 | 0.863281 | 0.343281 | 0.354473 |
| global | 4 | 0.875712 | 0.866699 | 0.319029 | 0.333295 |
| global | 5 | 0.875356 | 0.869141 | 0.311643 | 0.316374 |
| global | 6 | 0.888592 | 0.887695 | 0.271198 | 0.285073 |
| global | 7 | 0.891411 | 0.880371 | 0.269800 | 0.284882 |
| global | 8 | 0.889423 | 0.884277 | 0.273098 | 0.276518 |
| global | 9 | 0.896664 | 0.890625 | 0.252134 | 0.268499 |
| global | 10 | 0.895566 | 0.886719 | 0.259093 | 0.276746 |
| global | 11 | 0.904024 | 0.892578 | 0.244641 | 0.274335 |
| global | 12 | 0.897169 | 0.883789 | 0.255278 | 0.290100 |
| global | 13 | 0.903253 | 0.895508 | 0.238005 | 0.245945 |
| global | 14 | 0.908179 | 0.895508 | 0.220138 | 0.246240 |
| global | 15 | 0.897822 | 0.886230 | 0.248143 | 0.273570 |
| global | 16 | 0.910078 | 0.899414 | 0.213366 | 0.235094 |
| global | 17 | 0.911058 | 0.901855 | 0.209753 | 0.230134 |
| global | 18 | 0.913076 | 0.901855 | 0.206960 | 0.227759 |
| global | 19 | 0.911265 | 0.898926 | 0.211625 | 0.237484 |
| global | 20 | 0.914589 | 0.901367 | 0.208701 | 0.241225 |
| global | 21 | 0.909129 | 0.898926 | 0.217178 | 0.244829 |
| global | 22 | 0.912720 | 0.903320 | 0.210548 | 0.225109 |
| global | 23 | 0.911740 | 0.897949 | 0.208984 | 0.227195 |
| global | 24 | 0.914293 | 0.907715 | 0.198187 | 0.224677 |
| spatial | 1 | 0.811283 | 0.812988 | 0.537581 | 0.551993 |
| spatial | 2 | 0.849092 | 0.843262 | 0.405466 | 0.424481 |
| spatial | 3 | 0.874199 | 0.858887 | 0.332218 | 0.359938 |
| spatial | 4 | 0.884170 | 0.869629 | 0.292185 | 0.329786 |
| spatial | 5 | 0.888770 | 0.870605 | 0.283990 | 0.311082 |
| spatial | 6 | 0.899454 | 0.881348 | 0.247628 | 0.292309 |
| spatial | 7 | 0.912096 | 0.889648 | 0.225124 | 0.271234 |
| spatial | 8 | 0.908713 | 0.893555 | 0.230289 | 0.269421 |
| spatial | 9 | 0.912957 | 0.890625 | 0.209733 | 0.261600 |
| spatial | 10 | 0.911028 | 0.880859 | 0.221890 | 0.288433 |
| spatial | 11 | 0.924027 | 0.895020 | 0.192323 | 0.264971 |
| spatial | 12 | 0.918061 | 0.881348 | 0.198343 | 0.282514 |
| spatial | 13 | 0.919872 | 0.900391 | 0.207409 | 0.266091 |
| spatial | 14 | 0.929517 | 0.893066 | 0.172264 | 0.255864 |
| spatial | 15 | 0.927558 | 0.893555 | 0.175067 | 0.248041 |
| spatial | 16 | 0.932485 | 0.896484 | 0.165495 | 0.254139 |
| spatial | 17 | 0.932989 | 0.898438 | 0.163627 | 0.247186 |
| spatial | 18 | 0.937144 | 0.898438 | 0.154075 | 0.252750 |
| spatial | 19 | 0.935363 | 0.894531 | 0.164028 | 0.283679 |
| spatial | 20 | 0.943406 | 0.903809 | 0.141845 | 0.239531 |
| spatial | 21 | 0.944890 | 0.905273 | 0.140226 | 0.242377 |
| spatial | 22 | 0.955039 | 0.909180 | 0.120454 | 0.227901 |
| spatial | 23 | 0.938717 | 0.895996 | 0.164385 | 0.290750 |
| spatial | 24 | 0.947739 | 0.904785 | 0.128810 | 0.242774 |

## Paired-image accuracy differences

- spatial_minus_global: +0.146pp,95% interval[-0.879,+1.221]pp.
- global_minus_dino_spatial: +0.732pp,95% interval[-0.684,+2.150]pp.
- global_minus_videoprism_pooled: +2.881pp,95% interval[+1.562,+4.150]pp.
- spatial_minus_dino_spatial: +0.879pp,95% interval[-0.488,+2.295]pp.
- spatial_minus_videoprism_pooled: +3.027pp,95% interval[+1.611,+4.443]pp.

Paired512-image bootstrap,10,000 draws,seed17,linear percentile quantiles.
These intervals do not correct reused-validation/model-selection bias.
The DINO and old pooled VideoPrism references differ in preprocessing,
readout and/or encoder, so these are not single-factor backbone comparisons.
For the inspected640x480 training image, resize to384x288 then crop288x288
removes the outer25% of the original width. DINO used the full image resized
to224x224. This is a possible information-loss confound, not a measured cause
of count errors; no crop adjustment or further run is made in this comparison.
The new one-frame intermediate representation does not claim equality to
the legacy repeated-frame cache; the previous numerical parity gate failed.

## Compute and image dependence

CPU extraction1140.908s; GPU readout wall time203.668s. Peak allocated GPU memory593,045,504bytes; NVIDIA GeForce RTX 3080 Laptop GPU; cloud cost$0.
- global, fixed one-image rotation: Accuracy0.097168,NLL15.210860,Brier1.684266,ECE0.809482.
- spatial, fixed one-image rotation: Accuracy0.099121,NLL15.905672,Brier1.695870,ECE0.821377.

Artifacts: `C:\CodexArtifacts\pretrained-reuse-r2\shared-spatial-image-readout-01` and `C:\CodexArtifacts\pretrained-reuse-r2\videoprism-spatial-image-cache-01`. Old artifacts remain unchanged.
All best/final weights remain outside Git. This narrow attribute/counting
comparison does not establish general image reasoning, no-text operation,
simultaneous Omni fusion, or95–98% quality. No product promotion.

## Interpretation and next decision

Spatial retention adds only3 net correct answers out of2,048: its0.146pp
aggregate difference is uncertain. Count Accuracy improves66.41%->68.75%
but texture falls97.27%->95.51%. Brier improves while NLL/ECE worsen.
Do not claim that spatial retention is universally better or adopt it.

The spatial best has train95.50% versus validation90.92% Accuracy and
NLL0.120454 versus0.227901. Count train82.48% versus validation68.75%
shows a substantial generalization gap. Global has a smaller gap:
train91.43% versus validation90.77%, NLL0.198187 versus0.224677.
Spatial final24 is worse than best22; global selects24 and is not proven
plateaued. The fixed budget ends here; no additional epochs or seeds.

Both shared representations beat the legacy pooled VP result on these
reused IDs, with positive paired accuracy intervals. DINO differences
remain uncertain, and DINO spatial counting70.51% exceeds both new arms.
This supports further shared-tower integration checks, not general-image
superiority. The remaining counting weakness cannot be assigned solely
to crop, representation, capacity, or training budget by this experiment.

Preserve both alternatives and existing DINO. Prefer the global arm as
the conservative integration reference because its NLL/ECE and gap are
better, while explicitly retaining the count trade-off. A combined
Omni runtime and its multi-input quality remain unverified. No95% claim.

Verification:569 CPU tests pass,1 skipped; Ruff and all workflow
manifest/catalog validation commands pass locally. Final push CI is
checked independently. No paid compute or legacy artifact overwrite.
