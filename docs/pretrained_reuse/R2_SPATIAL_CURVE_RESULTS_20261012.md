# Fixed spatial/global 24-epoch curve

Execution PASS. Both fresh seed17 runs exactly reproduce their entire
eight-epoch prior history before extending to24. No optimizer resume.
Only the884,993-parameter readout trains; the deployment stack remains
40,331,393 parameters including frozen DINOv2-small and MiniLM.
Cached image features are byte-hashed and reused, with original global
feature parity <=1e-5. No encoder update, new data or input normalization change.
Source commit `9c4bdb583a284dfc9c107ea7dea651926758b880`; full argv/hashes in companion JSON.

Train8,424 unique images/33,696 unique questions; same512 selector images
and2,048 questions. Per condition808,704 consumed questions at epoch24,
Repeated encounters are not new data: exact repeats =808,704−33,696=775,008.
No1595-image reporting subset or sealed audit opened. This is reused
development validation, not blind image quality or general image reasoning.

## Fixed learning curve

| Condition | Epoch | Train Acc | Validation Acc | Train NLL | Validation NLL |
|---|---:|---:|---:|---:|---:|
| global | 1 | 0.461806 | 0.458008 | 1.494067 | 1.491962 |
| global | 2 | 0.567426 | 0.564453 | 1.148315 | 1.148356 |
| global | 3 | 0.641322 | 0.624023 | 0.940179 | 0.956078 |
| global | 4 | 0.686996 | 0.687500 | 0.811429 | 0.837390 |
| global | 5 | 0.721985 | 0.711426 | 0.721759 | 0.752042 |
| global | 6 | 0.749555 | 0.736816 | 0.655573 | 0.672040 |
| global | 7 | 0.746676 | 0.724609 | 0.656301 | 0.681641 |
| global | 8 | 0.765432 | 0.760254 | 0.588615 | 0.602430 |
| global | 9 | 0.785939 | 0.776367 | 0.543922 | 0.571507 |
| global | 10 | 0.787660 | 0.769531 | 0.527718 | 0.565865 |
| global | 11 | 0.803864 | 0.793457 | 0.491450 | 0.530126 |
| global | 12 | 0.792527 | 0.780762 | 0.533473 | 0.563333 |
| global | 13 | 0.815349 | 0.813965 | 0.458901 | 0.488041 |
| global | 14 | 0.821314 | 0.795898 | 0.451302 | 0.503736 |
| global | 15 | 0.812886 | 0.800781 | 0.472092 | 0.498607 |
| global | 16 | 0.816032 | 0.806152 | 0.459709 | 0.488286 |
| global | 17 | 0.821463 | 0.804199 | 0.450541 | 0.495162 |
| global | 18 | 0.821077 | 0.808594 | 0.444000 | 0.476451 |
| global | 19 | 0.826982 | 0.809082 | 0.431967 | 0.464177 |
| global | 20 | 0.824371 | 0.810547 | 0.435483 | 0.480210 |
| global | 21 | 0.827843 | 0.810547 | 0.429496 | 0.478100 |
| global | 22 | 0.835233 | 0.813477 | 0.402382 | 0.445838 |
| global | 23 | 0.776858 | 0.766113 | 0.598188 | 0.659075 |
| global | 24 | 0.833304 | 0.814453 | 0.405312 | 0.455145 |
| spatial | 1 | 0.443910 | 0.449707 | 1.554444 | 1.553135 |
| spatial | 2 | 0.550273 | 0.550781 | 1.218684 | 1.233318 |
| spatial | 3 | 0.619539 | 0.623535 | 0.995707 | 0.994040 |
| spatial | 4 | 0.662363 | 0.669922 | 0.873524 | 0.879953 |
| spatial | 5 | 0.695186 | 0.698730 | 0.788754 | 0.782725 |
| spatial | 6 | 0.725605 | 0.709473 | 0.692312 | 0.711943 |
| spatial | 7 | 0.748754 | 0.728516 | 0.647897 | 0.695085 |
| spatial | 8 | 0.772644 | 0.764160 | 0.590185 | 0.624262 |
| spatial | 9 | 0.781873 | 0.777344 | 0.562271 | 0.583795 |
| spatial | 10 | 0.792557 | 0.782715 | 0.523426 | 0.548764 |
| spatial | 11 | 0.809562 | 0.792969 | 0.476674 | 0.512691 |
| spatial | 12 | 0.800926 | 0.783203 | 0.514614 | 0.551376 |
| spatial | 13 | 0.811283 | 0.790039 | 0.470379 | 0.513841 |
| spatial | 14 | 0.805110 | 0.780273 | 0.483864 | 0.539293 |
| spatial | 15 | 0.827517 | 0.820801 | 0.430605 | 0.473649 |
| spatial | 16 | 0.813242 | 0.812012 | 0.471613 | 0.488114 |
| spatial | 17 | 0.833571 | 0.822754 | 0.410507 | 0.434642 |
| spatial | 18 | 0.832413 | 0.815430 | 0.408179 | 0.462218 |
| spatial | 19 | 0.834787 | 0.826172 | 0.400208 | 0.421713 |
| spatial | 20 | 0.831790 | 0.807129 | 0.409205 | 0.471497 |
| spatial | 21 | 0.826567 | 0.805664 | 0.426915 | 0.459007 |
| spatial | 22 | 0.841999 | 0.817383 | 0.384411 | 0.434772 |
| spatial | 23 | 0.841138 | 0.821289 | 0.387344 | 0.446219 |
| spatial | 24 | 0.845264 | 0.824707 | 0.368731 | 0.453043 |

## Selected checkpoints

### global

Best epoch22, selected consumption741,312; whole run808,704. Earliest minimum validation NLL.
Final checkpoint retained separately; selected reload logits match exactly.

| Taxonomy | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| all | 0.813477 | 0.445838 | 0.220725 | 0.023978 |
| color | 0.968750 | 0.106290 | 0.051030 | 0.013626 |
| texture | 0.873047 | 0.419350 | 0.183605 | 0.035254 |
| count | 0.417969 | 1.237382 | 0.639931 | 0.074315 |
| shape | 0.994141 | 0.020331 | 0.008335 | 0.007494 |

Rotated-image Accuracy: 0.104004.

### spatial

Best epoch19, selected consumption640,224; whole run808,704. Earliest minimum validation NLL.
Final checkpoint retained separately; selected reload logits match exactly.

| Taxonomy | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| all | 0.826172 | 0.421713 | 0.214679 | 0.014706 |
| color | 0.966797 | 0.108703 | 0.055863 | 0.022581 |
| texture | 0.878906 | 0.363579 | 0.169939 | 0.030183 |
| count | 0.468750 | 1.183125 | 0.616220 | 0.035191 |
| shape | 0.990234 | 0.031447 | 0.016694 | 0.009397 |

Rotated-image Accuracy: 0.101074.

## Paired uncertainty

- global_selected_minus_parent8: +5.322pp; percentile95%[+3.662,+6.934]pp.
- spatial_selected_minus_parent8: +6.201pp; percentile95%[+4.492,+7.861]pp.
- spatial_minus_global_selected: +1.270pp; percentile95%[-0.342,+2.881]pp.

512-image paired bootstrap,10,000 draws,seed17; no simultaneous correction.
These intervals do not remove validation selection bias.

## Interpretation and next decision

More optimization improved both selected results versus epoch8. Eight
epochs alone was insufficient evidence for a plateau. However, the spatial
gain over the capacity-matched global head remains uncertain (interval
includes zero). Both trail the earlier147,713-parameter head's85.06% on
these same validation images. Keep that smaller DINO reference; do not
adopt this larger head or automatically extend the budget again.
The strongest retained VideoPrism image reference is87.89% on these IDs.

Counting remains weak:41.80% global,46.88% spatial versus53.71% for the
earlier small DINO head. Color/shape above95% do not meet all-image95%.
Spatial best NLL occurs at19 rather than24; later train fit improves
without better validation NLL. This is an overfitting/optimization warning,
not a proof of a backbone limit. Global epoch23 degrades on both train
and validation, then recovers: pure generalization overfit does not explain
every fluctuation. Read all curves and preserve these failed configurations.

A next spatial design must address representation/readout fit rather than
claiming more cell features necessarily help. No full encoder fine-tuning
or new data ingestion is justified by this diagnostic alone.

## Compute and preservation

- Total173.902s, cache loading 0.301s; no new encoder extraction.
- Peak allocated VRAM308,419,584bytes, GPUNVIDIA GeForce RTX 3080 Laptop GPU,cloud cost$0.
- Local artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/dinov2-spatial-curve-01`.
  Weights stay outside Git. Original caches, checkpoints and corpora retained.
- Hashes, task curves, gradient norms and source configuration in JSON.
- No automatic extension beyond24 or product adoption from this validation.
  Four-modality95–98% remains a separate unmet goal.
