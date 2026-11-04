# Full-image versus cropped global readout

Fixed seed17,24epochs completed. Same initialization and full question order
as the cropped reference, with exact best reload. All hashes checked.
Run source `896b06b11e2131ac0382c6d00e8577159a0ed883`; exact argv/config in companion JSON.
Only preprocessing policy changed: full image bilinear288x288 versus
resize-short-side288 then center crop. Wider field of view, aspect ratio
and effective horizontal resolution change together; no isolated crop-cause claim.

8,424 training images/33,696 questions;512 reused selector images/2,048 questions.
808,704 encounters,775,008 repetitions. No reporting images or sealed audit.
1,720,576 trainable parameters,133,475,584 total image-stack parameters.

| Policy | Best epoch | Selected encounters | Acc | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| crop | 24 | 808704 | 0.907715 | 0.224677 | 0.124737 | 0.012183 |
| full | 24 | 808704 | 0.907227 | 0.219238 | 0.124966 | 0.012256 |

## Task metrics

| Policy | Task | Acc | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| crop | color | 0.996094 | 0.006115 | 0.003003 | 0.004867 |
| crop | texture | 0.972656 | 0.102003 | 0.048089 | 0.015795 |
| crop | count | 0.664062 | 0.785480 | 0.445594 | 0.052186 |
| crop | shape | 0.998047 | 0.005112 | 0.002263 | 0.003388 |
| full | color | 0.998047 | 0.006149 | 0.002593 | 0.003524 |
| full | texture | 0.955078 | 0.114496 | 0.065998 | 0.015390 |
| full | count | 0.675781 | 0.753651 | 0.430736 | 0.045102 |
| full | shape | 1.000000 | 0.002657 | 0.000539 | 0.002499 |

## Paired image differences (full minus crop)

512-image groups,10,000 resamples,seed17,linear percentile95% intervals.
Reused validation and checkpoint selection limit inference; not blind testing.
Intervals are descriptive, without correction for multiple task comparisons.

| Scope | Metric | Difference | Lower | Upper |
|---|---|---:|---:|---:|
| all | accuracy | -0.000488 | -0.011719 | 0.010742 |
| all | nll | -0.005439 | -0.022317 | 0.010774 |
| all | brier | 0.000229 | -0.009028 | 0.009141 |
| color | accuracy | 0.001953 | -0.003906 | 0.009766 |
| color | nll | 0.000034 | -0.005125 | 0.005630 |
| color | brier | -0.000411 | -0.004719 | 0.004462 |
| texture | accuracy | -0.017578 | -0.037109 | 0.001953 |
| texture | nll | 0.012494 | -0.037880 | 0.059915 |
| texture | brier | 0.017910 | -0.008294 | 0.044401 |
| count | accuracy | 0.011719 | -0.027344 | 0.050781 |
| count | nll | -0.031828 | -0.081984 | 0.016553 |
| count | brier | -0.014858 | -0.040672 | 0.010028 |
| shape | accuracy | 0.001953 | 0.000000 | 0.005859 |
| shape | nll | -0.002455 | -0.008932 | 0.001811 |
| shape | brier | -0.001724 | -0.005776 | 0.000689 |

## Full-image learning curve

| Epoch | Train Acc | Val Acc | Train NLL | Val NLL |
|---|---:|---:|---:|---:|
| 1 | 0.783268 | 0.792969 | 0.605153 | 0.598522 |
| 2 | 0.808048 | 0.811523 | 0.505152 | 0.501829 |
| 3 | 0.861734 | 0.862305 | 0.365882 | 0.360793 |
| 4 | 0.866275 | 0.868164 | 0.340905 | 0.338789 |
| 5 | 0.859865 | 0.862305 | 0.350129 | 0.342851 |
| 6 | 0.877582 | 0.874023 | 0.309233 | 0.313524 |
| 7 | 0.886366 | 0.881348 | 0.290668 | 0.296272 |
| 8 | 0.891797 | 0.891113 | 0.268961 | 0.275647 |
| 9 | 0.891827 | 0.890625 | 0.269492 | 0.281027 |
| 10 | 0.894142 | 0.892578 | 0.268493 | 0.277272 |
| 11 | 0.890848 | 0.898926 | 0.272147 | 0.278295 |
| 12 | 0.889156 | 0.882812 | 0.269265 | 0.283412 |
| 13 | 0.895863 | 0.896484 | 0.261465 | 0.259542 |
| 14 | 0.907645 | 0.903809 | 0.227186 | 0.235813 |
| 15 | 0.887553 | 0.883789 | 0.274973 | 0.287499 |
| 16 | 0.896753 | 0.893555 | 0.249827 | 0.264534 |
| 17 | 0.895715 | 0.883789 | 0.245316 | 0.251808 |
| 18 | 0.906013 | 0.906250 | 0.224604 | 0.240430 |
| 19 | 0.911087 | 0.900879 | 0.216605 | 0.230437 |
| 20 | 0.910672 | 0.905762 | 0.219600 | 0.231031 |
| 21 | 0.893637 | 0.886719 | 0.260648 | 0.282418 |
| 22 | 0.914975 | 0.903320 | 0.204342 | 0.223107 |
| 23 | 0.911740 | 0.906250 | 0.207400 | 0.222904 |
| 24 | 0.914025 | 0.907227 | 0.205416 | 0.219238 |

## Resources and controls

Extraction1116.001s CPU; readout102.320s; peak allocated VRAM592,128,000bytes.
Fixed rotated-image Accuracy0.095215,NLL14.727031.
All gradients finite; best and final stored separately. Local compute only,$0 cloud.
Artifacts: `C:\CodexArtifacts\pretrained-reuse-r2\shared-full-image-global-01` and `C:\CodexArtifacts\pretrained-reuse-r2\videoprism-full-image-cache-01`. Existing artifacts unchanged.
No arbitrary task95% or product-quality claim; other modality weaknesses remain.

## Interpretation and decision

Whole-image resizing does not establish a meaningful aggregate improvement:
one fewer correct answer overall; Accuracy,NLL and Brier difference intervals
all include zero. Counting gains6/512 answers (+1.17pp), with an interval
[-2.73,+5.08]pp. Texture loses9/512 answers. These are mixed, uncertain results.

This experiment does not support removing the crop as a sufficient solution
to counting. It also cannot prove that cropping has no effect: aspect ratio
and sampling density change together, and uncertainty remains substantial.
Keep the existing cropped runtime reference and preserve this alternative.
Do not select another preprocessing variant or extend this budget from these results.

Both arms select epoch24. Full-image train Accuracy91.40% versus90.72%
validation, NLL0.205416 versus0.219238, has a small aggregate gap.
Counting train68.93% versus validation67.58% remains weak in both splits.
NLL improves late in the run, so this is not a proven plateau or a model
capacity ceiling. The predeclared budget is complete; no automatic extension.

The weak counting task still needs a training-side representation/readout
diagnosis before another selector-based search. This result provides no
evidence for the general Omni95-98% objective or for quantization readiness.

Verification:578 CPU tests passed,1 skipped; Ruff and manifest/catalog
validation passed. Final pushed commit CI is checked separately.
