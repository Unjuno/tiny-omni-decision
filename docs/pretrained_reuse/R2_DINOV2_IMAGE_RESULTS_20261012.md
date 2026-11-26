# Frozen DINOv2 CLEVR-4 baseline results

Execution PASS, source128aa7d. Eight fixed epochs completed; earliest minimum validation NLL selects epoch8. Exact saved-best reload and frozen weight hashes passed. No old artifact changed, no reporting1595-row or sealed split read.

| Epoch | Fixed train Acc | Fixed train NLL | Validation Acc | Validation NLL | Validation Brier | Validation ECE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.71243 | 0.84433 | 0.70996 | 0.84618 | 0.37068 | 0.10786 |
| 2 | 0.76606 | 0.62797 | 0.76953 | 0.64093 | 0.29947 | 0.05225 |
| 3 | 0.81995 | 0.48578 | 0.81445 | 0.51363 | 0.24142 | 0.04941 |
| 4 | 0.82995 | 0.43928 | 0.82129 | 0.45754 | 0.22506 | 0.03165 |
| 5 | 0.84043 | 0.41212 | 0.83008 | 0.42813 | 0.21637 | 0.02567 |
| 6 | 0.85108 | 0.38087 | 0.82910 | 0.41585 | 0.21173 | 0.00873 |
| 7 | 0.87076 | 0.33418 | 0.84717 | 0.37807 | 0.19915 | 0.02181 |
| 8 | 0.86725 | 0.33250 | 0.85059 | 0.37786 | 0.19347 | 0.01554 |

| Task | Matched Acc | NLL | Brier | ECE | Rotated image Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| color | 0.96484 | 0.10985 | 0.05496 | 0.01381 | 0.09375 |
| texture | 0.90625 | 0.29017 | 0.14067 | 0.02567 | 0.09570 |
| count | 0.53711 | 1.08385 | 0.56605 | 0.05391 | 0.08594 |
| shape | 0.99414 | 0.02757 | 0.01221 | 0.00546 | 0.11133 |

## Same-ID comparison and interpretation

All512selection images /2,048questions match the saved prior VideoPrism predictions by ID, target, options and order. VideoPrism Accuracy87.89%, NLL0.28781, Brier0.15422, ECE0.02187; DINOv2 Accuracy85.06%, NLL0.37786, Brier0.19347, ECE0.01554. Lower ECE alone is not higher quality. Paired image bootstrap (512groups,10,000replicates,seed17,percentile95%) gives DINO-minus-VideoPrism Accuracy−2.83pp, interval[−4.30,−1.37]pp. Repeated checkpoint-selection validation is not a blind comparison.

DINOv2 color96.48% and shape99.41% meet the aspirational threshold on these specific synthetic taxonomy tasks; texture90.63% and count53.71% do not. Do not report general image95%, discard count, or redefine the task. The matched-versus-cyclic+1image control drops near10%, supporting use of image content. Options themselves encode a fixed taxonomy; this does not test arbitrary user-written criteria.

Train86.73% versus validation85.06%, NLL0.33250 versus0.37786 at selected epoch8. This small aggregate gap does not prove absence of per-task overfitting. Both losses broadly fall across the fixed budget; no clean plateau was demonstrated. Low count performance cannot yet uniquely identify data, pooling, readout capacity or backbone limits. No additional epochs/seeds were launched.

Same training image coverage, fixed options, MiniLM candidate embeddings, scorer size and eight-epoch optimizer recipe as the prior reference. Backbone features and preprocessing differ: whole640x480image is resized224x224 without crop, which preserves objects but distorts aspect ratio. This is a practical architecture comparison, not encoder-only causality.

## Parameters, exposure and cost

DINOv2-small22,056,576 + frozen MiniLM-L3 17,389,824 + trained scorer147,713 =39,594,113 resident parameters. Prior VideoPrism image stack131,902,721. This is a measured image-path reduction, not a complete Omni model or native-video replacement. Existing higher-quality reference remains retained; do not adopt the smaller model solely for size.

8,424unique training images;33,696unique image/taxonomy questions. Eight epochs consume269,568questions; selected epoch8 uses the same total. Repeats are intentional and not counted as new unique coverage.512independent validation images; four questions each are correlated. All source hashes and feature row ordering are pinned in config.

Feature extraction107.71s; readout training/evaluation/reload24.66s; total133.41s. RTX3080Laptop, FP32, peak CUDA allocation1951523328bytes, no cloud costs. These are batch experiment timings, not deployed request latency.

Artifacts `C:\CodexArtifacts\pretrained-reuse-r2\dinov2-clevr4-probe-01`: best/final weights, cached features, histories, configs and matched/rotated probabilities. Best SHA `3a19fe2b2f4f09abb27f1773479808edc422b8af11ffde496f4953157a08145c`; final SHA `7ebb386f5f57ebcf5f41b4d670a375f1408ce92aa4143e26c7515a7b43aaa8da`. Serialization hashes differ although selected epoch is final; exact reload checks establish predictions. Weights-only, not exact optimizer/RNG resume.

CPU tests362passed1skipped; Ruff passed. Smoke used8train+8validation images and one epoch in a separate process/output; encoder repeat exact, finite checks and readout reload passed. It was an implementation check, not a quality estimate. No live training process was interrupted.

## Next evidence needed

Global CLS+patch-mean features remain a promising small perception baseline for some tasks, but counting is unresolved. Before replacing the current encoder, compare a representation retaining spatial information or a justified controlled readout budget on validation; do not rerun pooled temporal controls. Broader image conditions, true video temporal reasoning, text transfer, joint decisions and calibrated95–98% across four modalities remain open. No product promotion or compression.
