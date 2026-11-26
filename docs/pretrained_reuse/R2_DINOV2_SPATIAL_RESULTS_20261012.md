# Capacity-matched spatial readout results

Execution PASS, source71021e9. Both fixed8-epoch runs completed; both select epoch8 by earliest minimum validation NLL. Initial tensors and sample-order hashes match exactly; saved-best reload exact. All old artifacts retained.

| Condition | Accuracy | NLL | Brier | ECE | Image-rotation Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| global | 0.76025 | 0.60243 | 0.29482 | 0.02125 | 0.10107 |
| spatial | 0.76416 | 0.62426 | 0.29583 | 0.02740 | 0.10498 |

| Task | Global Acc | Spatial Acc | Global NLL | Spatial NLL |
| --- | ---: | ---: | ---: | ---: |
| color | 0.87891 | 0.93164 | 0.34131 | 0.22492 |
| texture | 0.83008 | 0.76367 | 0.50285 | 0.69809 |
| count | 0.33398 | 0.36914 | 1.54741 | 1.54329 |
| shape | 0.99805 | 0.99219 | 0.01816 | 0.03076 |

| Condition | Epoch | Train Acc | Train NLL | Val Acc | Val NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| global | 1 | 0.46181 | 1.49407 | 0.45801 | 1.49196 |
| global | 2 | 0.56743 | 1.14831 | 0.56445 | 1.14836 |
| global | 3 | 0.64132 | 0.94018 | 0.62402 | 0.95608 |
| global | 4 | 0.68700 | 0.81143 | 0.68750 | 0.83739 |
| global | 5 | 0.72198 | 0.72176 | 0.71143 | 0.75204 |
| global | 6 | 0.74955 | 0.65557 | 0.73682 | 0.67204 |
| global | 7 | 0.74668 | 0.65630 | 0.72461 | 0.68164 |
| global | 8 | 0.76543 | 0.58861 | 0.76025 | 0.60243 |
| spatial | 1 | 0.44391 | 1.55444 | 0.44971 | 1.55314 |
| spatial | 2 | 0.55027 | 1.21868 | 0.55078 | 1.23332 |
| spatial | 3 | 0.61954 | 0.99571 | 0.62354 | 0.99404 |
| spatial | 4 | 0.66236 | 0.87352 | 0.66992 | 0.87995 |
| spatial | 5 | 0.69519 | 0.78875 | 0.69873 | 0.78273 |
| spatial | 6 | 0.72561 | 0.69231 | 0.70947 | 0.71194 |
| spatial | 7 | 0.74875 | 0.64790 | 0.72852 | 0.69509 |
| spatial | 8 | 0.77264 | 0.59018 | 0.76416 | 0.62426 |

## Interpretation and decision

Spatial minus matched-capacity global Accuracy+0.39pp; paired512-image bootstrap95% interval[-1.32,2.10]pp (10,000replicates,seed17,percentile). NLL/Brier/ECE worsen. Color improves while texture declines; there is no clear aggregate gain. This exploratory interval does not correct for multiple tasks or repeated validation selection.

Count36.91% versus33.40% in the matched control is still below the prior small-head53.71%. Both new mean accuracies (~76%) are worse than prior147,713-parameter head85.06%. Do not select only the favorable color/count comparison or adopt the larger head. Preserve the smaller-head baseline and the higher-quality VideoPrism reference.

This does not prove spatial information is useless: the larger input/readout changes initialization fan-in and optimization versus the old compact head, and both new learning curves are still improving. It shows this fixed4x4flattened representation and8-epoch recipe is not a demonstrated quality gain. The matched global/spatial comparison controls parameter count, initialization and sample order; the historical old-head comparison does not isolate one factor. No budget extension or extra seeds were launched.

The model predicts fixed synthetic taxonomies; candidate vocabulary reveals task identity. This is not arbitrary criteria following or video temporal reasoning. Global/image95% and the full four-modality95–98% objective remain unmet.

## Numerical failure, correction and integrity

Attempt01 stopped after training-feature extraction, before any optimizer step: nested FP32 means differed from the old direct mean by1.335144e-5. Preserve its config/output and do not treat it as a quality failure. A256-image diagnostic found direct extraction exactly identical to saved features; nested means differ in FP32 and exactly agree in FP64. Synthetic regression failed before the fix and passes after it.

A uniform per-channel centering offset across16cells aligns the original direct FP32 mean. It uses no labels and changes cell differences only by rounding. The1e-5gate is unchanged; corrected full-run max differences are2.861e-6train and1.907e-6validation. All image and source hashes/split disjointness remain checked. Smoke02 passed before attempt02. Source files and diagnosis are pinned in the companion JSON/Git history.

## Resources and artifacts

Each condition:40,331,393resident parameters,884,993trainable (vs147,713 prior). Both share frozen22,056,576DINOv2 and17,389,824MiniLM. Extraction102.18s; global28.90s; spatial29.01s; successful run total160.91s, peak CUDA allocation327687168bytes. Failed attempt and diagnostic consumed extra local compute; exact failed-attempt duration was not saved and is UNKNOWN. No cloud costs.

Each condition uses8,424unique images /33,696unique image-taxonomy questions and269,568consumptions. Selected epoch8 has the same exposure as full run. These two runs share data; do not sum their unique counts. Validation512images/2,048correlated questions. Per-task train/validation metrics and gradients are retained in every epoch record.

Artifacts `C:\CodexArtifacts\pretrained-reuse-r2\dinov2-spatial-control-02` include FP32 cell caches, separate global/spatial best/final weights, predictions and histories. All hashes are in the companion JSON. Weights-only checkpoints are not exact optimizer/RNG resume. No reporting1595-image subset or sealed audit was loaded. GPURTX3080Laptop; code uses bounded batches rather than expanding the entire image/option corpus in memory.

CPU tests365passed1skipped; Ruff passed. This tested representation is not promoted. Further work must address optimization and useful structured/conditional readout, not infer a backbone ceiling or claim that static-grid improvements would establish video understanding.
