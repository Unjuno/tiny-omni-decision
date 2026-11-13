# Broader original-text coverage result

Execution completed; development validation only.
Optimizer reset from control step1664; no exact resume. All old outputs preserved.
No isolated coverage-effect claim: training exposure and total updates both changed.

| Step | Selector | Original Acc | Original NLL | Curriculum Acc | Curriculum NLL |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.667203 | 0.615354 | 0.885243 | 0.815430 | 0.449164 |
| 128 | 0.647504 | 0.609843 | 0.872074 | 0.824219 | 0.422935 |
| 256 | 0.688724 | 0.614567 | 0.902278 | 0.811035 | 0.475171 |
| 384 | 0.678015 | 0.613780 | 0.875719 | 0.804199 | 0.480312 |
| 512 | 0.679515 | 0.615354 | 0.875277 | 0.807617 | 0.483753 |
| 640 | 0.657139 | 0.624803 | 0.865994 | 0.814941 | 0.448284 |
| 768 | 0.674922 | 0.621654 | 0.868211 | 0.798828 | 0.481634 |
| 896 | 0.644620 | 0.610236 | 0.875995 | 0.834961 | 0.413245 |
| 1024 | 0.626677 | 0.614961 | 0.855090 | 0.830566 | 0.398264 |
| 1152 | 0.620153 | 0.612205 | 0.850526 | 0.840332 | 0.389780 |
| 1280 | 0.629796 | 0.625591 | 0.861009 | 0.836426 | 0.398583 |
| 1408 | 0.618460 | 0.629528 | 0.848670 | 0.843750 | 0.388251 |
| 1536 | 0.617472 | 0.627953 | 0.837735 | 0.847656 | 0.397209 |
| 1664 | 0.612234 | 0.636614 | 0.836666 | 0.838379 | 0.387801 |
| 1792 | 0.605480 | 0.635827 | 0.835591 | 0.840820 | 0.375370 |
| 1920 | 0.649468 | 0.627559 | 0.839596 | 0.829102 | 0.459340 |
| 2048 | 0.608961 | 0.629921 | 0.851505 | 0.848633 | 0.366416 |

## Selected checkpoint

| Source/type | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| original | 0.635827 | 0.835591 | 0.470606 | 0.018915 |
| curriculum | 0.840820 | 0.375370 | 0.214324 | 0.016823 |
| original:choice | 0.607143 | 0.964864 | 0.507640 | 0.019454 |
| original:noul | 0.740628 | 0.519355 | 0.349726 | 0.022057 |
| original:score | 0.499083 | 1.169201 | 0.621023 | 0.057137 |
| curriculum:conditional_selection | 0.710938 | 0.747521 | 0.408208 | 0.064063 |
| curriculum:count | 0.562500 | 0.935010 | 0.531066 | 0.097462 |
| curriculum:extremum | 0.896484 | 0.235327 | 0.140419 | 0.011995 |
| curriculum:temporal_order | 0.933594 | 0.189560 | 0.106822 | 0.031118 |

Best step1792; selected encounters14336, unique states5735.
Full run16384 unique questions (8192 original+8192 curriculum); no within-run repeats.
Original rows were seen in inherited final-block training; curriculum was reused.
Total loop1248.211s, evaluation613.269s,
peak allocatedVRAM3621577216bytes. Cloud charges0; electricity unmeasured.
Total22713601 parameters; trainable10647169. Best/final separate; selected reload exact.
All probability rows and paired state-bootstrap comparisons are in JSON/local artifacts.
Bootstrap10000 draws/seed17/percentile95%; selection validation is not blind.
Rolling training CE uses changing examples; it is not a fixed-sample generalization gap.
No transfer/audit evaluation or automatic adoption. Other modalities not re-evaluated.
Artifact `C:/CodexArtifacts/pretrained-reuse-r2/text-coverage-training-01`; source `9dcf16398ba9f504adba53e98e74b06835fa8ffc`.

## Measured interpretation

Selected original Accuracy improves2.047pp (95% paired state-bootstrap interval
+0.399 to+3.691pp); curriculum improves2.539pp (+0.879 to+4.199pp). NLL/Brier
intervals favor the selected checkpoint in both sources. Intervals are diagnostic:
checkpoint selection reused these labels, no multiple-comparison adjustment was
made, and this is a single seed. Do not call this independent generalization proof.
Original predictions correct249 earlier errors but introduce197 new errors;
curriculum corrects179 and introduces127. Preserve the trade-offs, not only net gain.
All seven recorded source/task Accuracy point estimates improve at selected1792;
score remains49.91% and conditional counting56.25%, far from95%.

Final2048 originalAccuracy62.99% is lower than selected63.58%, and originalNLL
rises0.835591 to0.851505. Curriculum finalAccuracy84.86%/NLL0.366416 improves,
so averaging conceals a source trade-off. Best selection matters; do not replace
best with final. Rolling loss alone cannot establish a fixed-train overfitting gap.
There is improvement within this budget, but neither a clean global plateau nor
proof that broader data alone caused the gain. Do not automatically extend steps.

Full run consumes16384 questions over6083 states; selected14336 over5735 states.
Selected source counts7168+7168 differ from full8192+8192. Original source full
counts: choice3221/noul3238/score1733. Reused curriculum: extremum4096,
temporal_order2048,count1024,conditional_selection1024. No within-run ID repeats.

Keep this text candidate separately. Existing shared image/video heads were
trained on the older frozen encoder; do not silently replace their text tower.
No new four-modality quality result, held-out-domain transfer or sealed audit.

The timed loop includes concurrent bounded CPU tests/report-code work; it is
an end-to-end local experiment duration, not an isolated throughput benchmark.
Model loading/tokenization precedes the loop timer;613.269s is scheduled validation,
while final best reload validation is also included in1248.211s total loop time.
The full command is in the immutable localconfig. Best/final hashes and every
artifact hash are included in JSON. Original model/data/checkpoints are preserved.
