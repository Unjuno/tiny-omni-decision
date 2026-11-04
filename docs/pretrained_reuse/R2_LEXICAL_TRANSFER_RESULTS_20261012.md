# Fixed lexical versus numeric transfer result

Execution PASS; sourcec3f81d9. Seed127,256new states,10,240inputs/model. All group/content and paraphrase-template overlap checks passed before inference. Same fixed selected1408weights,22,713,601parameters each, no additional training, prompt search or calibration. Here parent means numeric-only warm start; candidate means lexical-range training.

| Arm | Numeric Acc | Lexical Acc | Difference (pp) | Paired95% interval (pp) | Numeric NLL | Lexical NLL |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| control | 0.80908 | 0.82080 | 1.17 | [-0.15, 2.44] | 0.43999 | 0.44609 |
| interpolation | 0.72363 | 0.71484 | -0.88 | [-2.44, 0.68] | 0.65020 | 0.70087 |
| extrapolation | 0.43994 | 0.44531 | 0.54 | [-1.12, 2.20] | 2.35435 | 2.40976 |
| wording | 0.80713 | 0.81055 | 0.34 | [-0.83, 1.51] | 0.48456 | 0.45167 |
| combined | 0.40234 | 0.44873 | 4.64 | [2.73, 6.49] | 1.92842 | 2.14117 |

| Arm | Numeric Brier | Lexical Brier | Numeric ECE | Lexical ECE |
| --- | ---: | ---: | ---: | ---: |
| control | 0.24969 | 0.24239 | 0.02496 | 0.01202 |
| interpolation | 0.35692 | 0.38099 | 0.04582 | 0.07934 |
| extrapolation | 0.81927 | 0.82597 | 0.29596 | 0.32459 |
| wording | 0.26770 | 0.25361 | 0.03796 | 0.01906 |
| combined | 0.83313 | 0.80711 | 0.27175 | 0.29382 |

## Interpretation

The chosen novel wording is already handled by the numeric model (80.71%); lexical coverage reaches81.05%, a smallAccuracy change with improved NLL/Brier/ECE. This template uses minimum/maximum, unlike the previously failed smallest/largest formulation. Novel full-sentence wording is not equivalent to novel comparison vocabulary. This experiment does not show the old wording failure is repaired, and its higher level must not replace that old observed result.

Interpolation Accuracy/calibration deteriorate slightly. Mixed extrapolation remains~44% with poor NLL/ECE. Combined Accuracy improves but NLL/ECE worsen; the additional correct decisions do not establish reliable confidence. Retain both models as research artifacts; neither is a general Decision solution. Extending this same recipe is not justified by these results alone.

Next language evidence should explicitly separate comparative vocabulary from sentence structure, on fixed new states and all predefined variants, without selecting favorable words. Before further training, determine whether the task definition and numeric representation—not simply a larger budget—are the limiting factor. Keep the broader image/audio/video work and total prequantization parameter gate in scope.

## Limits and execution

Single seed, synthetic four-item English tasks. State-bootstrap percentile95% intervals,10,000resamples,seed20261012; no multiple-comparison/selection correction. Development follow-up after observed failures, not blind audit. Same initialization/budget/order for the two training stages were checked; selected step coincides, but their token sequences differ by design.

Local CUDA FP32 inference 91.325s, peak allocatedVRAM 195885568bytes; maximum tokens 165, no truncation. Inputs SHA `39866689aa62fd2118e64d7b6c5a33232705070239034aeaf0cf2ef44b3dc3c6`. All per-task metrics and input/output hashes in JSON. Outputs `C:\CodexArtifacts\pretrained-reuse-r2\lexical-transfer-eval-01`. No model/media overwrite or cloud cost.

Reproduce with `python -m scripts.lexical_transfer prepare --root <artifact-root> --output <new-input-dir>`, then `evaluate --root <artifact-root> --model <pinned-base-dir> --data <input-dir> --input-sha256 39866689aa62fd2118e64d7b6c5a33232705070239034aeaf0cf2ef44b3dc3c6 --output <new-output-dir>`.
