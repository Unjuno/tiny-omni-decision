# Paired comparison-word diagnosis

Execution PASS; sourceb2e81f8.128new states,4cost/risk extrema questions/state,8pairedconditions,4,096inputs/model. Numeric-only and lexical-range selected1408weights fixed. No training, calibration or selection. This focused diagnostic excludes other task families and cannot replace their benchmark scores.

| Condition | Numeric Acc | Lexical Acc | Numeric NLL | Lexical NLL | Lexical Brier | Lexical ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base-minimum-select | 0.89648 | 0.91211 | 0.23794 | 0.22550 | 0.13210 | 0.02656 |
| base-minimum-which | 0.89453 | 0.90039 | 0.26271 | 0.23911 | 0.14158 | 0.02750 |
| base-smallest-select | 0.45117 | 0.52734 | 2.37145 | 1.57896 | 0.63015 | 0.19695 |
| base-smallest-which | 0.48828 | 0.57031 | 1.52358 | 1.29685 | 0.56337 | 0.15072 |
| expanded-minimum-select | 0.77930 | 0.76758 | 0.54839 | 0.58027 | 0.32152 | 0.09171 |
| expanded-minimum-which | 0.68359 | 0.78320 | 0.89438 | 0.57131 | 0.31250 | 0.06758 |
| expanded-smallest-select | 0.50977 | 0.65820 | 1.77371 | 0.90503 | 0.47820 | 0.14094 |
| expanded-smallest-which | 0.42969 | 0.68945 | 2.42872 | 0.83487 | 0.44735 | 0.11361 |

## Within-model paired effects

Word effect: smallest/largest minus minimum/maximum, at fixed structure/range. Structure effect: which minus select, at fixed vocabulary/range.

| Model | Effect | Accuracy delta (pp) | 95% interval (pp) | NLL delta |
| --- | --- | ---: | --- | ---: |
| parent | word:base:select | -44.53 | [-46.88, -41.99] | 2.13351 |
| parent | word:base:which | -40.62 | [-43.75, -37.30] | 1.26087 |
| parent | structure:base:minimum | -0.20 | [-2.34, 1.95] | 0.02477 |
| parent | structure:base:smallest | 3.71 | [1.17, 6.25] | -0.84787 |
| parent | word:expanded:select | -26.95 | [-31.05, -23.05] | 1.22532 |
| parent | word:expanded:which | -25.39 | [-29.88, -21.09] | 1.53434 |
| parent | structure:expanded:minimum | -9.57 | [-12.89, -6.25] | 0.34599 |
| parent | structure:expanded:smallest | -8.01 | [-11.72, -4.30] | 0.65501 |
| candidate | word:base:select | -38.48 | [-41.21, -35.74] | 1.35346 |
| candidate | word:base:which | -33.01 | [-36.33, -29.69] | 1.05774 |
| candidate | structure:base:minimum | -1.17 | [-3.32, 0.98] | 0.01361 |
| candidate | structure:base:smallest | 4.30 | [1.17, 7.42] | -0.28212 |
| candidate | word:expanded:select | -10.94 | [-14.65, -7.23] | 0.32476 |
| candidate | word:expanded:which | -9.38 | [-12.50, -6.25] | 0.26356 |
| candidate | structure:expanded:minimum | 1.56 | [-0.39, 3.71] | -0.00896 |
| candidate | structure:expanded:smallest | 3.12 | [0.20, 6.25] | -0.07016 |

## Consequence for the next action

Comparison-word sensitivity is supported within this controlled setup: changing minimum/maximum to smallest/largest at fixed state, number range and sentence structure produces large losses in both models. Sentence structure alone has smaller or mixed effects, including interactions with range/vocabulary. This does not prove that embeddings alone or any particular layer is the cause.

Lexical training improves the smallest/largest arms, especially expanded-range which (42.97%→68.95%), but basic-range smallest/largest remains52.73–57.03% against90.04–91.21% for minimum/maximum. Thus the previous novel-minimum wording test understated the remaining semantic gap. Retain that result alongside this one rather than replacing it.

The next intervention should cover comparison vocabulary and bidirectional condition consistency while retaining prior tasks; a further numeric-only budget increase is poorly targeted. Any intervention still requires fixed held-out vocabulary/structure tests and numeric-range retention checks. No inference-time favorable-word substitution, benchmark simplification, or general95% claim is justified.

## Limits and reproducibility

Known failure-guided development diagnosis, single training seed, four-item synthetic English.128-state paired bootstrap10,000replicates,seed20261012,percentile95% intervals for Accuracy; other metric differences are point estimates. No multiplicity or model-selection correction. Same selected-weight hashes as prior lexical-transfer protocol; all IDs/options/targets/probabilities checked, no examples removed.

FP32 local CUDA; 27.748s; peak allocatedVRAM 180766208bytes. Input SHA49593515a179956af2b64dac7af79e63ea3d1d87be7163d87c2cd7d7effee27c. Full per-condition and paired model metrics/file hashes in JSON. Artifacts `C:\CodexArtifacts\pretrained-reuse-r2\comparative-word-eval-01`. No old artifact overwrite or cloud charge.

Reproduce generation with `python -m scripts.comparative_word_diagnostic prepare --root <artifact-root> --output <new-input-dir>` and inference with `evaluate --root <artifact-root> --model <pinned-model-dir> --data <input-dir> --input-sha256 49593515a179956af2b64dac7af79e63ea3d1d87be7163d87c2cd7d7effee27c --output <new-output-dir>`.
