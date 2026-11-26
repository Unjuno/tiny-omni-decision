# Fixed affine endpoint: reused development validation

## Outcome

Execution PASS; the primary counting-improvement hypothesis is not supported by
its point estimates and remains uncertain within paired intervals. Do not adopt
the unregularized endpoint. Keep the previous runtime reference.

Exactly 512 existing development images / 2,048 questions, with identical crop
features, options and order. The 200-iteration endpoint was frozen before this
evaluation. No fitting, calibration, checkpoint selection, extra seed or budget
extension occurred. The reporting set and sealed audit were not opened.

## Validation metrics

| Endpoint | Task | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---|---:|---:|---:|---:|---:|
| initial | overall | 90.7715% | 0.224677 | 0.124737 | 0.012183 | 0.906650 |
| initial | color | 99.6094% | 0.006115 | 0.003003 | 0.004867 | 0.995680 |
| initial | texture | 97.2656% | 0.102003 | 0.048089 | 0.015795 | 0.972318 |
| initial | count | 66.4062% | 0.785480 | 0.445594 | 0.052186 | 0.661699 |
| initial | shape | 99.8047% | 0.005112 | 0.002263 | 0.003388 | 0.996901 |
| final | overall | 90.8691% | 0.208769 | 0.116869 | 0.017195 | 0.919199 |
| final | color | 99.4141% | 0.008794 | 0.005655 | 0.005042 | 0.996176 |
| final | texture | 98.2422% | 0.052940 | 0.028032 | 0.013049 | 0.978639 |
| final | count | 65.8203% | 0.771299 | 0.433162 | 0.061061 | 0.703735 |
| final | shape | 100.0000% | 0.002041 | 0.000628 | 0.001752 | 0.998248 |

## Paired differences: final minus initial

512 image groups, 10,000 bootstrap resamples, seed 17; linear percentile 95%
intervals. These are descriptive intervals on repeatedly used validation,
without adjustment for adaptive experiment history or multiple comparisons.
They do not establish blind test quality.

| Task | Metric | Difference | 95% lower | 95% upper |
|---|---|---:|---:|---:|
| overall | accuracy | 0.000977 | -0.009277 | 0.011230 |
| overall | nll | -0.015909 | -0.031590 | -0.001131 |
| overall | brier | -0.007868 | -0.016154 | 0.000014 |
| color | accuracy | -0.001953 | -0.005859 | 0.000000 |
| color | nll | 0.002679 | -0.000325 | 0.006628 |
| color | brier | 0.002651 | -0.000033 | 0.006106 |
| texture | accuracy | 0.009766 | -0.003906 | 0.023438 |
| texture | nll | -0.049063 | -0.089147 | -0.015393 |
| texture | brier | -0.020057 | -0.037338 | -0.004138 |
| count | accuracy | -0.005859 | -0.044922 | 0.033203 |
| count | nll | -0.014181 | -0.062591 | 0.035910 |
| count | brier | -0.012432 | -0.040333 | 0.015466 |
| shape | accuracy | 0.001953 | 0.000000 | 0.005859 |
| shape | nll | -0.003071 | -0.007105 | -0.000457 |
| shape | brier | -0.001635 | -0.004469 | 0.000016 |

## Fixed full-training versus validation gaps

These compare complete fixed training and validation sets, not rolling minibatches.

| Endpoint | Task | Train Acc | Val Acc | Train minus Val | Train NLL | Val NLL |
|---|---|---:|---:|---:|---:|---:|
| initial | overall | 91.4292% | 90.7715% | 0.6578% | 0.198187 | 0.224677 |
| initial | color | 100.0000% | 99.6094% | 0.3906% | 0.001975 | 0.006115 |
| initial | texture | 97.6971% | 97.2656% | 0.4314% | 0.063313 | 0.102003 |
| initial | count | 68.0199% | 66.4062% | 1.6137% | 0.726062 | 0.785480 |
| initial | shape | 100.0000% | 99.8047% | 0.1953% | 0.001398 | 0.005112 |
| final | overall | 94.4623% | 90.8691% | 3.5931% | 0.146408 | 0.208769 |
| final | color | 100.0000% | 99.4141% | 0.5859% | 0.001651 | 0.008794 |
| final | texture | 99.6795% | 98.2422% | 1.4373% | 0.020115 | 0.052940 |
| final | count | 78.1695% | 65.8203% | 12.3492% | 0.563198 | 0.771299 |
| final | shape | 100.0000% | 100.0000% | 0.0000% | 0.000668 | 0.002041 |

## Interpretation and next action

Counting loses 3/512 correct answers despite a 10.15-point training increase.
Its paired Accuracy/NLL/Brier intervals all cross zero. Its train-validation
Accuracy gap grows from 1.61 to 12.35 points and validation ECE worsens.
This supports concern about overfitting; it does not prove the encoder lacks
count information or that the model family has reached its ceiling.

Aggregate NLL improves (descriptive paired interval excludes zero), while
aggregate Accuracy changes by only 2/2,048 answers and its interval crosses zero.
Texture accounts for a larger NLL improvement; no task loses more than 2 points
in its Accuracy point estimate. Aggregate calibration nevertheless worsens.

The affine family can fit training data better, but unconstrained fitting has
not solved the weakest task on validation. Stop this unregularized extension.
A separately fixed small regularization comparison is justified before more
encoder capacity or another budget increase. Do not tune against the audit.
The broader text/audio/video/fusion and 95–98% product gates remain unmet.

## Reproducibility

Source commit: `1b2908c8ccf3e41ecba2736c2fba1a735066ae2b`. Protocol was committed before evaluation.
Command: `python -m scripts.evaluate_shared_affine_endpoint`.
Output: `C:\CodexArtifacts\pretrained-reuse-r2\shared-affine-endpoint-validation-01`.
Two CPU threads, float64, 0.544s measured evaluation/analysis loop;
no GPU training or cloud cost. This is not an end-to-end encoder latency claim.
Initial probability replay max delta 2.58311033e-06;
zero initial answer changes. Source weight/cache hashes remain unchanged.
Exact input, checkpoint, prediction and configuration hashes are in the adjacent
JSON file and local artifact directory. Both endpoint predictions are retained.

## Verification

Ruff passed; 583 CPU tests passed, 1 skipped. Model/dataset/catalog validation
and manifest audit passed. Target-commit GitHub Actions is checked after push.
