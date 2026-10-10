# CLEVRER temporal-order control (2026-10-11)

## Scope

This is a diagnostic of temporal-order sensitivity for the already-fixed
VideoPrism-B + MiniLM readout on the 2,777-row, 200-scene development
validation set described in
[the new-scene video/text control report](CLEVRER_VIDEO_TEXT_CONTROLS_20261011.md).
It reuses the same questions, targets, text features, and readout checkpoint.
The sole intervention reverses the order of the same eight uniformly sampled
frames before frozen VideoPrism feature extraction. No model was trained,
selected, or calibrated. No sealed audit was loaded. This is neither a blind
evaluation nor a product-quality result.

## Procedure and integrity

- Dataset: `MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef`;
  200 scenes, 2,777 binary choices (explanatory 1,185, predictive 337,
  counterfactual 1,255). All questions from each scene remain grouped.
- VideoPrism-B revision `be719a406d563b66f0ac969e7c94bab8e997c81a`,
  checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`;
  eight frames, 288 px, frozen CPU/JAX feature extraction.
- Text encoder, scorer checkpoint, preprocessing and validation order are
  identical to the prior control. The forward scorer was reloaded and reproduced
  every saved probability exactly (maximum absolute delta `0.0`).
- Scene-paired bootstrap: 10,000 resamples, seed 17, percentile 95% interval.
  It resamples the 200 scenes with all their choices together.
- Existing feature cache and result artifacts were not overwritten. New reverse
  feature and result directories are under
  `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-temporal-reverse-features-v0`
  and `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-temporal-order-results-v0`.

## Results

Metric order is Accuracy / NLL / Brier / ECE-15. The overall row is example
weighted; macro is the unweighted mean across the three question types.

| Condition | Rows | Overall | Macro task type |
|---|---:|---:|---:|
| Forward frames | 2,777 | `0.50666 / 0.69371 / 0.50056 / 0.01695` | `0.49899 / 0.69432 / 0.50117 / 0.02426` |
| Reversed frames | 2,777 | `0.50630 / 0.69372 / 0.50057 / 0.01739` | `0.49873 / 0.69433 / 0.50118 / 0.02463` |

| Task | Rows | Forward Accuracy / NLL | Reversed Accuracy / NLL |
|---|---:|---:|---:|
| Explanatory | 1,185 | `0.51224 / 0.69324` | `0.51224 / 0.69323` |
| Predictive | 337 | `0.47478 / 0.69622` | `0.47478 / 0.69626` |
| Counterfactual | 1,255 | `0.50996 / 0.69349` | `0.50916 / 0.69350` |

Paired reversed-minus-forward intervals were Accuracy
`[-0.000818, 0.000000]`, NLL `[-0.0000124, 0.0000393]`, and Brier
`[-0.0000123, 0.0000390]`. The NLL and Brier intervals include zero; the
accuracy point difference is negligible and the interval is bounded by the
discrete scene-level resampling distribution. Both conditions are near chance.

## Interpretation and next step

The fixed readout is nearly invariant to reversing these frames. Combined with
the prior same-data text-only and shuffled-video controls, this provides no
evidence that this particular readout extracts useful scene-specific or
directional temporal evidence on these binary CLEVRER questions. It does not
prove VideoPrism lacks temporal information, nor establish a model-family
ceiling: the bottleneck may be the readout, representation pooling, question
construction, or task signal. Therefore do not proceed directly to Lens/merge
training based on this probe. The next R2 work should inspect task-label
construction and establish a prespecified temporal-sensitive control with a
fixed scorer before any costly shared-backbone experiment. All findings remain
development-validation evidence, with prior validation exposure explicitly
retained.

### Read-only label/choice audit

The frozen validation manifest contains 894 parent questions across 200
scenes. Positive-label rates are close to balanced overall and by task:
explanatory 590/1,185 (`49.79%`), predictive 174/337 (`51.63%`), and
counterfactual 615/1,255 (`49.00%`). A simple task-level majority label does
not explain the near-chance result. After exact-content exclusions, the number
of retained candidates per parent varies: explanatory parents retain 1–4 rows,
predictive 1–2, and counterfactual 1–4. Candidate `choice_id` positive rates
also vary modestly (predictive choice 0 `46.11%` vs choice 1 `57.06%`, for
example). Future controls should stratify or report by choice position and
parent question. These are descriptive counts from the already-used validation
manifest, not a new evaluation and not evidence of a statistically reliable
position shortcut.

## Compute and artifacts

- Reverse feature extraction: `268.70 s`, CPU `TFRT_CPU_0`, JAX `0.6.2`, peak
  working set `4,175,921,152` bytes. No GPU or cloud compute used.
- Fixed MiniLM/scorer evaluation and bootstrap: `10.65 s` total on the local
  RTX 3080 Laptop GPU (PyTorch `2.6.0+cu124`, peak allocated `129,222,656`
  bytes); no training. No cloud cost. Sealed audit access: none.
- External feature cache SHA-256:
  `adc7a97b6929684a092bd7cde27590cb65dacc46632f34afe26e56adb5a78f7b`;
  cache report SHA-256:
  `e8f5c3ca7af55c406f79b16335769c1ab891d75728d886af3f8e5300eff4f70b`.
- External prediction SHA-256:
  `1c973c49a60ded6340349efbe6cb7337faf2df35c843d291c39c4b4704fccde3`;
  result report SHA-256:
  `a27b4f78a575a50454a04474a965ea7243a0e857e9e0b7980ec5868fb183b52a`.
- Source scripts are `scripts/extract_frozen_videoprism_features.py` and
  `scripts/evaluate_clevrer_temporal_order_control.py`; the latter records
  artifact hashes, validation hashes, fixed readout hash and runtime.

Reproduction commands are in the report's experiment ledger/PR history; both
commands use the pinned local VideoPrism environment and create fresh output
directories. The forward baseline remains immutable.

```powershell
$env:PYTHONPATH = 'C:\CodexWorktrees\pretrained-reuse-omni\src;C:\CodexWorktrees\pretrained-reuse-omni'
& 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe' `
  -m scripts.extract_frozen_videoprism_features `
  --sample-dir 'C:\CodexArtifacts\pretrained-reuse-r2\clevrer-binary-controls-sample-v2' `
  --checkpoint 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz' `
  --source-root 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source' `
  --output-dir 'C:\CodexArtifacts\pretrained-reuse-r2\clevrer-temporal-reverse-features-v0' `
  --temporal-order reverse

py -3.11 -m scripts.evaluate_clevrer_temporal_order_control `
  --config 'C:\CodexWorktrees\pretrained-reuse-omni\configs\pretrained_reuse\path_a_videoprism_clevrer_binary_reasoning.yaml' `
  --sample-dir 'C:\CodexArtifacts\pretrained-reuse-r2\clevrer-binary-controls-sample-v2' `
  --forward-features 'C:\CodexArtifacts\pretrained-reuse-r2\clevrer-binary-controls-features-v0\video-features.npz' `
  --reverse-features 'C:\CodexArtifacts\pretrained-reuse-r2\clevrer-temporal-reverse-features-v0\video-features.npz' `
  --reference-results 'C:\CodexArtifacts\pretrained-reuse-r2\clevrer-binary-controls-results-v0' `
  --output-dir 'C:\CodexArtifacts\pretrained-reuse-r2\clevrer-temporal-order-results-v0' `
  --bootstrap-replicates 10000
```
