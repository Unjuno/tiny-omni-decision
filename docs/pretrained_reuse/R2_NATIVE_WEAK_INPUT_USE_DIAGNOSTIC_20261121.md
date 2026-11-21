# Frozen native weak-input diagnostic (2026-11-21)

## Result

This diagnostic used the selected, frozen Path A baseline checkpoint (step 1,792)
to test whether the readout responds to image observations, text-state
observations, and candidate-query features. It performed no fitting, checkpoint
selection, or calibration. Normal-input logits exactly reproduce the saved
baseline logits (`max_abs_delta=0`). The already exposed development validation
was used as a mechanism probe; this is not a blind quality result.

The image readout uses visual content: zeroing or replacing the image observation
with a same-task observation from another image materially reduced accuracy and
worsened NLL/Brier. The text readout uses candidate-query features, but its
decisions are nearly unchanged when the text-state observation is swapped with
another group from the same source/task. This points to weak state-to-candidate
binding in the text path, rather than a general failure to encode all text.

The frozen TRAIN sampler audit also found 93,924 unique question rows outside its
8,192 draws. There is substantial unused Text/Image/Audio coverage, but all 512
video questions and all 256 video groups in this corpus were consumed. The
balanced sampler repeated those video questions 1,536 times. Extending this same
sampler alone would add no new video groups.

## Integrity and protocol

- Baseline receipt SHA-256: `aaa1fe96a4be85a71ce146981c6b2e8b58e1308d91fc110e1ece6804b4d8048c`.
- Selected checkpoint SHA-256: `3eed6558731de60da82e75e38e599699f29663dd601f942505fec26a48b8dc1d` (step 1,792).
- Inference used the exact receipt-pinned `NativeCandidateReadout`, `FeatureStore`,
  and feature-position implementations. Their hashes and the used cache hashes
  are recorded in the external run receipt.
- Frozen input hashes: corpus receipt
  `47a6d2f3495810bac58f57dec1a962f8b490fa409c75671b46bb2999fde87d49`, validation
  JSONL `1f545a6e8d10b2712980f9ada58a0e5eef5b21021e4e01d6483965889a970326`, query
  receipt `7f5ffd7362e84770b864a189d50bc29f43be3738f54a07f6dbdc21f1ef7d909e`.
- Only Image (2,048 rows) and Text (10,732 rows) were scored. Sealed audit was
  not loaded. The corpus, checkpoints, and cached features were read-only.
- Conditions: original inputs; observation values zeroed while token type,
  position, and mask were retained; observation replaced by a different-group
  sample with the same source/task and feature kind; and all candidate-query
  vectors zeroed. Text observation replacement covered 10,730/10,732 rows. Two
  `text:original` rows had no different-group donor in their task and retained
  their original observation; both remain in the metrics and are listed in the
  donor manifest.
- Confidence intervals: 5,000 paired bootstrap draws, seed 17, whole-group
  percentile 95% intervals. The interval deltas below are row-weighted within
  each modality; they are separate from the existing hierarchical task/source
  macro metrics. ECE is descriptive and has no interval here.
- Local RTX 3080 Laptop inference took 112.53 s, with 148,203,008 bytes peak
  allocated CUDA memory and no training updates. Coverage audit took 21.07 s on
  CPU. Cloud cost was zero.

## Image result

Hierarchical macro metrics follow the same source/task aggregation as the fixed
baseline. The 40-choice first-option condition has 2.59% accuracy, close to its
2.5% uniform chance rate.

| Condition | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|
| Original | 0.20117 | 2.16512 | 0.86518 | 0.07223 |
| Image content zeroed | 0.11084 | 2.34147 | 0.90598 | 0.03872 |
| Image swapped within task | 0.10449 | 2.40528 | 0.91828 | 0.07446 |
| Candidate-query vectors zeroed | 0.02588 | 3.68888 | 0.97500 | 0.00557 |

| Intervention minus original | Accuracy delta (95% CI) | NLL delta (95% CI) | Brier delta (95% CI) |
|---|---:|---:|---:|
| Image content zeroed | −0.09033 [−0.10791, −0.07373] | +0.17635 [+0.15673, +0.19694] | +0.04081 [+0.03582, +0.04600] |
| Image swapped within task | −0.09668 [−0.11427, −0.07959] | +0.24016 [+0.21370, +0.26891] | +0.05310 [+0.04757, +0.05884] |
| Candidate-query vectors zeroed | −0.17529 [−0.19336, −0.15723] | +1.52376 [+1.50783, +1.53986] | +0.10982 [+0.10543, +0.11443] |

The image-content effect differs by task. Shape/count have the clearest NLL
degradation under image replacement (+0.308/+0.584); color is weak (+0.013),
and texture is intermediate (+0.055). The image path does receive useful visual
signal, but 20.1% validation accuracy remains far from the product objective.

## Text result

| Condition | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|
| Original | 0.18314 | 2.06565 | 0.84710 | 0.08888 |
| Text-state content zeroed | 0.18625 | 2.26307 | 0.84100 | 0.05337 |
| Text state swapped within task | 0.18401 | 2.06929 | 0.84704 | 0.08823 |
| Candidate-query vectors zeroed | 0.15998 | 2.52437 | 0.85314 | 0.04721 |

| Intervention minus original | Accuracy delta (95% CI) | NLL delta (95% CI) | Brier delta (95% CI) |
|---|---:|---:|---:|
| Text-state content zeroed | +0.00354 [−0.00224, +0.00934] | +0.13861 [+0.12201, +0.15464] | −0.00827 [−0.01177, −0.00498] |
| Text state swapped within task | +0.00112 [−0.00320, +0.00564] | +0.00410 [−0.00213, +0.01040] | −0.00010 [−0.00310, +0.00286] |
| Candidate-query vectors zeroed | −0.00578 [−0.01605, +0.00447] | +0.31845 [+0.28748, +0.34878] | −0.00236 [−0.00696, +0.00196] |

Zeroing candidate queries makes all candidate logits tie, so the scorer selects
option 0; this is a position-zero diagnostic, not random guessing. Its
source-level accuracy is 3.13% on 40-choice binding, 2.15% on 40-choice
condition-relevance, 42.36% on `text:original`, and 17.32% on
`text:programmatic_decision_v1`. The selected model exceeds that simple position
rule on the first, second, and fourth source, but not on `text:original` (37.32%
versus 42.36%). This makes option-position balance and task-specific input use
important data checks; it does not invalidate the existing label audit.

At source level, text-state shuffling changes NLL by only +0.0004 to +0.0121
across `text:curriculum`, `text:binding`, `text:condition_relevance_v1`,
`text:original`, and `text:programmatic_decision_v1`. In contrast, zeroing the
state increases macro NLL substantially for binding (+0.4568) and
condition-relevance (+0.4709), while changing Accuracy by only +1.27 and
−0.39 percentage points. The state features affect confidence in those tasks,
but the selected answer rarely tracks a different state. Candidate-query
features contribute more consistently.

## Frozen TRAIN coverage

The audit used the exact seed-17 sampler from 2,048 updates and hashed the
sampled question-ID order. It read only `train.jsonl`; the file SHA-256 matched
`d279ca44c6ff8e619529805be21675a8b4088fdcc476c7bb04649abdb5e60ebb` before and
after. Validation and sealed audit were not opened.

| Modality | TRAIN questions | Unique consumed | Unique unseen questions | Unseen observations | Unseen media assets | Unseen groups |
|---|---:|---:|---:|---:|---:|---:|
| Text | 18,432 | 2,048 | 16,384 | 1,422 | — | 1,422 |
| Image | 33,696 | 2,048 | 31,648 | 6,544 | 6,544 | 6,544 |
| Audio | 47,940 | 2,048 | 45,892 | 4,268 | 3,151 | 0 |
| Video | 512 | 512 | 0 | 0 | 0 | 0 |

Across all modalities, the sampler consumed 8,192 draws, 6,656 unique questions,
and repeated 1,536 questions. All repeats were Video: each of its 512 rows was
visited four times over 2,048 updates. Audio's 42 speaker/session groups were
all represented, although 3,151 unique media assets remained unconsumed. The
sampler's broad balance therefore hides substantial new Text/Image/Audio
coverage and an exhausted Video pool.

## Decision and next work

- Keep the native Image path; the intervention confirms it uses image content.
  Improve the weak color/texture tasks through train-only source/task diagnostics
  before changing the shared architecture.
- Prioritize text-state-to-candidate binding. Do not repeat the existing
  matched-batch experiment or the LSTM/capacity sweep. A next text fit needs a
  predeclared train-only group holdout or fold and must retain all text tasks and
  option counts; the exposed 24,346-row validation is descriptive only.
- Before extending the balanced common sampler, account for its exhausted Video
  pool. More steps on this exact corpus increase Video repetition, not unique
  group coverage. Use only already admitted, split-safe sources; this report
  does not admit a new Video source.
- Preserve all checkpoints and artifacts. No model was promoted; sealed audit,
  quantization, compression, and the 95–98% four-modality target remain open.

## Artifacts

- Input-use runner source commit: `bafe55ce85d0a1f8b5b1fe351d064391de72a238`;
  script SHA-256 `54341663d2d2c46546baf781d27cf8b9e112c40876625ec9c2fb98675b181028`.
- Input-use receipt:
  `C:\CodexArtifacts\pretrained-reuse-r2\native-weak-input-use-04\result.json`,
  SHA-256 `e511ee29f2bd61495eec24cc3c8b7bd756692501fc7aaed718feaed9f6d03746`.
- Per-example logits, donor manifests, task metrics, and input hashes remain in
  `C:\CodexArtifacts\pretrained-reuse-r2\native-weak-input-use-04\`.
- Coverage runner source commit: `6e82ca6dcf4d0bb1bd0dee72bdd28d18c41ffe39`;
  script SHA-256 `08a210d6abcee532a4a56467a344f09d50e38f8b03a9ae2c549203b71e78a765`.
- Coverage receipt:
  `C:\CodexArtifacts\pretrained-reuse-r2\native-common-train-coverage-audit-02\receipt.json`,
  SHA-256 `a8cd8f13b3411e4474e8a778eba9fa9cdf1f1edee0d32f94cdf7b26cffb6afa5`;
  `coverage.json` SHA-256 `c83305bf15ad7b69028e9c070425642051f669db2f6f2cdf61e6dd9fad399be0`.
- The first three input-use preflights failed before predictions were saved
  (missing deterministic CUDA workspace, exact token-length donor restriction,
  then two donorless text rows). Those output folders remain marked
  `FAILED_PREFLIGHT`; successful attempt 04 is separate and complete.
