# CLEVRER unique-scene coverage v2 results

## Outcome

The fixed-budget, seed-17 comparison completed on the local RTX 3080 in
`C:/CodexArtifacts/pretrained-reuse-r2/clevrer-coverage-512train-256val-v2-attempt-05`.
Both arms received 2,048 optimizer updates (16,384 question presentations),
with no encoder updates. The only fitted parameters were the existing shared
candidate readout. Best and final checkpoints are separate, and reloading each
best checkpoint reproduced its saved validation predictions exactly.

The expanded 768-scene training pool improved the selected macro validation
metrics over the 256-scene pool on the same 512 ordered validation questions.
The paired scene-cluster bootstrap supports a positive pooled effect, but the
per-question-type results show that it is driven by static descriptive items;
the temporal result is unresolved. This is development validation used for
checkpoint selection, not blind or sealed evaluation.

## Frozen protocol and inputs

- Source code starting commit: `3b5803638110e4942cbe5b640f0bf04aef4240f1`.
- Experiment: `clevrer-videoprism-unique-scene-coverage-v2`, seed 17.
- Effective config SHA-256:
  `d00ebd067c02d14442206737cbc11319b088d28cfc21aa3400bd1cac33997639`.
- CLEVRER source revision: `98b842082ba4f7c18b6b9e3f39145871782a65ef`.
- VideoPrism 8-frame frozen cache and MiniLM-L3 frozen text encoder; only the
  shared CandidateScorer was trained. Both arms used the same initialization,
  optimizer, LR, batch size, 2,048-update schedule, and ordered validation IDs.
- Validation ID order SHA-256:
  `9f3868d727568904ca55dac48aff6bc3c0dd656853dce1c87bd08f303f396804`.
- Baseline train JSONL SHA-256:
  `5cd4e5155b0161e35b10aad2c98e14afba7d46407e635bd1c08d786e1e51765d`.
- Expanded train JSONL SHA-256:
  `2691c4494141c9685805977fdb42690269952061f7924cb90ccac3d7580ee62b`.
- Validation JSONL SHA-256:
  `322859a0e8885e228daff1f463aa5a2f14bf8d873aa54f8bea972926c6a58d34`.
- Feature cache SHA-256: baseline
  `40eaff323e3654bf34653950d5fea4cd532f1a0e096438ec9cb73703dc589ea1`,
  expanded `bbc803fedb5f4f901fa9800ff1e3a7fa5d1557bbca5eec4f15d12ad4e83d9e27`.
- The cache covered 1,024 unique train/validation media assets with no missing
  features or feature conflicts. Across baseline train, added train, old
  validation, and new validation, scene and media identity overlaps were zero.
- The existing train pool had 256 scenes /512 questions; the expanded pool had
  768 scenes /1,536 questions. Validation had 256 fresh scenes /512 questions,
  balanced 256 static and 256 temporal descriptive items.
- Normalized CLEVRER question templates repeat across scenes. The runner
  counted 156 exact normalized question-plus-options pairs shared between
  train and validation. The feature report separately counted 134 normalized
  question-content templates shared with validation (324 validation rows).
  This was explicitly permitted and counted, but means validation is not
  question-template independent.
- No official test or sealed-audit data was loaded.

The sampler consumed 16,384 presentations in each arm. The 256-scene arm
consumed 512 unique questions /256 scenes and repeated 15,872 presentations;
the 768-scene arm consumed 1,536 unique questions /768 scenes and repeated
14,848 presentations. At selected checkpoints, the sampler had consumed 5,120
presentations (baseline, step 640) and 13,312 (expanded, step 1,664),
respectively. These are distinct from full-run consumption.

## Selected checkpoint metrics

The selector minimized unweighted macro NLL across static and temporal
descriptive validation items, with earliest tie and `min_delta=1e-5`.
Values below are selected checkpoints, not the final update.

| Arm / step | Question type | n | Accuracy | NLL | Brier | ECE (15 bins) |
|---|---|---:|---:|---:|---:|---:|
| 256-scene / 640 | Macro | 512 | 0.3887 | 1.1830 | 0.6598 | 0.1055 |
| 256-scene / 640 | Static descriptive | 256 | 0.4180 | 1.0738 | 0.6246 | 0.1081 |
| 256-scene / 640 | Temporal descriptive | 256 | 0.3594 | 1.2922 | 0.6949 | 0.1029 |
| 768-scene / 1,664 | Macro | 512 | 0.4590 | 1.1322 | 0.6222 | 0.0720 |
| 768-scene / 1,664 | Static descriptive | 256 | 0.5352 | 0.9593 | 0.5447 | 0.0616 |
| 768-scene / 1,664 | Temporal descriptive | 256 | 0.3828 | 1.3051 | 0.6997 | 0.0825 |

Paired bootstrap resampled 256 video scenes with replacement 10,000 times,
seed 17, percentile 95% intervals. Differences are expanded minus baseline.

| Subset / metric | Difference | 95% interval |
|---|---:|---:|
| Macro Accuracy | +0.0703 | [+0.0195, +0.1211] |
| Macro NLL | -0.0508 | [-0.0909, -0.0112] |
| Macro Brier | -0.0376 | [-0.0618, -0.0137] |
| Macro ECE | -0.0317 | [-0.0716, +0.0148] |
| Static Accuracy | +0.1172 | [+0.0469, +0.1875] |
| Static NLL | -0.1145 | [-0.1707, -0.0588] |
| Static Brier | -0.0799 | [-0.1184, -0.0412] |
| Temporal Accuracy | +0.0234 | [-0.0508, +0.0938] |
| Temporal NLL | +0.0129 | [-0.0492, +0.0735] |
| Temporal Brier | +0.0048 | [-0.0299, +0.0382] |

The pooled improvement is supported within this comparison, chiefly by static
descriptive questions. Temporal descriptive Accuracy, NLL, and Brier intervals
include zero; the experiment does not establish a temporal-reasoning gain.
Macro ECE also has an interval spanning zero.

## Learning, overfit, and runtime

Both runs completed all 2,048 updates. Validation was evaluated every 128
updates and the selected steps were 640 and 1,664. Baseline fixed-train versus
validation macro Accuracy was 0.5742 versus 0.3887 at selection and 0.8809
versus 0.4238 at the final step; its final validation NLL degraded to 1.7720
from the selected 1.1830. This is a clear late overfit signal for that arm.
Expanded fixed-train versus validation macro Accuracy was 0.5710 versus 0.4590
at selection and 0.5866 versus 0.4434 at the final step; final validation NLL
was 1.1590 versus selected 1.1322. Fixed-train and validation sets differ, so
these are descriptive gaps, not a paired generalization estimate. Rolling train
loss is not treated as a fixed-set gap.

| Arm | Training seconds | Evaluation seconds | Peak CUDA allocated |
|---|---:|---:|---:|
| 256-scene | 6.67 | 0.67 | 45,804,032 bytes |
| 768-scene | 7.20 | 1.28 | 90,024,448 bytes |

The host was a local NVIDIA GeForce RTX 3080, Windows WDDM. GPU memory was also
occupied by Orbitune during the runs; the process yielded enough capacity for
both runs to complete. Peak allocated memory above is the readout process's
PyTorch allocation, not total device usage. Cloud compute and cost: $0.
Runtime versions: Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4,
Transformers 5.6.2, NumPy 2.4.6. Result JSON SHA-256:
`138222f8f9c8c25b19ef3387fe93cff9b41f666afeda3f8f08e5b6bdb8324209`.

## Failed attempts and retained artifacts

- The initial acquisition attempt was rejected before fitting due to
  cross-split scene reuse; 409 partially downloaded clips were preserved and
  excluded. No model fit used those files.
- The first training attempt stopped at the first validation selector because
  the V2 config omitted `min_delta`. The selector's existing `1e-5` value was
  made explicit; legacy V1 config/defaults were not changed.
- A subsequent invocation exposed two command-path mistakes before training;
  those attempt directories/logs remain preserved.
- Attempt 04 completed both arms and saved their histories/checkpoints, but its
  final bootstrap failed because CLEVRER items have variable choice counts. No
  result was claimed from that attempt. The bootstrap metric implementation
  now computes multiclass Brier/ECE per example and supports varying choice
  counts. Attempt 05 is the complete run reported here.
- Attempt 05 result and artifact hashes are recorded in
  `C:/CodexArtifacts/pretrained-reuse-r2/clevrer-coverage-512train-256val-v2-attempt-05/artifact-hashes.json`.
  It includes both best/final checkpoints, every evaluation history, selected
  and final per-example predictions, preflight, and the complete result.
- No pre-existing V1/V2 corpus, cache, model, or checkpoint was overwritten.

## Conclusion

Increasing unique CLEVRER scene coverage from 256 to 768 improved the selected
pooled development metrics at the same 2,048-update budget for this frozen
VideoPrism + MiniLM readout. The effect is concentrated in static descriptive
questions; temporal performance remains unresolved, and question-template
overlap limits the independence of the validation text. This supports testing
data coverage as one useful factor, not claiming it is the only bottleneck or
that the model has broad video reasoning capability. Keep this as a research
result; no product promotion, sealed-audit evaluation, or compression step is
authorized by these metrics.
