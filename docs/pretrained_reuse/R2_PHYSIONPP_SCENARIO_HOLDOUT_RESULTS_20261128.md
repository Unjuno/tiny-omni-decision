# Physion++ leave-one-scenario-out Path A result — 2026-11-28

## Finding

The fixed Path A readout showed useful video sensitivity on the exposed
random-group development validation, but its transfer to an entirely held-out
simulation scenario was weak and poorly calibrated. The frozen scenario
holdout protocol completed without changing the 8-frame feature cache, data,
architecture, optimizer or epoch budget. This is a train-only exploratory
diagnostic, not a blind/independent evaluation or a product-quality result.

## Design and data integrity

Ten folds each held out one complete scenario; all 652 TRAIN rows and 492
scenario+scene-seed groups were held out exactly once. Per-fold group overlap
was zero. Each model trained on the other nine scenarios for 200 epochs at
seed 17, then the final fixed-budget checkpoint was evaluated once on its held
scenario. There was no validation-based checkpoint selection. Comparators were
the same fitted model with video features zeroed, a deterministic within-fold
video permutation, and each fold's training-label prior.

This diagnostic follows the fixed protocol in
[`R2_PHYSIONPP_SCENARIO_HOLDOUT_PROTOCOL_20261128.md`](R2_PHYSIONPP_SCENARIO_HOLDOUT_PROTOCOL_20261128.md).
It uses the same 652 training records already used by the full-train Path A
fit. Although each fold model held out its test scenario, the overall analysis
is post hoc on known training data; it is not independent evidence.

## Aggregate out-of-scenario metrics

| OOF condition | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Video Path A, final fixed epoch 200 | 0.5567 | 2.3797 | 0.7925 | 0.3696 | 0.9106 |
| Same fitted readout, zero video | 0.5199 | 1.7637 | 0.8018 | 0.3784 | 0.8984 |
| Fold training-label prior | 0.5199 | 0.6967 | 0.5035 | 0.0406 | 0.5204 |
| Same fitted readout, permuted video | 0.5322 | 2.4784 | 0.8118 | 0.3805 | 0.9106 |

Equal-weight macro Accuracy over the ten scenarios was **0.5526**, with a
minimum scenario Accuracy of **0.4600**. The per-scenario Accuracy range was
46.0%–66.7%. Several scenarios matched or fell below the zero-video accuracy;
none reached the Path A aspiration.

A paired bootstrap over the 492 scene-seed asset groups (10,000 replicates,
PCG64 seed `20261128`) compared video Path A with zero video. Accuracy delta
was +3.68 percentage points (95% descriptive percentile interval −1.58 to
+8.98 pp), NLL delta was +0.615 (interval +0.037 to +0.993; higher is worse),
and Brier delta was −0.009 (interval −0.099 to +0.082). The accuracy and
Brier intervals include zero; NLL is worse. The training-label prior was
substantially better calibrated than either fitted condition.

| Held-out scenario | Rows | Labels No / Yes | Video Accuracy | Video NLL | Zero-video Accuracy |
|---|---:|---:|---:|---:|---:|
| `bouncy_platform_pp` | 87 | 55 / 32 | 63.2% | 3.031 | 63.2% |
| `bouncy_wall_pp` | 89 | 43 / 46 | 55.1% | 3.282 | 48.3% |
| `deform_clothhang_pp` | 84 | 45 / 39 | 51.2% | 3.275 | 53.6% |
| `deform_clothhit_pp` | 84 | 32 / 52 | 60.7% | 0.796 | 38.1% |
| `friction_cloth_pp` | 56 | 28 / 28 | 51.8% | 1.735 | 50.0% |
| `friction_collision_pp` | 50 | 28 / 22 | 46.0% | 2.069 | 56.0% |
| `friction_platform_pp` | 51 | 34 / 17 | 66.7% | 1.674 | 66.7% |
| `mass_collision_pp` | 53 | 21 / 32 | 49.1% | 1.497 | 39.6% |
| `mass_dominoes_pp` | 46 | 29 / 17 | 60.9% | 3.510 | 63.0% |
| `mass_waterpush_pp` | 52 | 24 / 28 | 48.1% | 2.440 | 46.2% |

## Data-first interpretation

This result does not establish that the VideoPrism representation lacks
physical information. It does show that the present 652-example task and
training recipe do not produce robust, calibrated transfer to held-out
scenarios. Scenario-specific label/factor balance, sparse strata, inadequate
temporal coverage, visual ambiguity, and the 200-epoch unselected endpoint may
all contribute. The different full-train development run selected epoch 55 on
random-group validation, while this diagnostic used nine-scenario training
and the fixed final epoch 200; their difference cannot be attributed to a
single cause.

Data quality remains central. The official metadata-to-label mapping and
train/dev media grouping checks passed, but human visual observability has not
yet been rated. The prior 8-frame sample includes an exact labeled positive
contact frame in 63.6% of positive train clips. A newly generated 75-pair
label-blinded review packet is ready; its response file is blank and its
separate answer key must remain unopened until answers are frozen.

Do not launch BERT or shared-Lens work to address this scenario-transfer
result. The task's language is fixed, and the data issue concerns temporal
coverage, scenario diversity, label observability and calibration. Preserve
this result; do not change the Path A baseline. The next useful step is to
collect the blinded human review, then design a separately versioned,
data-first scenario-generalization experiment. No frame-count, data or model
change is justified by this diagnostic alone.

## Execution and artifacts

- Source commit: `2412074217eeba2c13d987893fb57004e709d876`.
- Run: 2026-11-28 06:52:31–06:53:53 UTC; 76.51 seconds accumulated fold
  training time; 98,817 trainable parameters per fold; CPU only, no cloud/GPU.
- Successful command:

  ```powershell
  $env:CUDA_VISIBLE_DEVICES='-1'; python -m scripts.evaluate_physionpp_scenario_holdout_path_a `
    --train C:\CodexArtifacts\tqpp\fullclip-v1\train.jsonl `
    --train-feature-cache C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-train-videoprism-features-20261128-attempt-02\video-features.npz `
    --train-feature-report C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-train-videoprism-features-20261128-attempt-02\run-report.json `
    --config configs\pretrained_reuse\path_a_video_videoprism_physionpp_contact_seed17.yaml `
    --output C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-path-a-scenario-holdout-20261128-attempt-01
  ```

  A first direct-file invocation exited before creating any output because
  Python could not import the repository's `scripts` package. Running the same
  command as a module fixed that import-path issue; no partial training run was
  resumed.
- Train manifest SHA-256:
  `0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b`.
- Frozen train feature cache SHA-256:
  `32846b085df7642d9190df8f553a58496d275f8580efe6c6a8adfa3a83217360`.
- Config SHA-256:
  `57f8886c6637ff691f00b0a57663e935e9870eaa02ea30cbc25f1a393cece3ad`.
- Run report SHA-256:
  `5a7ad20569382052fd2739a9e33b7672bad82d57c2736c0dfb40246c4342251`.
- OOF predictions SHA-256:
  `582d56f156e70eb28b1f7b4a59c3f0152f79be499c6525daed8d94f31b66c7aa`.
- Fold training-loss history SHA-256:
  `5f70de0e8555c99815067812a36068b11bdcc324721aff82c6b7998691166a65`.
- Local output:
  `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-path-a-scenario-holdout-20261128-attempt-01`.
- No development validation, publisher test, sealed audit, media, or existing
  checkpoint was modified. Ten small fold checkpoints remain only in the
  local output directory and were not added to Git.
