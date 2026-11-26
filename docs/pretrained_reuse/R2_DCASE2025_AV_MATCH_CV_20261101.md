# DCASE2025 paired AV recording match — nested room CV

## Status

This is a CPU-only development experiment on frozen official DCASE2025
dev-train audio/video features. It creates a 4-choice paired-input task: given
an audio segment, choose its matching video recording from four candidates in
the same room and event class. The exact positive is the query's parent
recording. Candidate order is balanced across positions. This tests recording
identity correspondence; it is not fine-grained timestamp synchronization,
general AV understanding, or product quality. The result does not close R2's
both-input-required gate.

The task uses only the existing frozen dev-train features. It does not read the
public dev-test annotations, sealed audit data, or new media. No prior
dev-test result was reopened. The audio-missing, video-missing, and uniform
controls return uniform probabilities. `video_shuffled` keeps the query fixed
and permutes the four video candidates within each query.

## Frozen task and provenance

- Source branch baseline: `docs/pretrained-reuse-roadmap-20261010` at
  `cb38e867b795caeb5df5aab0e9cd24cd04f9eb9a`. The task builder, model,
  evaluator, bootstrap, and tests were uncommitted working-tree files when
  this experiment ran; their individual hashes are recorded below.
- Experiment-code SHA-256: task builder
  `f7733e469c39fd13f580fff23bc4a7a3efb12355ea8ef52fa875a12391d8f5be`,
  readout `06e2201dfb334e57687b029a7cffc53f7848a5d9eb3edd8a179b54199291f256`,
  CV runner
  `5be54d380dffa9509db05b9daecaa4450b2990ce7f83f13a9a087ddd3a2973f4`,
  bootstrap
  `2b32b6be875c5ff2d097edd528b87e097faeca0caa0e6d87dd0a4bde8988f661`.
- Test-code SHA-256: task builder
  `9ad346cd272a78fd97188787dd4c12780c670a2d7314ea0e679a4f24419a49a8`,
  readout `e50401a2bc2ebb82192bbb9ceea550136bd1bae140dee4aa465faf0e9004e306`,
  CV `ea87ee1afbb1e6dd9c8ba08c3596a131901ff462bf7af209e70472efba5d7d54`,
  bootstrap
  `5b6a7bd1a8d39c0b5fb5d01de7bc285de3c7dbaf25d46711e403e88fda282f42`.
- Eligible support: 39 room/event cells with at least four distinct parent
  recordings. The frozen task contains 256 query units, 78 parent recordings,
  and 9 rooms. Every choice set has four unique parent recordings from the
  same room/event cell; the positive candidate is the exact source row for
  the query parent. Target indices are balanced 64/64/64/64.
- Source manifest SHA-256:
  `f6f1b690ea0853ee7f78fafed535a2a897f170bfefba108873f87b17c3be8b7f`.
- Source preflight SHA-256:
  `1be433e7f94a997e30b37caab89f74ee77477528c61437f61ce68d6917458524`.
- Query ID order SHA-256:
  `7a963e607efbf688e7581c5473aa9e1910f30553775eb01895034c087518d901`.
- Task manifest SHA-256:
  `017b5a915396f52e8e2a07b35dfcefa82071df158d0df0a6bb1b6af7d57a71ce`.
- Task effective-config SHA-256:
  `267c5ec69985efa1e0fa6c6a3bfb06ce563003aefb47b3a467b2a3be2b31ec42`.
- Audio feature array SHA-256:
  `5025bbc4f718827d1afd779fdacd9e77253f4d2f22416866e765ad9652ec9618`.
- Video feature array SHA-256:
  `5c04393fcef5f501ce77c10775000c0b88729b015e457927e32d2b110e7c20aa`.
- Frozen audio checkpoint SHA-256:
  `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`;
  source commit `a425fdce92572e602a1d5634799bd9f1f2efa806`.
- Frozen VideoPrism checkpoint SHA-256:
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`;
  source commit `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`.
- The audio feature uses the official 5-second FOA-to-stereo clip; the video
  feature uses 8 frames from a fixed 0.7-second window centered on the event
  midpoint. This aligns by parent recording, not exact sample-level time.
- Dataset source record: DCASE 2025 STARSS23, `doi:10.5281/zenodo.15559774`.
- Rights/source screening for this bounded DCASE/STARSS23 research source is
  documented in [the source audit](R2_STARSS23_SOURCE_20261011.md). These
  cached features do not grant rights to redistribute the source media.

Artifacts are outside Git under
`C:\CodexArtifacts\pretrained-reuse-r2\`:

- Task: `dcase2025-av-matching-task-20261101-attempt-01`.
- CV result: `dcase2025-av-match-cv-20261101-attempt-01`.
- CV effective-config SHA-256:
  `e0ca08aaf5fbb4fa21a9ff68d81ac04843313669d56c5b75148ecec9c644ba2e`.
- Result JSON SHA-256:
  `d0196ef0b54202cd9234747f5d0a4a6084bf782b6f649538e09d10c09492e686`.
- OOF predictions SHA-256:
  `f49cec00e59ce58749a69fb1dba0ea0a1e5dee9e131a738e24d0f3eacea9691c`.
- Paired bootstrap JSON SHA-256:
  `6ed285538e508e27918d3ae460ac5c715368f8d6f6b4c12a41554737618820b6`.

Executed CPU commands (paths are pinned to the frozen local feature/task
artifacts):

```powershell
python scripts/build_dcase2025_av_matching_task.py --features C:\CodexArtifacts\pretrained-reuse-r2\dcase2025-av-features-20261031-attempt-01 --output C:\CodexArtifacts\pretrained-reuse-r2\dcase2025-av-matching-task-20261101-attempt-01 --seed 17
python scripts/run_dcase2025_av_match_cv.py --features C:\CodexArtifacts\pretrained-reuse-r2\dcase2025-av-features-20261031-attempt-01 --task C:\CodexArtifacts\pretrained-reuse-r2\dcase2025-av-matching-task-20261101-attempt-01 --output C:\CodexArtifacts\pretrained-reuse-r2\dcase2025-av-match-cv-20261101-attempt-01 --seed 17 --rank 8 --max-epochs 30 --patience 5
python scripts/bootstrap_dcase2025_av_match.py --run-dir C:\CodexArtifacts\pretrained-reuse-r2\dcase2025-av-match-cv-20261101-attempt-01 --replicates 10000 --seed 17
```

## Method

The readout has two bias-free 768-to-8 linear projections (12,288 trainable
parameters total), L2-normalizes both projected embeddings, and scores each
candidate by a dot product with the projected query audio. No pretrained
encoder is fine-tuned. Training is nested leave-one-room-out: each outer room
is scored once, while a separate inner held-out room selects the checkpoint by
validation NLL (minimum improvement 0.0001, patience 5, maximum 30 epochs).
Each fold uses seed 17, AdamW (learning rate 0.003, weight decay 0.0001),
batch size 32, and CPU. Feature standardization is fit only on training
examples in each fold. Outer test-room predictions are never used for
checkpoint selection.

After fitting, a paired cluster bootstrap was run on the saved out-of-fold
predictions only: 10,000 replicates, seed 17, separately resampling the 78
parent recordings and 9 rooms. Intervals are percentile 95% intervals. No
training, model reload, dev-test access, or audit access occurred in this
post-hoc step.

## Results

Four-choice chance is 25%. Metrics use all 256 out-of-fold predictions.

| Arm | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Joint audio + video | 26.56% | 1.4104 | 0.7615 | 0.0744 | 0.3264 |
| Audio missing / uniform | 25.00% | 1.3863 | 0.7500 | 0.0000 | 0.2500 |
| Video missing / uniform | 25.00% | 1.3863 | 0.7500 | 0.0000 | 0.2500 |
| Uniform | 25.00% | 1.3863 | 0.7500 | 0.0000 | 0.2500 |
| Video candidates shuffled | 23.44% | 1.4201 | 0.7667 | 0.0932 | 0.3264 |

Joint-minus-uniform point deltas were +1.56 percentage points in Accuracy,
**+0.0241 NLL** and **+0.0115 Brier** (positive NLL/Brier deltas are worse).
Against video-shuffled, the joint arm was +3.13 points in Accuracy, -0.0098
NLL, and -0.0052 Brier. All paired bootstrap intervals included zero:

| Grouping | Contrast | Accuracy delta 95% CI | NLL delta 95% CI | Brier delta 95% CI |
|---|---|---:|---:|---:|
| Parent (78) | Joint − uniform | [−5.95, +9.41] pp | [−0.0065, +0.0549] | [−0.0041, +0.0271] |
| Parent (78) | Joint − video-shuffled | [−6.64, +12.98] pp | [−0.0645, +0.0447] | [−0.0323, +0.0218] |
| Room (9) | Joint − uniform | [−4.12, +7.25] pp | [+0.0031, +0.0430] | [+0.0008, +0.0218] |
| Room (9) | Joint − video-shuffled | [−4.55, +9.12] pp | [−0.0644, +0.0388] | [−0.0300, +0.0181] |

The room bootstrap indicates worse NLL and Brier for joint versus uniform in
this sample; only nine room clusters make that interval fragile. Inner-selected
epochs by outer room were 2, 2, 4, 2, 1, 1, 1, 6, and 22. Train-validation
NLL gaps were positive (about 0.12–0.64), consistent with rapid overfit and
high fold variability.

## Interpretation and next gate

This fixed task and readout did not establish reliable benefit from combining
audio and video. Accuracy is near chance, proper scoring rules are worse than
uniform for the joint arm, and all paired intervals include zero. The shuffled
point comparison is directionally encouraging but too uncertain to support a
fusion claim. The result is **weak/negative development evidence**, not an
architecture ceiling and not a blind audit.

Keep R2-M open. Do not tune this readout or task against public dev-test labels,
and do not promote it or begin compression from this result. A follow-up would
need an independently defined task with exact temporal correspondence,
recording/room-disjoint train/validation/test, enough independent groups, and
strong unimodal controls before any fusion-capacity experiment is justified.

The CV fit completed on CPU in 3.43 seconds; GPU time and cloud cost were $0.
Frozen feature extraction predates this CV and is not included in that timing.
No model weights, media, or large checkpoints were added to Git. The source
dev-test evaluation recorded in
[`R2_DCASE2025_AV_READOUT_20261101.md`](R2_DCASE2025_AV_READOUT_20261101.md)
was not reopened.
