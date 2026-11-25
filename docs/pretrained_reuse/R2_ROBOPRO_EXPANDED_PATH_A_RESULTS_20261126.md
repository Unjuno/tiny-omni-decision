# RoboPro expanded frozen-feature Path A results — 2026-11-26

## Scope and conclusion

This fixed-seed follow-up addresses the small 16-prompt-group support in the
[initial RoboPro Path A readout](R2_ROBOPRO_PATH_A_ACTION_READOUT_20261126.md).
It uses the same pinned pretrained encoders, label contract, query timing,
readout family, optimizer, and early-stopping rule, but freezes 64 new exact
prompt groups per partition. This is still project-created development data
from publisher TRAIN, not a blind test.

**Result:** on this task, robot state is the strongest signal. The video-only
GRU is above four-choice chance and beats mean-pooled video, and video ablation
controls show that the selected joint model uses temporally aligned video.
However, both `State+Video-GRU` and `Text+State+Video-GRU` perform below
`State-only` on the fixed development set. This does not establish a limit for
video representations or multimodal fusion generally; the target is agreement
with one recorded demonstrator action, not task success, safety, or a unique
correct action.

## Frozen protocol and lineage

- Source: `Hoshipu/RoboPro`, revision
  `90ec789bf4018eb9c0f75da9f69aab5c185f0fd0`, publisher-declared CC BY 4.0.
- Sample manifest SHA-256:
  `6c78afea26b2c1862b3b6a75b10cf761df2a723456717afd3e1da54369a4039a`.
- Main effective config:
  `configs/pretrained_reuse/robopro_action_probe_expanded_v1.json`, SHA-256
  `78efc26f4dc078e84111234f738d71560dc4d15462f474c88d0bf7c801647805`.
- Sample: seed 17; 64 train and 64 development exact prompt groups; four
  episodes per prompt; 256 episodes per partition; 768 queries per partition
  at three causal progress points (25%, 50%, 75%).
- Old 32 prompt groups and all old sample/smoke episodes were excluded. Train
  and development have zero exact-prompt overlap, zero file-hash overlap, and
  zero normalized 32×32 frame overlap. Physical scene/base-task identity is not
  available, so scene-level independence is unknown.
- Data receipt SHA-256:
  `dc7241dafa14aff8c3716a58c1aa5d7fb6715cd1c49b28c35708023785e72a61`; 1,024
  files / 596,960,654 bytes. All receipt hashes verified locally.
- Parquet sidecar SHA-256:
  `7f3945660614a74cc536292431b2518e872d10b2b006244e4500de3801745056`;
  126,586 finite state/action rows.
- Full data audit: 512 episodes, 384 unique candidate-group queries, zero
  duplicate candidate queries, zero cross-partition media/content overlap.
- VideoPrism-B frozen encoder: 114,365,184 parameters;
  revision `be719a406d563b66f0ac969e7c94bab8e997c81a`; checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
  One causal eight-frame window spans at most two seconds and ends at each
  query. MiniLM-L3 is frozen at 17,389,824 parameters, revision
  `4ca70771034acceecb2e72475f72050fcdde4ddc`; checkpoint SHA-256
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- Model-selection rule: per-arm minimum development macro NLL, max 80 epochs,
  patience 12, `min_delta=1e-4`. Metrics are macro-averaged over exact prompts.
  Paired intervals use 10,000 prompt-group bootstrap replicates, seed 17.

## Main 8-arm comparison

All values below are prompt-group macro metrics on the 64 development prompts.
The checkpoints were selected on this same development set, so these results
are selection-exposed. `Total parameters` includes the frozen pretrained tower
and trainable readout; package bytes and target-device latency were not measured.

| Arm | Best epoch | Trainable params | Total params | Accuracy | NLL | Brier | ECE-15 | Train → dev Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | 13 | 50,753 | 17,440,577 | 25.00% | 1.3864 | 0.7500 | 0.0036 | 25.00% → 25.00% |
| State | 44 | 33,345 | 33,345 | **75.65%** | **0.5478** | **0.3160** | 0.1715 | 82.03% → 75.65% |
| Video mean | 8 | 75,329 | 114,440,513 | 36.46% | 1.3046 | 0.7041 | 0.2319 | 50.00% → 36.46% |
| Video GRU | 6 | 186,113 | 114,551,297 | 41.54% | 1.2059 | 0.6638 | 0.2157 | 49.35% → 41.54% |
| Text + State | 20 | 66,305 | 17,456,129 | 65.63% | 0.6724 | 0.4028 | 0.1891 | 72.66% → 65.63% |
| Text + Video mean | 4 | 108,289 | 131,863,297 | 33.07% | 1.3215 | 0.7171 | 0.1826 | 38.28% → 33.07% |
| Text + State + Video mean | 15 | 123,841 | 131,878,849 | 61.59% | 0.7700 | 0.4417 | 0.1848 | 72.66% → 61.59% |
| Text + State + Video GRU | 13 | 234,625 | 131,989,633 | 61.72% | 0.8017 | 0.4551 | 0.2117 | 73.96% → 61.72% |

The selected full joint model is 13.93 percentage points below State-only in
macro Accuracy (paired 95% interval `[-17.84, -10.16]` points). Video GRU beats
mean-pooled video by 5.08 points (`[2.21, 7.94]`). The full joint model and
mean-video joint model are nearly tied (`+0.13` points; `[-4.56, 4.56]`).
These bootstrap intervals are descriptive: each arm's checkpoint used the same
development set for selection, and prompt groups do not prove scene independence.

## Focused post-hoc State + Video ablation

After inspecting the fixed 8-arm result, one missing modality combination was
added to test whether text caused the video fusion weakness. This is explicitly
a post-hoc diagnostic on the already exposed 64-group development split, not an
independent confirmation or a preregistered comparison. It used the same frozen
features, seed, readout, selector, and training settings. Config SHA-256:
`311da5ec204c2d82a222c74a9c5856729c7d28583041a453d7f414bb672062b7`.

| Arm | Best epoch | Trainable params | Total params | Accuracy | NLL | Brier | ECE-15 | Train → dev Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| State + Video GRU | 17 | 201,665 | 114,566,849 | 64.58% | 0.7425 | 0.4230 | 0.1987 | 78.52% → 64.58% |

This is 11.07 points below State-only (paired interval `[-13.67, -8.33]`) and
not clearly different from Text+State+Video-GRU (+2.86 points,
`[-0.39, 6.25]`). Adding the temporal video representation did not overcome the
state-only baseline in either tested fusion.

## Video sensitivity controls

On the fixed selected `Text+State+Video-GRU` checkpoint, the default prediction
was 61.72% Accuracy / 0.8017 NLL. Zeroing the video sequence gave 55.86% /
0.8609; permuting video windows within prompt and progress gave 55.73% / 0.9207.
The default-vs-zero-video Accuracy difference was 5.86 points (paired 95%
interval `[2.34, 9.38]`); default-vs-misaligned-video was 5.99 points
(`[2.47, 9.64]`). Zeroing text had little effect (60.94% / 0.8344; paired
interval for default minus zero-text `[-1.17, 2.73]`). Zeroing state reduced
Accuracy to 26.56% and raised NLL to 3.3896. Thus the fitted joint readout uses
both aligned video and state, but the video contribution is not enough to
improve overall selection quality over state-only.

## Initial 16-group comparison and interpretation limits

The earlier development set reported State-only at 55.73% / 0.8461 NLL,
Text+State at 58.33% / 0.8356, Video-GRU at 31.25% / 1.3562, and full
Text+State+Video-GRU at 51.56% / 0.9089. These are different exact prompt
groups from the expanded run; do not interpret the changed scores as a paired
learning improvement. The larger 64-group evaluation changes the ranking:
State-only is best; text alone is at four-choice chance; and adding text or
video to state decreases development accuracy for the tested readouts.

The training-to-development gaps (e.g. full joint Accuracy 73.96% vs 61.72%,
NLL 0.5705 vs 0.8017) are evidence of some overfit to the finite prompt groups.
They are not a formal estimate of unseen-scene generalization. The success
target itself is weak: a recorded demonstrator action is one observed behavior,
not proof that other candidates are wrong or that the selected action succeeds.

## Runtime, artifacts, and tests

- Device: CPU only; PyTorch 2.6.0+cu124, one torch thread for readout fitting.
  No GPU or cloud compute was used. Cloud cost: 0.
- Data acquisition: 596,960,654 bytes. Full 512-video data audit took about
  12 minutes. Text feature extraction: 7.03 seconds. Corrected VideoPrism
  feature extraction: 3,072.30 seconds (51.2 minutes) for 1,536 windows;
  forward time 2,798.90 seconds. Feature-cache SHA-256
  `6750414ed0f173b42a936c0e883478bc7070a66e85ca38fb8c53192785af8d81`.
- The initial extractor had a hard-coded 128-episode shape guard. The first
  expanded extraction spent about 53 minutes computing features, then failed
  before cache save on that guard. The second, corrected extraction completed
  in 51.2 minutes. The guard now validates dimensions from the selected sample;
  regression tests cover 512 episodes and mismatches. The fitter was also
  generalized from 16/32 groups and 192 queries to manifest-derived counts.
- Main readout fit: 38.59 seconds; peak working set 736,829,440 bytes. The
  post-hoc 9-arm fit: 45.54 seconds; peak working set 747,655,168 bytes.
- Main run report SHA-256
  `c45e14d7eb079ce192cfeaf8bb261db35cfe4a3c265e648ad489c6df6a90ea08`;
  predictions SHA-256
  `3caa5f05710d12fc854ade9697e0918b790a2978de05ece919314a67e06556ec`;
  history SHA-256
  `f1f8895de405c94ce1e7d5740122006fbe0edf45d5261239856878f1d714a82e`.
- Post-hoc report SHA-256
  `090d455e45394adf60fe143cc6ea80113646dc9b85fbb81ce24ca67458223da3`;
  predictions SHA-256
  `815825790c52b52a6ab8da60131cf0875a7abd131485b116c8cb0c095c680477`.
- Weights and media remain outside Git. Frozen source references, configs,
  sample manifest, tests, and results are in Git. The main branch and existing
  v1/RoboPro attempts were not modified.
- Targeted checks after the focused arm: 25 RoboPro action/readout/split/download
  tests pass; Ruff passes. CI for commit `068b8365` passed. The latest pushed
  commit's Actions status must be refreshed after this report is pushed.

## Recommendation and open gates

Do not promote or merge this readout as a general Decision model. Do not start
a broad BERT sweep: this task's text feature arm is at chance and text fusion
reduced the state baseline; previous Japanese BERT results remain
task-specific, while the matched BERT-family common multimodal fit did not
improve its baseline. Do not add another encoder or temporal head on this
dataset alone.

The next useful evidence is better task labels and a source with scene/task
group IDs or an official publisher split: evaluate action success/goal
completion (and multiple valid actions), then compare fixed state-only,
state+video, and video-only readouts on a newly frozen scene-disjoint split.
Keep the 100–200M pre-quantization parameter aim provisional; package size,
quantized runtime, target-device latency, general text/image/audio/video task
quality, and R0–R8 admission gates remain open.
