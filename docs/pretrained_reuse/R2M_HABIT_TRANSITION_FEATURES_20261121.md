# HABIT causal video and state feature pilot — 2026-11-21

## Outcome

Extracted and packaged frozen VideoPrism-B features for every transition in
the previously frozen, non-audit HABIT pilot. This is a data/runtime plumbing
pass only. No model was fitted, no quality metrics were calculated, and the
sealed audit remained unopened.

## Fixed inputs and integrity

- HABIT source revision: `bae97d251907743f8fb6db7e70c61d361df4dc1e`, CC BY 4.0.
- Frozen pilot audit receipt SHA-256:
  `9b7a9d6c0efbc0be055df55277ea107d9be0cd6a45dcbcb5e0571d0257796672`.
- Seed-17 pilot selection: 66 train episodes / 24 validation episodes; the
  selection receipt is unchanged. All selected media and Parquet payloads were
  rechecked against the audited file hashes before feature extraction.
- Train: 202 transition examples over 36 task groups and 66 videos. Validation:
  76 examples over 6 task groups and 24 videos. Video media and task groups
  are disjoint across the two splits.
- The extraction uses only the selected train and development-validation
  media. It never accesses sealed-audit records, action columns, or future
  frames/states.

For each transition, the fixed eight video frames end at or before the
100 ms pre-onset cutoff. Each video-frame timestamp is paired with the latest
14-D robot state at or before that timestamp. The model inputs are therefore
`[8,768]` video features and `[8,14]` sensor values per example. Candidate
strings are taken from the pinned task metadata, deduplicated using the
existing exact-normalization rule, and checked against the frozen target index.
The generated question is the pinned high-level instruction followed by
“What should the robot do next?” No labels or future states are appended.

## Runtime and artifacts

The encoder is `google/videoprism-base-f16r288` at
`be719a406d563b66f0ac969e7c94bab8e997c81a`, loaded from preserved upstream
source commit `c6eba0ece3dd49a24ffc44582966997a4a59f1d8` and checkpoint SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
The run used Python 3.11.16, JAX 0.6.2, TensorFlow 2.18.1 and `TFRT_CPU_0`;
CUDA was disabled. PyAV 18.1.1 was used in input preparation and PyAV 18.1.0
in the encoder worker. The complete command took about 711 seconds, including
Parquet/video timestamp preparation. The worker process took about 652
seconds including model loading; its measured loop took 645.16 seconds, with
76.26 seconds in video decoding and 561.02 seconds in model forwards. It
reported 1.90 GB working set at exit; a true peak-memory high-water mark was
not recorded. No GPU, cloud compute, or training was used.

The frozen feature cache, aligned example metadata, receipts and exact script
copy remain outside Git under:

- `C:\CodexArtifacts\pretrained-reuse-r2\habit-transition-features-20261121-attempt-02`
- `C:\CodexArtifacts\pretrained-reuse-r2\habit-transition-decision-data-20261121-attempt-01`

The cache SHA-256 is
`654cd9001fde2329719840889381a9ffa3d5763f65386ba0d5fdba7fc1039076`; the
text-aligned example JSON SHA-256 is
`550c1725c547bef3784f8ef83f520aaf9baea4e23b19817cfe2cc631621d4292`. The
machine-readable source/runtime manifest is
[`habit-transition-features-20261121.json`](../../manifests/pretrained_reuse/habit-transition-features-20261121.json).

## Remaining gate

The 76 validation transitions come from only six held-out task groups, and
the source has about ten collectors from one institution. That is insufficient
for a broad human-aware robotics quality claim or the project's overall
95–98% target. This artifact only proves causal multimodal features and
candidate/target alignment can be constructed at bounded scale.

Before model fitting, the next useful step is a small, task-conditioned
candidate scorer using the frozen video and state sequences plus action text,
with text/options-only, state-only, video-only, paired, shuffled-video and
missing-stream controls. Keep selection on the same exposed development
validation and describe any result as a pilot. Obtain a larger group-disjoint
validation cohort before making quality or generalization claims. HubEgo access
remains gated pending the user's account approval.
