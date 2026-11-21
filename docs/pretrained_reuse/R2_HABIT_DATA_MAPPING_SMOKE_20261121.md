# HABIT target-mapping and VideoPrism smoke — 2026-11-21

This is a CPU-only plumbing check on one seed-fixed train episode (0) and one
derived development-validation episode (1802). It is not a quality result; no
model was fitted, and the sealed audit was not opened.

## Inputs and integrity

- HABIT revision: `bae97d251907743f8fb6db7e70c61d361df4dc1e`
- Video: `front_view/episode_000000.mp4`, 5,125,669 bytes, SHA-256
  `5ed917a199048b458d0d12d2a955ae397b273dfecc76d5b94d3d815069b12bd2`.
- Video: `front_view/episode_001802.mp4`, 4,143,516 bytes, SHA-256
  `5f2c242d63965b059bec78804acd3126da8f0a28d14d73a42aad937d27e681ba`.
- Full Parquet train episode 0: 1,877,629 bytes, SHA-256
  `5103ad7b6425a26687f6bfa0824830615793ae4617c5762dec0b83523eccff41`.
- Full Parquet validation episode 1802: 1,363,098 bytes, SHA-256
  `72b127129ed8d11edd2fa2cdf4b56485d259bb5dcb1cb3f691d06a65e19bc6ab`.
- Total downloaded video bytes: 9,269,185. Total downloaded full-Parquet bytes:
  3,240,727. Both video IDs belong to the non-audit train/validation splits.

## Video path

PyAV 18.1.0 decoded 653/653 frames for train episode 0 and 469/469 for
validation episode 1802. Both streams are H.264, 640×480, 10 fps, have
monotonic timestamps and no audio stream. Decode times were 0.7183 s and
0.5469 s in this run.

The existing pinned VideoPrism-B model was run from the preserved R1
environment and checkpoint without modification:

- Model revision: `google/videoprism-base-f16r288@be719a406d563b66f0ac969e7c94bab8e997c81a`
- Upstream source commit: `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`
- Checkpoint SHA-256:
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`
- Runtime: Python 3.11.16, JAX 0.6.2, TensorFlow 2.18.1, PyAV 18.1.0,
  `TFRT_CPU_0`; GPU disabled.
- Eight endpoint-inclusive uniform frames were resized short-side to 288,
  center-cropped to 288×288, and normalized to RGB float32 `[0,1]`.
- Both forwards returned finite `[1, 2048, 768]` token arrays and finite pooled
  768-D features. First compile+forward was 10.8575 s; the second forward was
  2.0739 s. Combined process wall time was 15.822 s and peak working set was
  4,157,022,208 bytes.
- Pooled feature hashes: train
  `ec15f448ce26b9f3fcc9936da549b98b5132a4ada3c23b033d5c7f2faf73284c`;
  validation
  `6392dba6bb093c690134ef7c2fad3fa26f1eceab92ba76a23ad6f2698fa7b2a1`.
- Exact one-off smoke script SHA-256:
  `70e082b10b2d8343b9104740494855a97d3357200db52f383cb392be02e46be8`.

## Label/option mapping

Both full Parquet episodes had monotonic 10 Hz timestamps and 14-D
`observation.state`. For episode 0, three candidate subtask strings mapped to
target indices 0, 1 and 2 (223, 63 and 367 frames). For episode 1802, four
candidate strings mapped to target indices 21–24 (94, 145, 79 and 151 frames).
No target index fell outside its episode's `low_level_tasks` list. These frame
counts are annotation support, not independent Decision examples. The
`action` field exists but was read only to inspect its dimensionality; it is
excluded from the intended input.

## Unique semantic choice mapping

The metadata audit across the frozen train and development-validation IDs
found repeated candidate wording in 5,273 of 9,113 episodes (11,333 repeated
entries). These repeats occur when the same action text appears at multiple
workflow positions. The Parquet target is a global `low_level_task_index`, so
the adapter resolves its catalog text, deduplicates exact NFKC/case/whitespace
normalized option text while preserving first-seen order, and maps the target
to that unique semantic option. It fails closed if the target text is absent
or fewer than two unique options remain. The schema and behavior are in
`src/tiny_omni_decision/habit_decision.py`; CPU tests cover duplicate collapse,
global-ID mapping and malformed targets.

After deduplication, option counts across the non-audit train/validation IDs
are: 2,421 episodes with one unique action, 2,532 with two, 2,334 with three,
1,707 with four, and 119 with six. Single-action episodes cannot form a
choice task and must be excluded from a multi-option Decision pilot; this is a
data-coverage limitation, not permission to invent distractors. On the two
already acquired Parquets, all 653 train and 469 validation frame labels map
to a unique semantic option (3 and 4 distinct options respectively). This is
still only a two-episode schema check, not evidence for the rest of the split.
The fixed mapping receipt is saved outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\habit-label-schema-20261121-attempt-02\habit-choice-mapping.json`
(SHA-256 `ae5e47d388224c26740c9001a4641d89950a6c35750e61c0d93f03c7bf4963ce`).

## Interpretation

The pinned video encoder, decoder, feature shape, robot-state field, and
candidate/target mapping are executable on two real clips. This does not prove
the proposed task is learnable, the modalities are complementary, a calibrated
readout works, or the current split has no duplicate media content. Before a
quality run, audit the selected train/validation media hashes and task-group
support, construct causal windows that exclude action/future fields, and
compare options/text-only, state-only, video-only, and paired readouts. No
model training, quality evaluation, CUDA use, or cloud work occurred.

Artifacts are in
`C:\CodexArtifacts\pretrained-reuse-r2\habit-media-runtime-smoke-20261121-attempt-01`
and
`C:\CodexArtifacts\pretrained-reuse-r2\habit-label-schema-20261121-attempt-01`.
