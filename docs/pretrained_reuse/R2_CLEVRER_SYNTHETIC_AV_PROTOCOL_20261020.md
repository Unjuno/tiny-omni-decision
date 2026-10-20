# CLEVRER synthetic AV timing protocol — 2026-10-20

## Frozen task

This is a controlled temporal-binding diagnostic, not a natural audio/video
benchmark. The pinned CLEVRER source is silent. For each eligible collision,
two examples use the same video and generated question: one places an 880 Hz,
80 ms Hann-windowed sound burst 16 video frames before the collision; the other
places the identical burst 16 frames after. The decision is one of two options:
`sound before collision` or `sound after collision`.

The question names the colliding objects using their unique color/material/shape
attributes but omits the event time. Paired variants have identical text and
video with opposite labels, so text-only and video-only systems are exactly at
chance on complete pairs. A model must locate the specified collision in the
video and compare its time with the sound. A deterministic scene-level time
shift of 0–255 frames is applied equally to both streams to reduce absolute
timestamp shortcuts. All paired variants stay within one scene split.

## Data and split

- Source: `MIT-IBM/CLEVRER` train at revision
  `98b842082ba4f7c18b6b9e3f39145871782a65ef`, CC0-1.0.
- Candidate scene IDs were screened against metadata and cached media filenames
  in 23 registered worktrees plus the R2 artifact root. This screening did not
  hash all old media contents, so “candidate” does not mean a cryptographically
  certified untouched source.
- Seed 17 assigns 324 scenes to train and 81 to development validation;
  scene overlap is zero. The scene-ID hashes and full ordered sample IDs are in
  the external protocol receipt.
- An event is retained only when its collision frame is 16–111 inclusive,
  its two object signatures uniquely identify both objects, and that pair has
  only one collision in the scene. This avoids asking about an ambiguous event.
- The frozen manifest contains 798 train and 195 validation events: 1,596 and
  390 paired examples respectively. Both classes are balanced exactly.
- The source question texts are not used. Query text is generated from the
  annotation attributes; the template is intentionally repeated, so this is
  not a language-generalization result.

## Shortcut preflight

A single threshold on absolute sound-onset frame was selected using train only.
Threshold 268 reached 57.02% on train and 56.67% on the 390 validation examples.
This reveals a modest audio-position shortcut. It is not an encoder or model
result. A learned audio-only arm is required; joint performance must be
compared with it before claiming that both streams help. If the audio-only arm
explains most of joint performance, stop treating this task as evidence for
cross-modal necessity.

## Frozen video representation

Use the already audited and locally present VideoPrism-B checkpoint
`google/videoprism-base-f16r288` at revision
`be719a406d563b66f0ac969e7c94bab8e997c81a`, with upstream source commit
`c6eba0ece3dd49a24ffc44582966997a4a59f1d8` and checkpoint SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
The weights are pinned locally in the v4 Path A assembly. Sample 16
endpoint-inclusive frames from each 128-frame video, average spatial tokens
within each temporal position, and retain `[16,768]` rather than pooling over
time. This gives roughly 0.31 s between sampled frames, shorter than the
0.625 s sound offset.

A CPU-only synthetic-tensor smoke on the exact checkpoint/revision passed for
both 8-frame `[1,2048,768]` and 16-frame `[1,4096,768]` inputs. The 16-frame
forward completed in 8.076 s including its JAX compile. A separate real-media
extraction then decoded all 401 videos referenced by eligible examples. The
external media receipt reports 405 fetched official-train videos (324 train,
81 development validation), 518,188,187 payload bytes, and zero scene or
media-hash overlap; receipt SHA-256 is
`94207aec340f2586a1a375b2a36124908926ebd5daf0d007d759d14bffd7a023`. The
event filters leave 322 train and 79 validation scenes for the 1,596/390
sample manifests, whose ordered-ID hashes are recorded below. No test or sealed
asset was fetched or opened.

The framewise cache is outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\clevrer-synthetic-av-framewise-16-cpu-attempt-01`.
It contains finite float32 `[401,16,768]` features, SHA-256
`141786649f0d4748b7dfe92790c58ac573bb15eaa4b615ec128d796e99215231`, and was
extracted on JAX 0.6.2 CPU in 1,073.43 seconds with a 4,191,924,224-byte peak
working set. The actual-video extraction passed scene/media disjointness checks.

The matched decision-input cache is outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\clevrer-synthetic-av-timing-features-v1-attempt-03`.
It contains row-aligned `[1986,16,768]` video, `[1986,16,1]` synthetic audio
energy, and `[1986,384]` MiniLM text features. Its cache SHA-256 is
`c7382bbe66ed5d0943a658d9c251feb455478f617db9516649956cb2011ebf2b`; the
report records source-manifest hashes and the exact VideoPrism/MiniLM revisions.
The audio path renders the frozen 880 Hz, 80 ms Hann-windowed burst in memory
from its synchronized onset and pools waveform energy into 16 eight-frame bins.
It is a deliberately simple synthetic timing cue, not natural audio encoding.
The pre-training CPU smoke passed with the existing gated LSTM readout (48,868
trainable parameters, `[2,16,2]` logits). The matched 2,048-update fits have
not started.

## Reproducibility and boundaries

The deterministic builder validates the exposure/annotation receipt hashes,
every annotation file hash and scene identity, the train-only source boundary,
scene grouping, paired targets, and the full sample-order hashes. The 2.38 MB
machine-readable receipt is stored outside Git at:

`C:\CodexArtifacts\pretrained-reuse-r2\clevrer-synthetic-av-protocol-v1-attempt-01\protocol.json`

SHA-256:
`82b76e813455061769883b6a46517dbfb3a7a54f7593c1093ab92d8cc261fab7`

The compact frozen configuration is
[`clevrer_synthetic_av_timing_v1.json`](../../configs/pretrained_reuse/clevrer_synthetic_av_timing_v1.json).
During protocol selection, no validation/test/sealed data, model, or feature
cache was used. After the task and split were frozen, the local pinned VideoPrism
and MiniLM weights were used only to create the external train/development
feature caches described above. The 405 source videos came only from the
official train archive; content and split hashes were checked before features
were built.

This diagnostic cannot establish natural AV understanding, broad decision
quality, or the 95–98% target. It is a narrow test of synchronized temporal
binding and the evaluation pipeline.
