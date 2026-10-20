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
No validation/test/sealed data, model, or feature cache was opened for this
protocol. The subsequent media fetch is limited to the 405 frozen official
train scenes and writes to a separate external attempt directory. A complete
media hash receipt and split hash check are required before feature extraction.

This diagnostic cannot establish natural AV understanding, broad decision
quality, or the 95–98% target. It is a narrow test of synchronized temporal
binding and the evaluation pipeline.
