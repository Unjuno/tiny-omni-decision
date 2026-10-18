# Perception Test AV event-to-object task protocol

Status: frozen protocol and media plan; pinned TRAIN/VALID media acquisition
is in progress. Single-event CPU feature-pipeline smokes passed, but full
feature extraction, readout fitting, and task evaluation have not started. This
is a development task for R2, not a product-quality claim or a blind test.

## Question and scope

For an annotated audible event and a set of visible object tracks in the same
video, select which supplied track produced the event. This evaluates
conditional audio-visual source selection: proposals and the event interval
come from annotations, so this is not autonomous object detection, sound
event detection, or unrestricted scene understanding.

The pinned Perception Test revision is
`3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf` (CC-BY-4.0 per that revision's
license). Only official TRAIN and VALID video archives are in scope. No TEST
archive or sealed audit is opened. Public annotations were already inspected
for feasibility; VALID is therefore development validation, not blind. The
public annotation set has no participant IDs, so participant-level separation
is UNKNOWN.

## Frozen eligibility and candidate set

Use the existing label-blind `event_candidate` rule unchanged. Keep only
events with one resolved non-Muted source parent, `is_visible == 1`, fully
before the split-specific cut, and at least two visible, unmasked, nonempty
object boxes at a frame inside the event. Select the eligible frame nearest
the event midpoint, breaking ties toward the earlier frame. Include every
visible track at that frame, without truncation. Hold an event if its source
track is absent at that frame. The resulting candidate sets contain 2 to 48
tracks. Do not turn held or multi-source events into negatives.

Each question is one event and its candidate crops. The target is the source
track among those candidates. Randomize candidate order deterministically per
event and store the permutation separately from model inputs. Model inputs
must exclude the sound label, source parent ID, ground-truth candidate track
IDs, video ID, filename, split ID, and any annotation-derived answer field.
These identifiers may be retained in a restricted evaluation join table only.

The intended raw inputs are the audio segment corresponding to the annotated
event interval and the selected-frame crop for each candidate track. Before
feature extraction, verify frame-to-audio timestamp mapping against the pinned
media metadata and decoder behavior; record decoder/version, crop coordinates,
resampling, and hashes. Use the same candidate set and preprocessing for all
arms. Do not use sound label text or object-category labels as model input.

## Pinned fresh media inventory

Canonical plan: external artifact
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-media-plan-02\plan.json`

- Plan SHA-256: `9ed4f44f3a0cc85465c5a2955706a957326ef36694311c25b4ee51e3c9f18a25`.
- TRAIN: 400 fresh videos, 818 eligible events, 5,214,316,862 bytes; 300
  videos / 630 events for fitting and 100 videos / 188 events for calibration.
- VALID: 400 fresh videos, 901 eligible events, 5,498,652,000 bytes, reserved
  for development validation.
- Every video belongs to exactly one video-level group; fit, calibration,
  and development-validation video IDs are pairwise disjoint.
- All 791 unique video IDs present in six pinned earlier Perception media
  manifests are excluded. The newly selected TRAIN and VALID IDs are disjoint.
- Largest planned clip: 67,109,562 bytes (TRAIN) and 74,725,760 bytes (VALID);
  each split remains below 8,000,000,000 bytes. Both acquisition output
  directories were absent at preflight; C: had 314.1 GiB free.

These exclusions concern video IDs in prior media manifests. They do not
establish independence by participant, source recording, or annotation
authoring process. The official VALID annotations have been inspected and
must not be described as blind.

## Evaluation and decision rules

The first candidate is a task-specific frozen-feature Path A system using the
already audited `facebook/dinov2-small` at revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056` (22,056,576 parameters,
Apache-2.0) for object crops and the `laion/clap-htsat-unfused` audio branch at
revision `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a` (28,190,872 parameters
including audio projection, Apache-2.0 model card; preserve distinct CC0 code
notice). Their combined frozen encoder count is 50,247,448 parameters before
the small Decision readout; this is only the component count for this task,
not the full Omni model or the final product size. These encoders were not
pretrained as a joint audio-visual pair, so the readout must learn the task
alignment. The CLAP processor has a 10-second limit; determine and test a
deterministic whole-event windowing/resampling path against decoded PTS before
using it. Do not accept its default random truncation. Metadata-based duration
estimates put 20/818 TRAIN and 17/901 development-VALID events over 10 seconds
(TRAIN median 1.351s, max 35.003s; VALID median 1.235s, max 30.692s); final
counts must be recomputed from actual decoded PTS. See the pinned R1 audits for
the checkpoint and runtime evidence.

Derive a nested selection split from the plan's 300 TRAIN videos before model
fitting: SHA-256 order by
`perception-av-event-v1:model-selection:17:{video_id}`, first 60 videos for
model selection and the remaining 240 for parameter fitting. Keep all videos
grouped. Use the separate 100 TRAIN calibration videos only for a predeclared
probability calibration fit after selecting the readout. Select architecture,
training budget, and checkpoint using fit/selection TRAIN groups only; report
the selected candidate once on the 400 official VALID development videos.
Keep all attempts and failures. Candidate order is deterministically permuted
by sorting track IDs on SHA-256 of
`perception-av-event-v1:17:{video_id}:{sound_id}:{track_id}`; use this order
for every comparison arm and do not expose identifiers or permutation seeds to
the model.

Evaluate the exact same events and randomized candidate lists in three
conditions: visual-only, audio-only, and joint audio+visual. Also include
uniform-choice and label-prior baselines. Audio-only cannot identify one of
several candidate tracks without visual evidence; retain it as a control rather
than forcing it to use annotation IDs. No condition may receive source labels,
track IDs, filenames, or video IDs. Report Accuracy, NLL, Brier, ECE, mean
confidence, and counts overall and by candidate-count bucket and sound label.
For joint benefit, report paired per-event changes against both unimodal
conditions and a video-grouped bootstrap interval. Calibration fit is limited
to the fixed TRAIN calibration group. No evaluation-driven text/prompt search,
candidate removal, or repeated VALID-based checkpoint selection.

Do not proceed to model fitting until acquisition hashes, decoder integrity,
split isolation, candidate-to-crop joins, and input-field exclusions pass
CPU-side tests. If the task or frozen feature pair cannot distinguish source
matching beyond controls, record that result and stop rather than redefining
the task after observing VALID outcomes.

## Current execution boundary (snapshot 2026-10-18 04:08:05 UTC)

The feasibility audit and media plan used CPU/network metadata only. Media
acquisition then started from source commit `b18340a04d2b9f3a0fceb06cbce2ad0a39c63a3c`
in fresh split-specific directories using the exact plan hash above. The
acquisition runner SHA-256 is
`39e7b3c51fa3b527b69ebb8283e4993cd4cd92ad2e48efa1d5348a22df814818`.
TRAIN started at 03:27:33 UTC and has 74/400 verified clips (991,192,038
bytes); VALID started at 03:28:00 UTC and has 78/400 verified clips
(991,316,915 bytes). Both manifests say RUNNING with no error. About 312 GiB
remains free. These are live snapshot counts, not completion claims.

A CPU-only partial PTS audit decoded the 46 TRAIN / 55 VALID videos in its
manifest snapshot that contain selected events: 76 and 100 events respectively. All 176
event windows were fully covered by the decoded audio PTS; none had zero or
partial overlap. The receipt is
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-decoder-pts-smoke-04\result.json`
(SHA-256 `5c37848b3f2786808bec0c2118e53634bce658f60b913fda4f0c5c90ba268395`),
using PTS audit script SHA-256
`1c5392ee54d2337758339317a3526fe1ee17e581be62d4c52ebcba2fc9b9e6eb`.

In those decoded videos, metadata differed from media for frame count in 1/46
TRAIN and 0/55 VALID videos; audio sample count in 38/46 TRAIN and 53/55 VALID;
and audio sample rate in 10/46 TRAIN and 19/55 VALID. One observed video had
825 metadata frames versus 824 decoded frames, but its selected event's frame
indices remained present. Audio intervals must therefore use actual decoded
video PTS and actual decoded audio rate/duration, not annotation-average FPS
or nominal sample count. A separate metadata-only estimate flagged 15 TRAIN
and 11 VALID event tails; of these, 2 TRAIN and 1 VALID events were present in
the PTS snapshot, and all 3 were fully covered under actual PTS. Do not
extrapolate the partial audit to all 800 planned videos. The full PTS audit
remains a gate before feature extraction.

The metadata-only event-duration audit receipt is
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-audio-coverage-estimate-01\result.json`
(SHA-256 `9de0ce9493e8e1508b46bb7419c8cc3ec534a651d2f4f3513ee244956bebda65`).
The frozen nested video-group split and candidate-order implementation is
`scripts/perception_av_event_protocol.py`, with CPU regression tests.

No feature extraction, model load, training, GPU evaluation, or TEST access
has occurred. The GPU is occupied by another process; this acquisition uses no
GPU. The runner checks archive/member pins and CRCs, hashes every extracted
clip, and preserves an incomplete manifest on failure. Existing datasets,
model artifacts, runs, and checkpoints are unchanged.

## Follow-up execution snapshot (2026-10-18 04:18 UTC)

The two original acquisition processes remain active in their original
split-specific output directories. Their manifests currently contain 95/400
TRAIN and 92/400 VALID verified clips; no new acquisition process was started.
These are live snapshot counts. Full acquisition and full PTS/integrity audit
remain incomplete, so model fitting is still gated.

Before fitting, CPU encoder plumbing was smoke-tested on one fixed TRAIN event
(`video_11362`, with `sound_id=4` used only for provenance joins). DINOv2-small
produced 9 deterministic finite candidate vectors from decoder frame 390 at
1080x1920, with no labels passed to the image encoder. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-vision-smoke-01\result.json`;
feature SHA-256 `6d58275a384c4f480de283f1bac81422d3ab728489ce72c5d6576381a3176d2b`;
script SHA-256 `a075fa088b7a1c95f8a02b28d721de45292e3197e6d28e22c0ea16f6fe80859c`.

The pinned CLAP audio branch was also exercised on that TRAIN event using its
actual decoded video PTS interval [12.6944575, 13.3679036] seconds and decoded
audio timeline. PyAV resampled to 48 kHz, yielding 32,325 event samples and one
512-dimensional normalized event vector. Repeated CPU inference was bitwise
equal and finite. No task score, training, GPU use, or TEST access was
involved. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-audio-smoke-02\result.json`;
feature SHA-256 `e7ad3c64a35fcad18e276fbf19026d5a75f1799087927eb3c1f3268cdecad7ac`;
receipt SHA-256 `6d58ac808fcfa5428710eefc453d7939b719bb4b1ae653581e2e74190b683142`;
script SHA-256 `17237851552a8def4f4e8557a213d4cb6725a98adc42970e9905363bb6df2b1f`.
The smoke uses PyAV's resampler and did not upgrade or add runtime packages.
CPU smoke tests: 13 passed.

These one-event encoder smokes do not substitute for the complete media,
PTS, split-integrity, and annotation/crop join gates. No readout fitting or
quality evaluation has started.

## Partial PTS audit follow-up (2026-10-18 04:38 UTC)

An initial attempt (`perception-av-event-decoder-pts-smoke-05`) completed its
TRAIN scan but ended before writing a receipt because the VALID annotation
source path was incorrectly pointed at the TRAIN-only source directory. The
exception was `FileNotFoundError` for `valid_annotations.zip`; this is an
operator path error, not an audit finding. No files in the source or media
directories were changed. The corrected attempt used the separately pinned
VALID source and is recorded at
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-decoder-pts-smoke-06\result.json`
(SHA-256 `a515f984ef3389b16f86c2031fb3d4f40da1b9b9e0d24b39988e35eb06c4dff0`),
with the existing PTS audit script SHA-256
`1c5392ee54d2337758339317a3526fe1ee17e581be62d4c52ebcba2fc9b9e6eb`.

This corrected `--allow-partial` snapshot decoded 111 TRAIN videos / 213
events and 115 VALID videos / 237 events. Every scanned event had full decoded
audio coverage (minimum 1.0; zero and partial overlaps both 0). Three scanned
videos per split differed from metadata frame counts; decoded audio sample
rates differed in 30 TRAIN and 29 VALID videos, and decoded sample counts
differed in 93 TRAIN and 104 VALID videos. The actual PTS path handled these
metadata differences; they reinforce that nominal metadata must not be used
for event slicing. The remaining planned media had not yet been acquired when
this snapshot was taken, so `PASS_PARTIAL_PTS_AUDIT` is not the full integrity
gate.

At 04:38 UTC the original acquisition processes remain active with 141/400
TRAIN and 129/400 VALID clips; both manifests have zero acquisition errors.
The GPU is not used by these CPU audits. No fitting, task scoring, or TEST
access has started.
