# Perception Test AV event-to-object task protocol

Status: frozen protocol and media plan; media acquisition and model evaluation
have not started. This is a development task for R2, not a product-quality
claim or a blind test.

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

The first model candidate is a frozen-feature Path A system using pinned,
licensed pretrained audio and visual encoders after their R1 compatibility,
license, runtime, and input-contract audits. Train only a small decision
module on the 300 TRAIN videos; use the 100 TRAIN calibration videos only for
predeclared probability calibration. Select a checkpoint/hyperparameters on
TRAIN fit/calibration only, then report the selected candidate once on the 400
official VALID development videos. Keep all attempts and failures.

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

## Current execution boundary (snapshot 2026-10-18 04:01:15 UTC)

The feasibility audit and media plan used CPU/network metadata only. Media
acquisition then started from source commit `b18340a04d2b9f3a0fceb06cbce2ad0a39c63a3c`
in fresh split-specific directories using the exact plan hash above. The
acquisition runner SHA-256 is
`39e7b3c51fa3b527b69ebb8283e4993cd4cd92ad2e48efa1d5348a22df814818`.
TRAIN started at 03:27:33 UTC and has 62/400 verified clips (844,253,578
bytes); VALID started at 03:28:00 UTC and has 62/400 verified clips
(837,091,947 bytes). Both manifests say RUNNING with no error. About 312 GiB
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

No feature extraction, model load, training, GPU evaluation, or TEST access
has occurred. The GPU is occupied by another process; this acquisition uses no
GPU. The runner checks archive/member pins and CRCs, hashes every extracted
clip, and preserves an incomplete manifest on failure. Existing datasets,
model artifacts, runs, and checkpoints are unchanged.
