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

## Partial annotation-to-crop join audit (2026-10-18 05:00 UTC)

Added `scripts/audit_perception_av_event_crop_joins.py` to validate the pinned
TRAIN/VALID event rows against ordered candidate track IDs, selected-frame
boxes, normalized crop coordinates, media resolution, acquired-file hashes,
and video-group partitions. The report keeps the source candidate's option
index in a separate label-join field; this audit does not expose it to either
encoder and computes no task score.

The first run's `eligible_events_total` denominator accidentally included
candidate events from all source videos, not only the frozen 400-video plan
(receipt SHA-256 `fab721db0b7431ed0511fc5b4dbe5c6e84a9c33dd57251dc68e1644008279358`).
That summary is invalid and must not be used. The corrected runner filters to
planned video IDs and asserts that event totals match the plan before reporting.
Attempt 02 confirmed corrected event accounting; attempt 03 additionally
verifies the exact 240 fit / 60 selection / 100 calibration / 400 development
validation video partition. All attempt outputs are preserved outside Git.

Authoritative partial receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-crop-join-audit-03\result.json`
(SHA-256 `4d3a57ebabcb830ceec2b30dbd4b1a0d1877d73f11892eaf9dabb00b462d2942`),
using audit script SHA-256
`d08b54cbf38fd6e5f2ae462695cb31d03bf41d3ac0ba0520d59d43abf1c7bb4c`.
It hash-checked all acquired files and validated crop joins for 361/818 TRAIN
and 349/901 development-VALID eligible events. The other 457 TRAIN and 552
VALID events belong to not-yet-acquired videos, not crop failures. Video IDs
remain split-disjoint; the exact nested video group counts match the frozen
plan. Ruff passes; full CPU suite: 2,449 passed, 11 skipped. Since media
acquisition is incomplete, this is still a partial gate and no feature-bank
fit or quality evaluation has started.

### Expanded pinned crop-join snapshot (attempt 04)

The next partial crop audit used byte-for-byte copies of the parsed live
manifests, kept outside Git, so later downloads cannot change its inputs.
Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-crop-join-audit-04\result.json`
(SHA-256 `ffa3e3f154a06555b17032c6a9dd99bfbcc6bce1837f1dca9aab8b7f53968c34`).
Manifest snapshot hashes are TRAIN
`685685c456140ab2f64d2ec2cb1aeaf9f6334ba78bfa768d002852ee6c5935e0` (249
videos) and VALID
`3b682a806ae7dcab02c9173a82a1a57480fc2b3724778319018dd445695576ce` (225
videos). It validated 509/818 TRAIN and 481/901 development-VALID event crop
joins, with 309 and 420 events respectively waiting for media; those are
unacquired-plan rows, not join failures. It rechecked all per-video file hashes
and the 240/60/100 TRAIN and 400-video VALID group partition. Status remains
`PASS_PARTIAL_CROP_JOIN_AUDIT`; full acquisition, complete PTS and split gates,
and all model fitting remain pending.

## Frozen-feature Decision readout implementation (2026-10-18)

Added `tiny_omni_decision.av_event.AVEventSourceScorer` as the task-specific
Path A readout for the later, gate-authorized feature bank. The default
hidden-128 scorer has 246,658 trainable parameters. Together with the pinned
22,056,576-parameter DINOv2-small and 28,190,872-parameter CLAP audio branch,
the intended task components total 50,494,106 parameters. This is not a full
Omni resident-model or package-size count.

The module has explicit `video_only`, `joint`, and `audio_only` conditions.
The joint branch uses an option-conditioned learned gate; the video-only
branch receives crop features only. With opaque candidate IDs and no option
text or visual features, audio-only has no candidate-specific evidence and
therefore returns uniform probabilities by construction. Inputs contain only
frozen numeric features and supplied option order; the module cannot accept
question text, transcripts, filenames, candidate IDs, or labels. CPU tests
cover option permutation equivariance, modality isolation, finite scores, and
gate bounds (6 passed; Ruff passes).

This is implementation evidence only: no feature extraction, fitting,
checkpoint selection, calibration, GPU use, or task-quality evaluation has
occurred. Extraction and training remain blocked on complete pinned-media
PTS, split, and candidate-crop gates.

## Long-event CLAP window smoke

A second TRAIN-only audio smoke covers `video_8679` / `sound_id=0` (IDs used
only for provenance joins), whose decoded video PTS event interval is
1.0666–35.0310 seconds. The 33.9644-second event yielded 1,630,291 samples at
48 kHz and four non-overlapping CLAP windows: three full 480,000-sample
windows and a 190,291-sample tail. All four 512-dimensional window features
were finite and bitwise-identical on replay; the duration-weighted pooled
feature was finite. No truncation, task scoring, training, or GPU use occurred.
Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-audio-smoke-03-long\result.json`
(SHA-256 `5a21386f04b4d420f66094b87c45157bef46ba946d6fb21a0751479ff15e6f86`);
window feature SHA-256
`57be08970b404179d15a530edb6a250f49e509ed2301a8a513b1c719f223efa3`;
runner SHA-256 `31096a825fd487303a3ce3490f3e38e9b5b20cb9085c1ae6282e1e86d0643613`.
The event-key helper requires exactly one eligible TRAIN record; targeted
CPU tests: 14 passed; full CPU suite: 2,442 passed, 11 skipped, with Ruff
passing.

The same runner was then exercised on a 20.3-second TRAIN event with decoded
native audio at 44.1 kHz (`video_2116` / `sound_id=8`, provenance joins only).
It used PyAV's decoder resampler to produce 974,400 event samples at 48 kHz,
then encoded two full windows and a 14,400-sample tail. Audio and video PTS
covered the complete event; all three window vectors were finite and repeated
bitwise-identically. No labels were passed to the encoder, and no scoring,
training, GPU use, or TEST access occurred. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-audio-smoke-04-44100\result.json`
(SHA-256 `26bf4210d40bf7f839235f388c360c04a411415cc0b9a3f31fcb8bafd6f1c6e7`);
event-feature SHA-256
`ac467e78cbde7c86f60e3dfd807b2c2da95765f476c1f0dea4e2cc2d62dded81`;
window-feature SHA-256
`40a1d03a73804ec911f3be66c9b2d27c1759f216ef1545b615ef18474ae4f97f`.

At this snapshot, original acquisitions were at 153/400 TRAIN and 139/400
VALID clips, both still running with zero recorded acquisition errors. This
does not close the full media/PTS/split audit gate.

## Candidate crop join and streamed-frame smokes

The vision smoke now accepts a provenance-only TRAIN event key through the
shared `select_training_event` helper, and retains only the selected decoded
RGB frame rather than holding every decoded frame in RAM. Unit tests require
the key to match exactly one candidate event. On the 1,051-frame
`video_8679` / `sound_id=0` event, the streamed implementation produced 14
finite DINOv2 crop feature tensors and repeated bitwise-identically. Its
feature SHA-256 (`0046d75584fd87be044c0dc3269bdec89b0dba0adee7064ceb1733ee47e544a2`)
exactly matches the earlier all-frames-in-memory smoke, demonstrating output
parity while avoiding retention of the full decoded clip. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-vision-smoke-05-streamed-long\result.json`
(SHA-256 `545bfbb0bda6e15c904fe676f9fa74b8e05f7b827a718358cab08923c08d4152`).

The streamed path also passed on `video_10372` / `sound_id=9`, where annotation
metadata says 825 frames and actual decode yields 824. Selected frame 510 lies
inside the event interval and decoded stream; all 8 DINO candidate crops were
finite and bitwise-repeatable. The resulting feature SHA matches the earlier
smoke on this same event. The runner records the one-frame discrepancy instead
of rejecting the usable selected frame. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-vision-smoke-04-streamed-decode\result.json`
(SHA-256 `01c69fa4f68244c24eb25b7cc47a105758cc54b3e1f7116e696745f552177456`).

Ruff passed; full CPU suite: 2,443 passed, 11 skipped. Neither smoke scored
task correctness, trained a readout, used the GPU, or accessed TEST. The
acquisition manifests at this update show 164/400 TRAIN and 151/400 VALID
clips, still RUNNING with zero recorded errors. Full acquisition and the
complete PTS/split gate remain open.

## Streamed selected-frame follow-up (2026-10-18 04:53 UTC)

The candidate event join is now shared between the audio and vision smoke
runners via `scripts/perception_av_event_protocol.py`; it fails closed unless
the provenance key selects exactly one eligible TRAIN event. The DINO runner
was changed to decode sequentially while retaining only the chosen RGB frame,
and now records `decoded_video_frames - metadata_video_frames` rather than
requiring equality.

For the 1,051-frame `video_8679` event, streamed decoding yields the exact same
candidate feature file hash as the earlier implementation that retained every
decoded frame: `0046d75584fd87be044c0dc3269bdec89b0dba0adee7064ceb1733ee47e544a2`.
The 14-candidate output is at
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-vision-smoke-05-streamed-long\result.json`
(receipt SHA-256 `545bfbb0bda6e15c904fe676f9fa74b8e05f7b827a718358cab08923c08d4152`).

The 824-decoded/825-metadata frame case also passes streamed decoding. It
selects actual decoded frame 510 inside the annotated interval and produces
the same eight candidate features as the prior smoke; no metadata-only frame
was indexed. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-vision-smoke-04-streamed-decode\result.json`
(receipt SHA-256 `01c69fa4f68244c24eb25b7cc47a105758cc54b3e1f7116e696745f552177456`).
Ruff passes and the full CPU suite passes: 2,443 passed, 11 skipped. This is
feature-pipeline parity/integrity evidence, not task quality. The live
acquisition snapshot is 167/400 TRAIN and 152/400 VALID, both RUNNING with no
recorded acquisition errors; the complete pre-feature audit remains pending.

## Expanded partial PTS audit (2026-10-18 05:08–05:27 UTC)

CPU-only attempt 07 decoded a partial acquisition snapshot against the pinned
plan. The receipt is
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-decoder-pts-smoke-07\result.json`
(SHA-256 `f5dc3cd585da036ef02d12e1a75b22978c9b5d0dce10ff11305f8533c60eca39`),
using audit-script SHA-256
`1c5392ee54d2337758339317a3526fe1ee17e581be62d4c52ebcba2fc9b9e6eb` and
PyAV 18.1.0. It decoded 199 videos per split and 417 TRAIN / 428 VALID events.
All decoded event windows had full decoded-audio PTS coverage (minimum 1.0;
zero/partial overlaps 0). Metadata differed for video frame counts in 7 TRAIN
/ 4 VALID videos, decoded audio sample counts in 167 / 176, and sample rates
in 52 / 49.

This remains a partial diagnostic, not a pass for the complete acquisition.
The acquisition manifests were mutable while the two split scans ran, and the
older runner did not include their exact byte hashes in the receipt; each
split's result therefore pins the plan and individual media hashes but not a
single paired manifest snapshot. The improved runner now hashes the exact
manifest bytes it parsed and rejects duplicate or unplanned video IDs. Its
regression tests pass; use it for the final full audit after acquisition is
terminal. No model, GPU, training, or TEST data was used.

## CPU Path A readout runner (implementation only; no task fit yet)

`scripts/train_av_event_source.py` now consumes a JSONL feature manifest whose
rows carry event/video IDs, split, target index, candidate count, deterministic
candidate-order hash, media and annotation identities, feature paths, and
SHA-256 hashes. It checks feature hashes before loading them, rejects
event/video/media identity leakage across
fit, selection, calibration, and development-validation splits, and requires
the pinned encoder output shapes. DINOv2 outputs one CLS token followed by 256
patch tokens; the runner currently uses the CLS vector per crop and preserves
candidate order. This is the first baseline pooling choice, not a measured
winner.

Training uses `fit` rows only. Checkpoint selection uses video-disjoint
`selection` rows only (minimum NLL, strict absolute `min_delta`, earliest tie),
with seed 17, AdamW, learning rate 1e-3, maximum 100 epochs, patience 20, and
`min_delta=1e-4` as the first fixed engineering schedule. Best and final
weights are kept separately. The saved best file is reloaded and its selection
predictions are verified before metrics and per-example probabilities are
written for calibration and development-validation rows; the runner refuses a
fit when either held-out split is missing. The `video_only`
condition freezes unused audio/gate/joint parameters. Audio-only remains
uniform because opaque candidate IDs provide no audio-side option features.

Nine CPU tests cover CLS pooling/order, video and media split separation,
feature-label bounds, selection logic, separate best/final files, saved-best
reload, and development-validation outputs. Targeted tests and Ruff pass. This
is training/evaluation plumbing only: no full feature manifest, readout fit,
validation quality result, calibration, or GPU run exists yet. The complete
media acquisition and frozen integrity gates remain mandatory before creating
the feature manifest.

## Independent acquired-media snapshot audit (2026-10-18)

While the pinned TRAIN and VALID media-acquisition processes were still live
(PIDs 8040 and 6700), an independent CPU-only snapshot audit verified every
completed file in the two atomic manifest snapshots. It confirmed the pinned
plan SHA-256, unique planned IDs, plan row metadata, split separation, exact
file sizes, and each file's SHA-256 against the acquisition receipt. Both
manifests remained byte-stable during their respective snapshots. The in-flight
`.partial` file in each directory was excluded.

| Split | Verified completed / planned videos | Verified bytes | Manifest snapshot SHA-256 | In-flight partials |
|---|---:|---:|---|---:|
| TRAIN | 367 / 400 | 4,841,516,285 | `7f076ab6a3f0f31692945a27c0be4c8b11c2b46d4b62a504de9c04c82b9b162e` | 1 |
| VALID | 348 / 400 | 4,861,970,073 | `64e2cef1cc877b1b115f581fdc1f9f4edb7b183e34fb9fbed84371625dabcae4` | 1 |

Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-media-snapshot-audit-01\audit.json`
(SHA-256 `8251d50204c51b4ad77af5aa3307c020479616c5dadf894b1cf50ab01a020f5d`).
The standalone auditor is
`C:\CodexArtifacts\pretrained-reuse-r2\audit-acquired-perception-av-media-snapshot.py`
(SHA-256 `a676a158336e0d3feafb78dec9fd883b4e9c8b84ca5fa229c416894ce81ba340b`).
This verifies only acquired-file identity and split membership; it does not
decode media timestamps, validate annotation-to-crop joins, or admit any rows
to feature extraction. The acquisition jobs were left untouched. Repeat the
full integrity/PTS/crop-join gates against stable terminal manifests before
extracting features.

## Partial TRAIN annotation-to-crop join audit (2026-10-18)

A second CPU-only check froze a later TRAIN manifest snapshot (380 acquired
videos; SHA-256
`6ecec75fb1a03cb0f63ea6b67999230ead3006162a7b0658931fe56506c22671`) in a
separate directory while the original downloader continued. Of the 818
eligible events in these planned TRAIN videos, 786 events across 380 acquired
videos passed the source-object/frame/candidate-box join checks; 32 events
remain tied to videos still awaiting download. The candidate-count distribution
and per-event joins are preserved in the receipt. No labels were used for model
fitting or feature extraction.

Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-train-crop-snapshot-01\result.json`
(SHA-256 `3a255f4c77a9a1951413da8837d736276030f4f901fad784034e995c6f5701d3`).
The auditor is
`C:\CodexArtifacts\pretrained-reuse-r2\audit-perception-av-event-train-crop-snapshot.py`
(SHA-256 `bb31abdbd2dc9f07f926b8d3eb1948335964d120998ccc122c1c6f1981d7dd9e`).
This covers only the acquired TRAIN snapshot. The corresponding VALID join,
complete two-split PTS coverage, stable terminal-manifest audit, and all
feature extraction remain pending.

## Complete TRAIN crop-join gate (2026-10-18)

After TRAIN acquisition reached its terminal `PASS_ACQUISITION_NOT_DECODE`
state, the complete 400-video manifest was copied to an immutable audit
snapshot and rechecked against the pinned plan and TRAIN annotation source. All
818 eligible TRAIN events across all 400 videos passed media-hash, resolution,
source-object, selected-frame and candidate-box join validation; no eligible
event remains waiting for media. This closes the TRAIN crop-join gate only.

- Manifest snapshot SHA-256:
  `5ac15ab7ef7efda29a803f783b4a3a13fd3895ebaecbdf9c9c7e0282b9195918`.
- Audit receipt:
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-train-crop-final-01\result.json`
  (SHA-256 `360ec9be151c33d9385de6171397543b6e15f860bdbd9ffb6b114dd37cfe49a1`).
- Snapshot runner:
  `C:\CodexArtifacts\pretrained-reuse-r2\audit-perception-av-event-train-crop-final.py`
  (SHA-256 `8dc00d82b31653b9aa1b9bbe09dd29b8ceaff40d08f09e8c8e50b9843d882007`).
- The audit was CPU-only, from source commit `e79fb9f189946b2a51fad7504bc7bb15cd7ffe83`;
  it loaded no model, did no training, evaluated no quality, and opened no TEST
  data.

The VALID acquisition and crop-join audit, plus complete two-split decoded PTS
and leakage checks, remain required before any feature extraction or readout
fit.
