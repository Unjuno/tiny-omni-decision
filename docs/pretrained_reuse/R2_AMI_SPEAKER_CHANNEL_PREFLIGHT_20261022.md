# AMI speaker-channel preflight — 2026-10-22

## Purpose

Continue the prospective audio-to-visible-speaker correspondence task without
training or looking at the sealed/unseen split. The design is useful only if an
audio segment is specific to its speaker and can be aligned to the mapped camera.
This preflight checks the existing local acquisition and the official signal
contract before any model run.

## Findings

The current local AMI selection does **not establish a speaker-isolated audio
input**. Its candidate builder reads one `Mix-Headset.wav` per meeting. The
24-row train-only timing receipt has exactly one full-file audio hash within
each of seven meetings, shared across all candidate agents in that meeting.
This shows that the source is a mixed meeting signal; it does not prove whether
each annotation-aligned crop is dominated by the annotated speaker.

The official signal documentation describes 24 WAV channels per meeting and
provides microphone/camera mappings in `corpusResources/meetings.xml`. A
participant headset channel plus that participant's synchronized close-up
video is the cleanest first input. If the design instead uses annotation-aligned
crops from the mixed WAV, it first needs a train-only check that the target
speaker dominates those crops. Either input design must compare audio-only,
video-only, shuffled-time, and candidate-order controls. The selected E/I
recordings have an official common-clock synchronization account; TNO has
looser, human-checked synchronization and should be excluded from the first
strict-timing cohort. The official sources also document frame-drop repair for
E/I videos.

The local annotation metadata contains participant, camera and channel fields.
However, the selected local media directory contains the mixed WAV and videos,
not the required individual `Headset-N.wav` streams. The official index lists
individual headset streams, but direct network access from this host timed out
over both default and IPv4 curl, and an in-app browser navigation also timed out.
No channel media was downloaded. This blocks the stricter channel-isolated
design here; the mix-based alternative remains unverified. This is an
environment/data-gate block, not a negative model result.

## Evidence and scope

- Source commit: `0bbb5d17cc9a2b7e5e0d9353e3e70d0c6c1ff17a` (PR #15 roadmap head).
- Existing expanded preflight: `C:\CodexArtifacts\pretrained-reuse-r2\ami-av-probe-20261021\media-and-annotation-preflight-expanded-20261021.json`, SHA-256 `bbead20d61d57429cbf7a5358bfb5ac890e6da108a35c20e6ad8621cd67a9b4b`.
- Existing train-only 24-row receipt: `C:\CodexArtifacts\pretrained-reuse-r2\ami-av-probe-20261021\train-av-sync-coarse-seed17-20261021-03\result.json`, SHA-256 `45fb9ea3ec815a991477cb39ca31dfe8bcd1ccac7bb1a42c9b183eb6a8615886`.
- Those 24 rows span seven meetings; all candidate agents in each meeting share the same single mixed-audio SHA-256. Receipt status remains `NOT_ADJUDICATED`.
- Existing visual contact sheets were viewed for train rows only. They show a mixture of clear, small, and absent target people; they do not prove who is speaking or establish lip synchronization.
- Official AMI sources: [audio/video signals and channel mappings](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml), [meeting-room synchronization](https://groups.inf.ed.ac.uk/ami/corpus/meetingrooms.shtml), [known data problems](https://groups.inf.ed.ac.uk/ami/corpus/dataproblems.shtml), [download index](https://groups.inf.ed.ac.uk/ami/download/).

No annotation or media outside the already acquired selected SA/SB subset was
opened. No predictions, fitting, checkpoint selection, audit evaluation,
feature extraction, GPU process, or existing artifact was touched. The current
GPU remains occupied by PID 12596 (15,821/16,384 MiB at the final read-only
check).

## Decision

Do not describe the current mixed-audio input as speaker-isolated, or begin a
model comparison before choosing and freezing one of these input contracts:
(a) individual participant channels, with hash-pinned files and
channel-to-participant mapping verified against official metadata; or (b)
annotation-aligned mixed-WAV crops, with train-only evidence that target speech
dominates and a fixed overlap policy. The source remains a **conditional
feasibility candidate**, not an admitted quality benchmark. The existing five
development parent groups remain low-power feasibility data, not a 95–98%
product-quality gate. No product or modality claim follows from this preflight.
