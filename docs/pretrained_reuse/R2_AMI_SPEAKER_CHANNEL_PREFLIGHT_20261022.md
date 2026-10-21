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

The selected local media directory contains the mixed WAV and videos, not the
individual headset streams. Direct access to the AMI download host timed out,
but a read-only Dataset Viewer query against the University of Edinburgh's
official Hugging Face mirror found a usable train example. At pinned dataset
revision `46f28f2503e2ec48f8867a84eef356c70476beab`, config `ihm`, split
`train`, row 16968 is meeting `ES2002a`, speaker `MEE006`, microphone `H00`,
and time 1096.030–1098.430 seconds. Local `meetings.xml` maps MEE006 to agent A
and camera Closeup1; the interval exactly matches the existing train video
candidate. PyAV decoded the official row as mono 16 kHz IEEE-float PCM, 38,400
samples / 2.4 seconds. The locally saved audio SHA-256 is
`6a4b8886a7ee16246d145beb21a5b3508cca2c225167cab5ad42e4a3319f1d7d`.
The dataset card at that revision states CC-BY-4.0. The signed download URL was
not recorded, and the audio file remains outside Git.

This proves a channel-specific acquisition/decode path for one train row; it
does not establish cohort coverage, speaker dominance across the prior mixed
crops, visible speaking status, or model performance. The mix-based alternative
remains unverified. The previous network block is therefore cleared for this
single-row smoke only, not for a model comparison.

A CPU-only audit of the already acquired train word annotations then checked
the frozen 24 timing rows against each target agent's own word intervals. The
target speaker had positive-duration annotated speech covering 100% of each
crop in 24/24 rows; other-speaker word intervals overlapped in 7/24. This
supports that the clips contain the intended speaker's speech, but because the
clips were selected around speech, it is not an independent estimate of target
dominance in the mixed waveform and does not establish visible speaking status.
The machine-readable result is
`C:\CodexArtifacts\pretrained-reuse-r2\ami-av-probe-20261021\ami-target-speech-overlap-audit-20261022.json`,
SHA-256 `c0174176be975ca3b28fbb17225afdaa77249308b8a2ffb81f47f63a1496965e`.

## Evidence and scope

- Source commit: `0bbb5d17cc9a2b7e5e0d9353e3e70d0c6c1ff17a` (PR #15 roadmap head).
- Existing expanded preflight: `C:\CodexArtifacts\pretrained-reuse-r2\ami-av-probe-20261021\media-and-annotation-preflight-expanded-20261021.json`, SHA-256 `bbead20d61d57429cbf7a5358bfb5ac890e6da108a35c20e6ad8621cd67a9b4b`.
- Existing train-only 24-row receipt: `C:\CodexArtifacts\pretrained-reuse-r2\ami-av-probe-20261021\train-av-sync-coarse-seed17-20261021-03\result.json`, SHA-256 `45fb9ea3ec815a991477cb39ca31dfe8bcd1ccac7bb1a42c9b183eb6a8615886`.
- Those 24 rows span seven meetings; all candidate agents in each meeting share the same single mixed-audio SHA-256. Receipt status remains `NOT_ADJUDICATED`.
- Existing visual contact sheets were viewed for train rows only. They show a mixture of clear, small, and absent target people; they do not prove who is speaking or establish lip synchronization.
- Official AMI sources: [audio/video signals and channel mappings](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml), [meeting-room synchronization](https://groups.inf.ed.ac.uk/ami/corpus/meetingrooms.shtml), [known data problems](https://groups.inf.ed.ac.uk/ami/corpus/dataproblems.shtml), [download index](https://groups.inf.ed.ac.uk/ami/download/).
- Official University of Edinburgh mirror: [`edinburghcstr/ami` pinned revision](https://huggingface.co/datasets/edinburghcstr/ami/tree/46f28f2503e2ec48f8867a84eef356c70476beab). Read-only Viewer discovery scanned 339 train rows; the acquired row and decode receipt are in `C:\CodexArtifacts\pretrained-reuse-r2\ami-hf-channel-smoke-20261022-attempt-03\receipt.json` (audio SHA-256 above).
- Viewer `/search` and `/filter` returned HTTP 500 for this dataset during the follow-up cohort lookup; a throttled `/rows` range search then received HTTP 429 and was stopped. The 24-row annotation audit above therefore uses the already acquired local train annotations and does not imply that 24 individual-channel files were fetched.

No held-out split, unseen meeting, or sealed audit was opened. The only newly
opened utterance was the single official-train row listed above; the Viewer
metadata scan was confined to its train split. CPU annotation interval analysis
and audio decode were performed, but no model inference, model feature
extraction, fitting, checkpoint selection, or audit evaluation occurred. No
GPU process or existing model artifact was touched. The GPU remained occupied
by PID 12596 (15,821/16,384 MiB at the final read-only check).

## Decision

Do not describe the current mixed-audio input as speaker-isolated. The
individual-channel contract is demonstrated for one hash-pinned train row and
the selected mixed-WAV crops do contain target-speaker speech, but cohort-wide
individual-channel acquisition or target dominance remains unverified. Before
model comparison, obtain the frozen cohort's participant channels or validate
dominance using an appropriate train-only signal-level/diarization method, then
freeze missing/overlap handling. The source remains a **conditional feasibility
candidate**, not an admitted quality benchmark. The existing five development
parent groups remain low-power feasibility data, not a 95–98% product-quality
gate. No product or modality claim follows from this preflight.
