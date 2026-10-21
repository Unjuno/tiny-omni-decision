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

## Evidence and scope

- Source commit: `0bbb5d17cc9a2b7e5e0d9353e3e70d0c6c1ff17a` (PR #15 roadmap head).
- Existing expanded preflight: `C:\CodexArtifacts\pretrained-reuse-r2\ami-av-probe-20261021\media-and-annotation-preflight-expanded-20261021.json`, SHA-256 `bbead20d61d57429cbf7a5358bfb5ac890e6da108a35c20e6ad8621cd67a9b4b`.
- Existing train-only 24-row receipt: `C:\CodexArtifacts\pretrained-reuse-r2\ami-av-probe-20261021\train-av-sync-coarse-seed17-20261021-03\result.json`, SHA-256 `45fb9ea3ec815a991477cb39ca31dfe8bcd1ccac7bb1a42c9b183eb6a8615886`.
- Those 24 rows span seven meetings; all candidate agents in each meeting share the same single mixed-audio SHA-256. Receipt status remains `NOT_ADJUDICATED`.
- Existing visual contact sheets were viewed for train rows only. They show a mixture of clear, small, and absent target people; they do not prove who is speaking or establish lip synchronization.
- Official AMI sources: [audio/video signals and channel mappings](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml), [meeting-room synchronization](https://groups.inf.ed.ac.uk/ami/corpus/meetingrooms.shtml), [known data problems](https://groups.inf.ed.ac.uk/ami/corpus/dataproblems.shtml), [download index](https://groups.inf.ed.ac.uk/ami/download/).
- Official University of Edinburgh mirror: [`edinburghcstr/ami` pinned revision](https://huggingface.co/datasets/edinburghcstr/ami/tree/46f28f2503e2ec48f8867a84eef356c70476beab). Read-only Viewer discovery scanned 339 train rows; the acquired row and decode receipt are in `C:\CodexArtifacts\pretrained-reuse-r2\ami-hf-channel-smoke-20261022-attempt-03\receipt.json` (audio SHA-256 above).

No annotation or media outside the already acquired selected SA/SB subset was
opened; the one additional audio segment came from the official train split.
No predictions, fitting, checkpoint selection, audit evaluation, feature
extraction, GPU process, or existing artifact was touched. The current GPU
remains occupied by PID 12596 (15,821/16,384 MiB at the final read-only check).

## Decision

Do not describe the current mixed-audio input as speaker-isolated. The
individual-channel contract is now demonstrated for one hash-pinned train
row; before model comparison, extend the acquisition to the frozen candidate
cohort, verify every channel-to-participant mapping, and freeze missing/overlap
handling. Alternatively, validate target-speaker dominance for mixed-WAV
crops. The source remains a **conditional feasibility candidate**, not an
admitted quality benchmark. The existing five development parent groups remain
low-power feasibility data, not a 95–98% product-quality gate. No product or
modality claim follows from this preflight.
