# AMI audiovisual speaker-binding preflight — 2026-10-22

## Decision

AMI is a promising, rights-cleared source for a narrow audio-to-visible-speaker
binding task, but it is **not yet admitted for a quality comparison**. The
small existing train cohort proves that the selected participant-channel
acquisition path can decode and join to synchronized camera frames. It also
contains target speakers who are absent or poorly visible in some frames. A
new, scene- and speaker-disjoint candidate split has enough audio metadata to
continue, but its videos could not be fetched in this environment. No model
training, feature extraction, validation scoring, or sealed-audit access was
performed.

## Task candidate

Given a short participant-channel speech segment and four simultaneous
participant close-up streams, select the stream showing the speaker. Candidate
order should be randomized and the output must be a probability distribution
over the supplied four choices. Train and validation are grouped by meeting
and use disjoint participant identities. This tests a specific cross-modal
binding relation; it is not general AV understanding. Audio-only voice identity
and visual-only mouth-state controls are still required, as are temporal shift,
stream shuffle, missing-stream, and participant-order controls.

## Existing train-only wiring check

At the pinned University of Edinburgh dataset revision
`46f28f2503e2ec48f8867a84eef356c70476beab` (`ihm`, `train`), 20 selected
participant-channel audio rows covering 17 of 24 previously frozen train clip
IDs were fetched. Adjacent source rows were retained where a frozen clip crossed
a row boundary. All 20 payloads decoded as mono 16 kHz WAV, totaling 3,397,640
bytes. The previously exposed adjacent row 16969 was excluded. No transcript
was saved or used.

For each selected audio interval, the speaker-to-close-up mapping was read from
the existing AMI `meetings.xml`; all four camera frames were decoded at the
interval midpoint. All source-video hashes matched prior local media receipts.
The observed PTS deltas were at most 0.040 seconds. Manual inspection found
several clear target views but also absent/poorly visible people, so the camera
mapping and timestamp alone are not an adequate positive-label gate. These
previously seen meetings are only a plumbing check and are excluded from the
new candidate split and from any model fit or validation metric.

Receipts:

- Audio fetch: `C:\CodexArtifacts\pretrained-reuse-r2\ami-hf-selected-train-audio-attempt-06\audio-receipt.json`, SHA-256 `bdd5de50892445d8e3cfdc4e404a767c47a5e6a47259bee2add027317600e6b4`.
- Audio/video timestamp join: `C:\CodexArtifacts\pretrained-reuse-r2\ami-hf-av-sync-check-attempt-07\sync-receipt.json`, SHA-256 `73289f21ae79ada796dbd863ddd2388b2710a72c65fe2a903796c8fd33420ab9`.

## New train-only metadata split

A pinned footer-only index of the official `ihm/train` parquet shards covers
108,502 rows at the same source revision. Only six newly selected meetings were
read at the row-group level, and only `meeting_id`, `audio_id`, `begin_time`,
`end_time`, `microphone_id`, and `speaker_id` were requested. No `text` or
`audio` parquet column was read and no media was fetched for these six meetings.
The resulting 4,233 audio-segment metadata rows are partitioned as follows:

| Experiment split | Meetings | Metadata rows | Rows >= 1 s | Rows >= 2 s | Rows >= 5 s |
|---|---|---:|---:|---:|---:|
| Train | TS3009a, ES2010b, IS1004b | 1,850 | 1,243 | 897 | 363 |
| Validation | TS3005b, ES2008b, IS1006b | 2,383 | 1,500 | 1,053 | 455 |

Participant IDs have zero overlap across these two groups. These are three
meeting families per split, not 4,233 independent scenes; uncertainty must be
clustered by meeting and participant. The table is a candidate inventory, not
a frozen or approved evaluation corpus. A visibility/overlap gate and adequate
independent group coverage have not been established.

The complete row-level metadata remains outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\ami-hf-ami-av-pair-metadata-attempt-09\selected-meeting-metadata.json`
(SHA-256 `3f7a18daaee4864b71279335b263df02919c5912658da97175b68ecdcb6b38e8`).
The all-train footer index is at
`C:\CodexArtifacts\pretrained-reuse-r2\ami-hf-train-meeting-index-attempt-08\train-meeting-index.json`
(SHA-256 `27e04c2e23215eb70ea0c90920f32233ef7dbaf42207e70512b35a72210dc2b6`).
A compact summary is checked in alongside this report.

## Blocking conditions and next gate

The official AMI media host timed out on both HTTP and HTTPS HEAD checks from
this local environment. No video bytes were downloaded. At the time of the
check, C: had 2,245,009,408 bytes free. The official download page estimates
400 MB per meeting for the four low-size DivX camera videos; six meetings would
exceed current free space. Its listed 40 MB RealMedia option is not yet shown
to expose all four synchronized close-up candidates, so it cannot be substituted
without a stream-layout and decode check.

Before any training:

1. Obtain the six new meeting videos through an accessible official source with
   hashes and licensing recorded, or reject this candidate if that cannot be
   done within local storage.
2. Verify per-meeting camera mapping, synchronization, and target visibility
   using a fixed, label-blind frame review rule; retain all failures in the
   coverage accounting.
3. Build the paired train/validation corpus from the selected participant
   channels, preserving whole-meeting and participant separation; add enough
   independent meeting families to support grouped uncertainty estimates.
4. Freeze the corpus, baseline controls, and task metrics before fitting any
   small readout. Keep the existing AMI/Greatest-Hits validation IDs and all
   sealed audit data out of selection.

The local RTX 3080 remained occupied by another process (PID 22868, about
15.8/16.4 GiB), so no GPU work was started. Cloud compute and purchases were
zero. No existing model, corpus, checkpoint, teacher, or live process was
modified.

## Sources

- [Official AMI signals and channel mappings](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml)
- [Official AMI synchronization and cameras](https://groups.inf.ed.ac.uk/ami/corpus/meetingrooms.shtml)
- [Official AMI download sizes and CC BY 4.0 release](https://groups.inf.ed.ac.uk/ami/download/)
- [Pinned University of Edinburgh AMI dataset revision](https://huggingface.co/datasets/edinburghcstr/ami/tree/46f28f2503e2ec48f8867a84eef356c70476beab)
