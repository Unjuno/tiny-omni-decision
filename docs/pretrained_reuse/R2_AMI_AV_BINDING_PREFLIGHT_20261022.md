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

## Official source-quality notes

The official AMI data-problems table flags `ES2008b`: its video files end at
about 34:45 (2,085 s), while audio continues to about 37:11. The metadata
receipt contains 688 candidate utterances for this meeting. By those published
endpoints, 57 utterances begin after video ends, one crosses the endpoint, and
630 end before it. Of the 430 utterances at least one second long, 397 are
fully inside the video interval. These counts are metadata-only and do not
establish that all four close-ups are decodable or that the speaker is visible.
Any later frozen corpus must either predefine and apply the common audio/video
intersection or drop the affected intervals with an auditable accounting;
audio-only tail rows cannot be scored as AV binding examples.

The same official data-problems table states that TNO (`TS*`) video is not
perfectly synchronized but was manually processed to acceptable synchronization
quality, and that E/I (`ES*`/`IS*`) frame drops were processed and are now
synchronized with audio. This is source-level assurance, not a substitute for
checking the exact six selected media files and their actual PTS after decode.
The table lists no additional issue for the other five selected meetings in
its searchable entries; absence from that table is not proof of complete media
or unobstructed target visibility.

The pre-existing local `meetings.xml` resource also resolves participant to
microphone-channel/close-up mapping for all six candidate meetings. Its
SHA-256 is `8ab6cdcf03ed863e839418f31c7380392397a2eed892d73b9bc1b438c7ec520a`.
This is a mapping receipt only; its prior acquisition provenance and exact
release lineage were not independently established in this pass.

| Split | Meeting | Participant → channel → camera |
|---|---|---|
| Train | TS3009a | MTD033PM:0:Closeup2; MTD034ID:2:Closeup4; MTD035UID:1:Closeup3; MTD036ME:3:Closeup1 |
| Train | ES2010b | FEE037:0:Closeup4; FEE038:1:Closeup2; FEE039:2:Closeup3; FEE040:3:Closeup1 |
| Train | IS1004b | MIO047:3:Closeup4; MIE090:1:Closeup2; MIO019:0:Closeup1; MIO022:2:Closeup3 |
| Validation | TS3005b | MTD017PM:0:Closeup2; MTD018ID:2:Closeup4; FTD019UID:1:Closeup3; MTD020ME:3:Closeup1 |
| Validation | ES2008b | FEE029:0:Closeup4; FEE030:1:Closeup2; MEE031:2:Closeup3; FEE032:3:Closeup1 |
| Validation | IS1006b | MIO008:3:Closeup4; MIO040:1:Closeup2; FIO041:0:Closeup1; MIO078:2:Closeup3 |

The channel-to-camera order varies by meeting, so a global mapping must not be
assumed. The corpus builder must join the meeting-specific mapping by speaker
identity and preserve candidate randomization. A local row-level join check
found all 4,233 metadata rows mapped to one of the four participants in their
meeting; all six meetings had four unique metadata speakers and zero unmapped
rows. This check inherits the local XML lineage limitation above.

## Blocking conditions and next gate

The official AMI media host timed out on HTTP and HTTPS requests from this
local environment; an IPv4 TCP check to its resolved address also timed out.
The ordinary Edge browser navigation to the index timed out, so no video bytes
were downloaded. The full low-size DivX AVI set is large (the download page
quotes an average 400 MB per meeting), but a smaller candidate route exists:
official mirror indexes list per-camera `Closeup1.rm` through `Closeup4.rm`
files. Examples show 8.2 MB per closeup for TS3004a, 7.7 MB for IN1009, and
23 MB for IN1014. The download page describes RealMedia as low-bitrate combined
A/V. This indicates a possible four-camera route that fits current disk headroom
(2,238,205,952 bytes), but the exact six selected meeting files, payload hashes,
codec support, camera coverage, and synchronization have not yet been verified.
The network timeout remains the active blocker; storage need not block the small
RealMedia experiment if those streams are available.

Before any training:

1. Retry the official per-camera RealMedia streams when the host is reachable;
   pin exact file sizes/hashes and confirm all four streams exist for the six
   selected meetings. Reject the route if its camera layout or license cannot be
   verified.
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
- [Official AMI data-problems table, including ES2008b video truncation and synchronization notes](https://groups.inf.ed.ac.uk/ami/corpus/dataproblems.shtml)
- [Official TS3004a per-camera media index showing AVI and low-size RM streams](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/TS3004a/video/)
- [Official IN1009 per-camera media index showing 7.7 MB RM streams](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/IN1009/video/)
- [Pinned University of Edinburgh AMI dataset revision](https://huggingface.co/datasets/edinburghcstr/ami/tree/46f28f2503e2ec48f8867a84eef356c70476beab)
