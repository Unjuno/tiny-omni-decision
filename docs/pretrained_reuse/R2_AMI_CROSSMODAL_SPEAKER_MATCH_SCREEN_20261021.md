# AMI audio/video source and task screen — 2026-10-21

## Expanded official-source follow-up — 2026-10-21

To check whether the first one-meeting-per-partition screen had too little data,
we selected two additional SA parent sets (ES2005a, ES2006a) and one additional
SB parent set (ES2011a) through the official selector. The combined sample now
contains three unique SA parent sets and two unique SB parent sets. Only each
selected meeting's four Closeup AVI streams and `Mix-Headset.wav` were downloaded;
optional views and all SC media/annotation members were excluded. The 25 files
total 825,256,200 bytes. SHA-256 values, selector-manifest hashes, and the
selective annotation extraction receipt are stored outside Git under
`C:\\CodexArtifacts\\pretrained-reuse-r2\\ami-av-probe-20261021`.

The selected meeting metadata lists 12 SA participants and 8 SB participants,
with no participant ID overlap across these selected groups. This is only a
selected-subset check, not a full split-wide identity/media proof. All 20 video
files opened and decoded sample frames at the beginning, middle, and 90% point;
the five mixed-headset WAVs are mono 16 kHz PCM. Full-stream decode and exact
audio/video clock synchronization remain unverified.

Applying the same word-window screen to all five meetings yields 32 preliminary
SA windows and 38 preliminary SB windows. ES2005a supplies only 2 of the 32 SA
windows: its WAV is 477.88 s, its four videos are 467.96–473.88 s, while the
meeting metadata reports 958 s. The media clock is internally close for that
meeting, but the large metadata discrepancy needs source/timing clarification.
This small, uneven window set is insufficient for a meaningful quality run and
does not justify starting a GPU process. No model was trained or evaluated.

| Meeting | Split | Audio (s) | Closeup video range (s) | Metadata (s) | Preliminary windows |
| --- | --- | ---: | ---: | ---: | ---: |
| ES2002a | SA | 1,272.64 | 1,230.40–1,248.72 | 1,242 | 16 |
| ES2005a | SA | 477.88 | 467.96–473.88 | 958 | 2 |
| ES2006a | SA | 1,284.34 | 1,275.76–1,280.16 | 1,290 | 14 |
| ES2003a | SB | 1,139.77 | 1,125.80–1,128.28 | 1,124 | 22 |
| ES2011a | SB | 1,113.85 | 1,113.96–1,114.32 | 1,128 | 16 |

The expanded receipts are:

- `media-download-receipt-expanded.json` — SHA-256
  `3bb1bb16dde2f73aa227e68d371ddaa64ec967d966e14b7b834b580589b451db`
- `media-and-annotation-preflight-expanded-20261021.json` — SHA-256
  `bbead20d61d57429cbf7a5358bfb5ac890e6da108a35c20e6ad8621cd67a9b4b`
- `annotations-expanded-20261021/extraction-receipt.json` — SHA-256
  `8f82eade6d08f098d61d16e1a815f8f028a019ec83317b5077c595817974deb0`

## Additional SB parent-set screen — TS3004a — 2026-10-21

To broaden the selected SB screen, TS3004a was selected in the official AMI
chooser. The chooser-generated response returned to the previous IS1008a
selection instead of TS3004a, so its manifest was **not** applied to TS3004a.
For this meeting, only the five policy-approved URLs (Closeup1–4 and
Mix-Headset) were fetched directly from the official AMI mirror after independent
HEAD checks. The selector page itself was unresponsive during follow-up. All
five files match their official `Content-Length`; the media receipt is
`media-download-receipt-ts3004a-20261021.json` (SHA-256
`2469bba07b96f36b4bc0531a6233461c11f6206c23c9258475f6cbada0932fdb`). The
selected four word files and four segment files were selectively extracted from
the already-hashed manual v1.6.2 archive; no SC member was opened. The
extraction receipt SHA-256 is
`1f9d834bc58ddedc79c84d2009ed6b4a84a61dcf87674a7fb645fa04a5ce5ae3`. The complete preflight, including media and
annotation hashes, is `preflight-ts3004a-20261021.json` (SHA-256
`94148155c34648a48ccf8f22900a03d2eae81be27bd4fb23a7b7033ac63e3452`).

TS3004a is in SB/development. Its four participant IDs do not intersect the
previously selected ES2003a/ES2011a SB subset or selected SA subset. This is
still not a split-wide proof. Applying the same preliminary word-window filter
adds 30 candidate windows (20 from agent A, 4 each from B/C, and 2 from D),
bringing the selected SB screen from 38 to 68 windows. This remains heavily
imbalanced by speaker and is not a frozen task set or quality metric.

All four TS3004a AVIs fully decoded to within exactly one frame of the
container-declared frame count (declared/decoded: 33338/33337, 33307/33306,
33267/33266, 33452/33451). The decoded frames span 1330.68–1338.08 seconds;
the mono 16 kHz headset WAV is 1345.323 seconds and the meeting metadata says
1345.324 seconds. This close duration agreement is useful but does not establish
timestamp synchronization or prove the forced-alignment word labels match visible
speakers. The standalone TS3004a preflight receipt hash is
`94148155c34648a48ccf8f22900a03d2eae81be27bd4fb23a7b7033ac63e3452`.

No GPU run was started. The selector-generated IS1008a manifest remains
preserved and must not be mistaken for a TS3004a manifest.

## Remaining SB parent-set screen — IS1008a and TS3006a — 2026-10-21

The two remaining official SB parent sets were added using only Closeup1–4 and
the common Mix-Headset WAV. IS1008a has a selector-generated manifest
(`amiBuild-224046-Tue-Oct-20-2026`, SHA-256
`94c72e0a13ca99823062c19b96f5b521bb50bf1a6c8e8f915680fd8c45fecfd0`). The
selector was not successfully queried for TS3006a in this pass; its five
official mirror URLs were independently HEAD-checked (HTTP 200) before download.
All ten media files matched their HEAD content lengths and were hashed. The
combined media receipt SHA-256 is
`d68f6feb9cd8bff843e8b77a1bf26662398c573ce392a57061ffd2618773922f`; media
bytes total 380,769,551. Only the sixteen selected word/segment XML members for
these two meetings were extracted from manual annotations v1.6.2. Their receipt
SHA-256 is `cf5d4a2ff904cecaf8ad72231b367e943b4dd170e301c3955db746827aca66cb`.
No SC signal or annotation content was accessed.

The same preliminary filter adds 16 windows for IS1008a and 27 for TS3006a,
bringing the five selected SB parent sets to 111 windows against 32 selected SA
training windows. IS1008a is imbalanced (9 agent-A, 4 agent-D, 3 agent-C, 0
agent-B); TS3006a is less skewed (10/8/5/4 across agents A/D/C/B). These rows
are not frozen examples and the counts do not establish answer quality. All four
AVI files in each meeting decode to one frame below the container declaration.
IS1008a audio is 943.833 s while its video range is 926.32–939.72 s and metadata
duration is 931 s. TS3006a audio is 1252.864 s while video is 1241.12–1248.12 s
and metadata duration is 1252.865 s. This is duration consistency only, not a
clock-sync validation.

One stronger split result is now available without reading any SC content. The
existing manual-annotation `corpusResources/meetings.xml` contains official
speaker-to-global-participant metadata for 98 SA and 20 SB meetings. Across all
of those SA/SB metadata records, there are 100 unique SA and 20 unique SB
participant IDs, with an empty intersection. The metadata file SHA-256 is
`8ab6cdcf03ed863e839418f31c7380392397a2eed892d73b9bc1b438c7ec520a`; the audit
receipt SHA-256 is
`b14e1d787e409ccbcb13e3124dc839f7a27fc3ab6e16b92104cb5e162e462fcb`. This
closes participant-ID split separation at the official metadata level. It does
not prove media identity/near-duplicate separation, synchronization, visibility,
or label correctness.

The combined two-meeting preflight (including all video decode counts, durations,
candidate windows and media hashes) is
`preflight-is1008a-ts3006a-20261021.json`, SHA-256
`9fe028d09e7b8d69a4e8749f61a788bbbba0de7a4aa59c91f2c2a8a8096698d8`. The
selection provenance note SHA-256 is
`283446a3ab2073f66ab3170fbde495f1e2df02b6973a074c19d791bb1ec082a0`.

At the point immediately after this SB expansion, the SA pool still contained
only 32 preliminary utterance windows from three selected SA sets. The later
IS1000a–d section below records the subsequent train-only coverage expansion.
The 111 SB windows remain weakly labeled. Timestamp synchronization and
visible-speaker labeling remain unverified; participant separation is only one
part of split integrity. No encoder, training, or quality-evaluation process
was started; the GPU was not used.

## SA training-coverage expansion — IS1000a–d — 2026-10-21

To address the 32-window SA training bottleneck, the metadata-only inventory
read word XML members from the 98 SA sessions only (no SB/SC words or signal
content) and ranked parent sets by the same preliminary window rule, using
meeting duration as a loose upper bound. IS1000 was the top parent set at 182
possible windows across its four sessions, with counts by session of 36, 31, 32
and 83. The ranking is only a source selection screen; actual media durations
must gate final windows.

All four IS1000 sessions are in SA/training, so the complete `a`–`d` parent set
was selected together. Only Closeup1–4 and Mix-Headset for each session were
downloaded: 20 files, 1,435,495,016 bytes. All 20 matched official HEAD content
lengths; the local SHA-256 receipt is
`media-download-receipt-is1000-20261021.json` (SHA-256
`11c7d6b306b4da2a36e7d2964da8d2591d9017ef440aeaac1491c251c4adc844`). The
official HEAD preflight SHA-256 is
`6ff6f95fb217f2fe48cf3b4f991b71b276b7a232780725a3e47e08ee925581f4`. Only the
32 IS1000a–d word/segment annotation members were extracted; receipt SHA-256
`396a2f727d991561035615f118e15cd15fafadada5c0386430c8d1602931be75`. No SC
annotation or media content was accessed.

Full decode and window recomputation retain 182 preliminary windows. For each
meeting, every camera stream decodes to one frame below its container's declared
frame count, consistent with the previously observed terminal-frame behavior.
Audio/video duration differences are small for IS1000b/c/d; IS1000a audio is
1582.67 s while the four videos span 1504.36–1579.56 s. The resulting candidate
window counts by meeting and speaker are in
`preflight-is1000-20261021.json` (SHA-256
`41f1084d933f0045446c968dbf66911b3fab7eb36e52b3509d4eb769f90e6b73`). A
An initial audit using the largest single segment interval flagged two windows;
that method undercounted candidates spanning adjacent speech segments. A corrected
union-overlap audit finds 181/182 IS1000 windows at or above 80%, with one IS1000c
agent-B window at 76.7% (1664.2–1666.2 s) left unresolved. The full SA audit
passes 213/214 candidates, including all 32 earlier candidates. The external
replay script is `C:\CodexArtifacts\pretrained-reuse-r2\ami-av-probe-20261021\audit_sa_word_segment_joins_20261021.py`;
it reads SA preflight windows and corresponding SA segment XML only. Script
SHA-256:
`EBBF2FA483F55A06391B506F09290306D1B40E8E8F2E5390C440BAAD6F1EED89`; receipt
SHA-256: `E457BC9FF7D940F385D9A153E7441A634E1D661480DB905BA1C2F2A1F3330A19`.

Across the current selected source screen there are now 60 media files and
2,891,874,753 bytes, all rehashed against their receipts. Preliminary windows
are 214 train (32 earlier + 182 IS1000) and 111 validation across all five SB
parent sets. This materially improves train coverage, but still does not
establish visible-speaker correctness or clock synchronization. The resulting
speaker-matching corpus remains a source candidate, not an admitted quality task;
no model training or evaluation has started.

The official scenario-only split assigns ES2002/2005/2006 to SA and ES2003/2011
to SB; the other official SB parent sets are IS1008, TS3004, and TS3006. We have
not accessed SC media or annotations. The independent-test and privacy/governance
gates remain open. The recommended scenario-only groups are listed on the
[official AMI split page](https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml).

## Decision

Keep the AMI Meeting Corpus as a **review-only candidate** for a narrow,
meeting-local audio/video task. The official corpus and annotation pages state
CC BY 4.0, and the recommended scenario-only split separates train (SA),
development (SB), and unseen evaluation (SC) by parent recording sets. The
current screen contains four selected SA parent sets (ES2002, ES2005, ES2006,
and IS1000) and all five SB parent sets; it includes all `a`–`d` submeetings for
IS1000 and one selected submeeting from each other set. Global official metadata
shows no participant-ID overlap between all 98 SA and 20 SB meetings. Hashes,
selected word annotations, and camera/channel mappings are recorded in
external receipts. The candidate is **not admitted** for training or quality
claims: stream synchronization and visible-speaker labels have not been
validated, word times are alignment-derived, media/near-duplicate overlap and
privacy/governance remain open. No SC annotation member was extracted or read.

## Candidate Decision task

Given a short voice reference from a meeting's mixed headset audio and four short
candidate face-camera clips from a different time in that same meeting, return a
calibrated four-way probability over which visible participant matches the voice.
The options are video clips, randomly ordered; the input contains no participant
IDs, role names, transcript, subtitles, per-person lapel channel, or target-time
audio. The candidate is a component task for cross-modal identity association,
not general meeting understanding or biometric identification across recordings.

This construction makes the modalities structurally complementary: audio-only has
no visible candidates, while video-only has no voice reference. It still needs
empirical controls: a model might exploit camera layout, repeated session structure,
option order, or recording artifacts. Same-time active-camera attribution is a
lower-risk alternative, but video-only mouth-motion cues may solve it; require
audio-only, video-only, and shuffled-alignment controls before claiming both inputs
are useful. Do not claim useful fusion from input sensitivity alone.

Potential labels can be derived from manual speaker-tagged word intervals and the
official meeting-level camera/audio-channel map. The official transcription page
says transcripts are synchronized to audio and speaker-differentiated by microphone
channel ID, while word/phoneme timings are automatically generated by forced
alignment. Treat this as noisy weak supervision, not manually verified audiovisual
identity labels. Use only a common mixed headset channel for the voice reference;
never use individual lapel/headset channel identity as the answer cue. Require two
non-overlapping utterances by the same speaker, and require all candidate clips to
be duration-matched, from the same meeting, and sampled with a fixed rule. Verify
alignment visually/audio-wise on TRAIN examples only before admitting generated
rows. Do not use `participants.xml`, demographic fields, speaker embeddings, face
recognition, or cross-meeting identity matching.

## Rights, split and task-integrity evidence

- The official license page says the AMI Meeting Corpus and its annotations are
  released under CC BY 4.0. The official download page separately lists manual
  annotations v1.6.2 (22 MB) and automatic annotations v1.5.1 (68 MB), both marked
  as CC BY 4.0. Retain required attribution and the exact upstream release/hash if
  the source advances beyond this screen.
- The recommended scenario-only split lists SA as 98 training meetings in 25 sets,
  SB as 20 development meetings in 5 sets, and SC as 20 unseen evaluation meetings
  in 5 sets. A set is the four submeetings `a`–`d` from one recording session. Keep
  the full set together. Use SA for train and SB for validation; leave SC and all
  its labels untouched. For selected meetings ES2002a (SA) and ES2003a (SB), the
  participant-ID intersection is empty and each has four speaker-to-camera/channel
  mappings. This covers one parent set per partition, not split-wide separation.
- The official signal documentation reports 16 kHz audio, a mixed headset recording,
  repaired synchronized video, and a machine-readable camera/channel-to-participant
  mapping in `corpusResources/meetings.xml`. Locally, both headset WAVs are mono
  16 kHz PCM and all eight AVI files are MPEG-4 at 352×288 and 25 fps. Start/mid/90%-
  point samples decoded in every selected file; this is not a full decode or a sync
  proof.
- ES2002a has 1,272.64 s of audio and 1,230.40–1,248.72 s across four camera streams
  (meeting metadata: 1,242 s). ES2003a has 1,139.77 s of audio and 1,125.80–1,128.28
  s of video (metadata: 1,124 s). A word interval in ES2003a extends beyond the
  shortest camera stream. Exclude unsafe boundaries and verify sync before clipping.
- The official chooser estimated 636 MB for the chosen meeting/stream selection.
  We downloaded only four close-ups and one headset mix per meeting: 373,953,060
  bytes total. Other streams were not downloaded.
- The official transcription page says each meeting has a word-level NXT transcript
  synchronized to audio, but the word/phoneme timings are forced-alignment outputs;
  it also lists known transcript/transform errors. Selected archive contents also
  include speaker segment XML with `transcriber_start`/`transcriber_end`. The exact
  relation between these segment intervals and the media clocks still needs QA.
- The consent/ethics page describes the original project review process and
  provides a blank participant consent form. Public release terms do not remove
  privacy concerns around recognizable faces and voices. Keep any future probe
  within meeting-local matching, avoid publishing media or identity outputs, and
  complete a purpose-specific governance review before any product claim.

## Acquisition and current blocker

Across the selected SA/SB meetings, 60 media files total 2,891,874,753 bytes.
All 60 were rechecked against the external download receipts (size and SHA-256);
no partial-download files remain. The 97 distinct selected annotation members
total 5,173,336 bytes and were also rechecked against their extraction receipts.
The original 22,887,865-byte manual annotation v1.6.2 archive remains outside
Git. Only selected SA/SB members and the global `meetings.xml` metadata were
read; no SC annotation or signal content was opened, and no third-party mirror
was used.

The preliminary filter produces 214 train and 111 validation candidate
windows. A corrected union-overlap audit finds 213/214 train candidates at or above the
80% same-agent segment-coverage threshold. One IS1000c agent-B window remains
below threshold (76.7%) and is quarantined pending review; all 32 earlier train
windows pass. These counts are not a frozen task set or quality results.
Audio/video clock alignment, visible-speaker truth, media/near-duplicate split
isolation, and privacy/governance remain open.

## GPU handoff review — HotUGPU

The current observation was 15,821/16,384 MiB used on the RTX 3080 Laptop GPU,
with Orbitune CUDA worker PID 24160 active (27% utilization). I left its process,
watchdog, sentinel, run files, and checkpoints untouched. I reviewed the
cooperative wrapper `scripts/run_with_hotgpu_guard.py` from the sibling
`codex/r2-clevrer-coverage-20261020` worktree at commit
`e914ba220ba1dbbb0822af4933db79638354f364`; the wrapper file SHA-256 is
`ae3ca95654a2f57e6652455f9cf3cf6a9639ec0f456d4f24318ce146130ecde8` and its
source commit is `2b052cf939485ba0c17d1848ad7e18563079ae1c`.

The wrapper initializes a small CUDA guard, waits until external CUDA PIDs are
gone and utilization is below 5%, reserves VRAM so the Orbitune idle-start gate
stays closed, then launches the child. Its CPU-only regression tests passed
(10 passed). It does not terminate or signal Orbitune; the existing idle
supervisor sees the foreign CUDA process and performs its own cooperative yield.
The AMI screen worktree does not contain this wrapper, so any future GPU command
must invoke the reviewed copy or add the pinned implementation first. No GPU
child was launched: the source-admission and synchronization gates are still
open, so there is not yet a justified encoder run.

## Admission gates before any model run

1. Four SA parent sets and all five SB parent sets are selected. The screen
   contains ES2002a, ES2005a, ES2006a, all IS1000a–d meetings, and one selected
   submeeting from each SB parent set. Keep parent sets together and do not
   download or inspect SC signals, annotations, or predictions.
2. Join every selected annotation interval to a valid speaker and camera mapping;
   reconcile audio/video durations and synchronization; full-decode selected files;
   and retain hashes. The 28 streams from TS3004a, IS1008a, TS3006a, and IS1000a–d
   each decode one frame below AVI declarations; the earlier five meetings have
   sampled decode only.
3. The metadata-only global participant-ID check passes for all 98 SA and 20 SB
   meetings. Still prove zero meeting-session, media, normalized-content, and
   near-duplicate overlap. Exclude the task if split integrity or actor separation
   fails.
4. Freeze a balanced task manifest and answer-generation rule before encoding.
   Include same-meeting hard negatives, randomized candidate order, temporal
   non-overlap between reference and candidate clips, and task-leakage checks.
5. Compare audio+video with structurally valid audio-only, video-only, and shuffled
   alignment controls. Report per-meeting and per-set Accuracy, NLL, Brier, ECE,
   uncertainty by parent set, coverage, and calibration. Even all five SB groups
   are a feasibility screen, not a strong independent quality claim or a 95–98%
   product benchmark.

## Result of this screen

AMI remains a **review-only source candidate**, not an admitted training or
evaluation dataset. Four SA parent sets and all five SB parent sets have been
screened; preliminary counts are 214 train candidates and 111 validation
candidates. A corrected union-coverage join audit passes 213/214 train windows;
one IS1000c agent-B window remains quarantined at 76.7%. Global SA/SB
participant-ID disjointness is verified from
official metadata. Next validate timestamp synchronization and visible-speaker
labels on train-only samples, and audit media identity/content overlap. Stop
this source path without training if these gates fail. No Teacher, checkpoint,
sealed audit, existing corpus, or model artifact was modified.

## Official sources

- [AMI Meeting Corpus home](https://groups.inf.ed.ac.uk/ami/)
- [AMI corpus license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
- [Official download page](https://groups.inf.ed.ac.uk/ami/download/)
- [Official manual annotation archive v1.6.2](https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip)
- [Recommended training/development/unseen split](https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml)
- [Audio/video signal formats and camera/channel mapping](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml)
- [Transcription format and timing caveats](https://groups.inf.ed.ac.uk/ami/corpus/transcription.shtml)
- [Ethics and consent information](https://groups.inf.ed.ac.uk/ami/corpus/ethicsandconsent.shtml)
