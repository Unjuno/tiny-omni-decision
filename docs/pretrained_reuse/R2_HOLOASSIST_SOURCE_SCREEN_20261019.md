# HoloAssist source screen — 2026-10-19

## Decision

**Metadata candidate; selected media acquisition is in progress, model fitting is not yet admitted.**
HoloAssist has a useful physical-task decision target: fine-grained action
correctness, with instructor and performer correction labels, and synchronized
egocentric video, mono audio, and sensor streams. It can support video-only,
audio-only, and joint-input comparisons on a common action interval. This is a
much closer fit to a lightweight task-assistant Decision model than generic
audio-visual question answering.

At the initial source-screen, no media payload had been downloaded or decoded,
no test labels had been opened, no model had been trained, and no GPU had been
used. One necessary nuance: the range indexer
walked all archive headers to locate the allowed session members. Its 64 KiB
HTTP blocks may have included adjacent initial payload bytes; those responses
were not saved or decoded, and the exact train/validation/test byte overlap is
unknown. Test labels were not opened, test member names were not persisted, and
the test split is not used by the selector or pilot.

## Reproduced metadata facts

| Item | Result |
| --- | ---: |
| Annotation records / unique sessions | 1,758 / 1,758 |
| Official train / validation / test sessions | 1,466 / 207 / 438 |
| Session overlap between any split pair | 0 |
| Train / validation fine-action intervals | 124,759 / 17,905 |
| Unassigned annotation sessions excluded | 85 |
| Reversed fine-action intervals, train / validation | 4 / 1 |
| Out-of-range fine-action intervals, train / validation | 0 / 0 |
| Compressed-video tar | 155,285,145,600 bytes |
| IMU tar | 4,971,059,200 bytes |
| Train/validation MP4 members indexed | 1,673 files / 119,346,824,532 bytes |
| Sparse index HTTP range bytes | 291,063,808 bytes across 4,442 verified requests |

All train and validation session IDs have annotation records. The downloaded
annotation release contains no test annotation records; test session IDs were
read only from the official split list to check disjointness. Its five-way
action labels are heavily imbalanced: validation contains 16,966 Correct
Action intervals, 540 wrong actions corrected verbally by the instructor, 359
corrected by the performer, 11 not corrected, and 29 otherwise. A five-class
result would therefore be weakly supported. The first viable task is binary
correct-vs-wrong, excluding “otherwise” and invalid reversed intervals, with
the natural validation distribution preserved.

The exact input files, hashes, per-split ID-set hashes, event counts, and
policy are in the [machine-readable receipt](../../manifests/pretrained_reuse/holoassist-source-screen-20261019.json).
The audit is reproducible with:

```powershell
python -m scripts.audit_holoassist_metadata `
  --annotation C:\CodexArtifacts\pretrained-reuse-r2\holoassist-source-screen-20261019-01\data-annotation-trainval-v1_1.json `
  --splits C:\CodexArtifacts\pretrained-reuse-r2\holoassist-source-screen-20261019-01\data-splits-v1_2.zip `
  --output C:\CodexArtifacts\pretrained-reuse-r2\holoassist-source-screen-20261019-01\NEW_AUDIT.json
```

The audit refuses to overwrite output, checks exact session disjointness,
requires annotations for every train/validation session, and never reads test
annotation rows or media. It reports reversed action intervals without
clamping them.

## Rights, grouping, and measurement limits

The official project and data README state that the dataset is released under
CDLA-Permissive 2.0. The CDLA FAQ explicitly includes model training and says
the agreement does not restrict model “Results.” This supports training use
under the stated data license; no media is being redistributed here. The
paper's supplementary material describes IRB approval and participant
consent. The release metadata inspected here exposes `video_name`, `taskId`,
`batch`, and `taskType`, but no instructor, performer, speaker, or participant
identity. Consequently the supplied split is recording-disjoint, not
verifiably participant-disjoint. Any result would be a session-disjoint
development measurement and cannot claim unseen-person generalization.

The dataset page lists a 144.62 GB compressed-video archive and a 4.63 GB IMU
archive. A verified sparse index parsed 8,444 TAR headers and retained only the
1,673 official train/validation MP4 paths. It fetched 291,063,808 bytes of 64
KiB ranges, not the 155 GB media payload; because those blocks abut TAR headers,
some may contain adjacent payload bytes. The resulting train/validation MP4
member total is 119,346,824,532 bytes. Test paths were filtered before being
written to the index. This was a transport-level header walk, not a test-media
evaluation; exact incidental payload overlap is unknown and the test split
remains prohibited from all modeling.

The immutable seed-17 pilot selects 160 train sessions (8 per task type) and
all 207 official validation sessions. The compressed-only selection is
28,228,194,803 bytes; its parallel attempt-04 was stopped at 108/367 files
(8.28 GB completed, eight prefixes totaling 0.37 GB) after a paired AV probe
showed that compressed media has no audio. All completed files and partial
prefixes are preserved. A separate
immutable selection binds the same session order to
`video_pitch_shifted.tar` (197,783,674,880 bytes; ETag
`"0x8DC8C9DD88CD02E"`). Its verified sparse index scanned 8,444 TAR headers,
persisted 6,692 train/validation member paths (including 1,673 videos and no
separate audio members), and fetched 292,118,528 bytes in 4,458 ranges. The
index SHA-256 is
`ad5a3843e6592d4be8eab1b187f36ff69f4fab90a9be57c06be2c0ce5fd2f8dc`. The
fixed pitch-shifted selection contains 160 train / 207 validation sessions,
36,117,365,135 media bytes, and has SHA-256
`577adbafe6be869de3aff3b814acf91db7be582a42e0a7a83e7ee773dde0e8b7`.
Only allow-listed train/validation paths were persisted; no test media was
opened. The media downloader accepts the pitch-shift member suffix only when
explicitly selected; compressed-video behavior remains the default.

A single selected training session (`R130-25Aug-RAM`) has been downloaded and
fully probed. The pitch-shifted file is 132,569,947 bytes (SHA-256
`d4a6356ffecb8eb8198fb568e03df6e7130dbb161a275ca70a962af1c81bf890`) and
contains H.264 video (5,960 frames, PTS 0, 205.65550 s) plus mono 48 kHz AAC
audio (PTS 0, 205.67467 s). Full audio decode produced 9,872,384 samples, RMS
0.00413, peak 0.5567; it is not silent. The paired compressed video has the
same 5,960 frames, starts at PTS 0, and lasts 205.65908 s. Declared stream
duration differs by 14.5 ms; decoded AAC tail padding accounts for the longer
audio decode end. This is source-format/timing proof for one recording, not
task-quality evidence or proof that all sessions are aligned. The full fixed
36.12 GB pitch-shifted selection is now downloading into a separate attempt
directory with per-file range/ETag/size/SHA checks. The latest observed
progress is 201/367 files (20,413,622,420 bytes, observed 2026-10-19 00:09:04
UTC); this is not the final receipt. The new read-only
`scripts.audit_holoassist_download_progress` tool matched all 201 unique
journal records to the pinned selection, archive ETag, planned sizes/offsets,
expected output paths, and on-disk file sizes. It found zero errors and 8
in-progress partial files totaling 679,477,248 bytes; 166 selected files
remain unlogged. The tool does not re-hash media contents, and does not admit
the partial download as a validated media corpus. The saved report is at
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-download-audit-attempt-01\partial-audit-20261019T0008Z.json`
(SHA-256 `46668969BA0C4D684C7B2B420DBEB1DC55B14F1E77CBB9430B9560A26D40AC11`).
The repository extraction decoder was also exercised on the verified probe:
the first causal 30-second window yielded 480,000 finite 16 kHz audio samples
(RMS 0.00655, peak 0.405) and eight finite 504x896 RGB video frames. This
verifies timestamped PyAV decode/resampling and 8-frame selection for one
window only; it is not a full bank or encoder-quality result.

Full-corpus feature extraction, event decoding, and dataset training or task
quality evaluation have not run. C: had over 200 GB free; no cloud compute was
used. GPU execution was only through the local HotUGPU guard. The CPU JAX VideoPrism path decoded a
selected train MP4 and returned finite `[1, 2048, 768]` tokens. A later
pitch-shifted-video smoke sampled eight frames from the first 30 seconds and
returned finite `[1, 2048, 768]` tokens; the 504x896 decoded frames were
preprocessed to 288x288. Its cold compile+forward was 6.62s; steady-state
timing is not yet measured for that exact path. HotUGPU yielded the RTX 3080
for a Whisper Tiny 30-second smoke (`[1, 1500, 384]`, 0.49s, 227 MiB peak
allocation) and a 32-update synthetic readout smoke (379,652 trainable
parameters, 0.56s, 79 MiB peak). A separate encoder-only Whisper run used PyAV
resampling (no environment upgrade because `torchaudio` is absent):
`[1,80,3000]` to `[1,1500,384]`, about 0.73s forward, 182 MiB peak
allocation, 204 MiB peak reserved. MiniLM L3 loaded locally (17,389,824
parameters; revision `4ca70771034acceecb2e72475f72050fcdde4ddc`) and encoded the
fixed task query in 0.122s on CPU. The proposed pre-quantization configuration
uses encoder-only Whisper (8,208,384), not the full 37,760,640-parameter
Whisper Tiny model. Together with VideoPrism (114,365,184), MiniLM L3
(17,389,824), and this readout (379,652), the inventory is 140,343,044
parameters. This is not a packaged/runtime residency measurement; the
VideoPrism checkpoint alone is about 457 MB, so the provisional 200 MB bundle
goal is not met. These are throughput/integrity checks only, not task quality
or training evidence.

If all selected companion files pass integrity and synchronization checks, the
proposed task is
causal 30-second task-monitoring windows: a window is positive when any
fine-action ending inside it is annotated incorrect. Empty windows remain in
the recurrent history but are masked from loss and metrics. This preserves
temporal order while reducing repeated encoder work relative to per-action
encoding. A small text-conditioned audio/video gate and causal LSTM readout
have been added and CPU-tested; no quality result exists yet. Compare joint,
video-only, and audio-only controls on the same windows; preserve natural
validation prevalence and report Accuracy, NLL, Brier, ECE, PR-AUC, and
session-clustered uncertainty. Do not treat windows from the same recording as
independent replicates.

The frozen-feature packer and matched training runner are now implemented in
`scripts/pack_holoassist_feature_cache.py` and
`scripts/train_holoassist_temporal_readout.py`. The packer requires each
modality bank to match the exact train-then-validation window ID sequence,
keeps empty windows as unlabeled recurrent context, binds model revisions and
weight hashes, and rejects session/media overlap. The runner checks those
receipts against `configs/pretrained_reuse/holoassist_temporal_readout_v1.json`,
then fits video-only, audio-only, and joint 128-hidden-unit LSTM readouts with
seed 17 and 2,048 updates per arm. It does not early-stop; it saves both best
validation-NLL and final checkpoints, fixed-step predictions, and a history
every 64 updates. The selected checkpoint is reloaded and its saved validation
probabilities must match within 1e-6. This is a development-validation
selection protocol, not a blind evaluation. Synthetic CPU tests exercise the
packing, training, and checkpoint-reload code; no real frozen feature banks or
HoloAssist task-fitting run exist yet.

The frozen encoders are pinned to VideoPrism-B revision
`be719a406d563b66f0ac969e7c94bab8e997c81a` (8 frames/window), Whisper Tiny
encoder revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`, and MiniLM-L3
revision `4ca70771034acceecb2e72475f72050fcdde4ddc`. These encoder settings and
task text preprocessing are now explicit in the experiment config; the
preprocessing still must be exercised on the complete selected corpus before
the run can start.

The raw-media feature extractor is implemented in
`scripts/extract_holoassist_frozen_features.py`. It verifies the selected
media SHA-256 values, the pinned 160/207 selection hash, and the
AV-validation/event/window receipts before decoding, binds resumable progress
to the exact ordered IDs, windows, model and processor hashes, configuration,
source commit, script hash, software versions, and
device, and writes progress atomically. It requires the pinned VideoPrism
commit with no tracked source edits; existing untracked build/egg-info/cache
outputs are preserved and do not affect the source-tree check. Corpus, model,
and progress checks run before the CUDA path waits for a slot. CUDA audio uses
the shared HotUGPU preflight and fails closed on an occupied GPU unless the
reservation handoff is valid. CPU tests cover progress reload/mismatch and
the HotUGPU telemetry path. The decoder has only been smoke-tested on one
training session; no full feature bank has been produced.

The committed Whisper encoder path was then exercised end to end through the
shared HotUGPU wrapper on the same verified train recording and first 30-second
window. It produced one finite 384D vector in 0.547 seconds of extraction
loop time (190,619,136 bytes peak allocated, 211,812,352 bytes reserved); the
saved feature reloaded and matched SHA-256
`8532f85881c5c6cc65c54edec902077516b586b06a05b70c76f237417bfbfd92`. The full
GPU reservation was 3,200 MiB. A Windows allocator warning said
`expandable_segments` is unsupported; the run completed. This remains a
single-window systems smoke, not task-quality evidence or a real feature bank.
The output is preserved in
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-extractor-audio-smoke-attempt-01`.

Metadata-only window preflight on the fixed pilot yields 1,551 train / 1,977
validation windows (median sequence lengths 8 / 7). Of these, 1,464 / 1,896
contain at least one valid fine action; 376 / 544 are positive under the
“any incorrect action” rule (25.68% / 28.69% prevalence). Maximum sequence
lengths are 43 / 41. These counts are derived only from selected train and
validation annotation rows; actual media duration, audio availability, and
decoder alignment remain pending.

At 2026-10-19 00:12:34 UTC, the live download journal audit recorded 206/367
selected MP4s (20,969,834,469 completed bytes), 161 not-yet-logged files, and
eight partial files (926,941,184 bytes). No selection, archive identity,
offset, path, or on-disk-size mismatch was found; this audit did not recompute
media content hashes. PID 16496 remained active. The immutable audit snapshot
is `C:\CodexArtifacts\pretrained-reuse-r2\holoassist-download-audit-attempt-01\partial-audit-20261019T001234Z.json`
(SHA-256 `7E4F5546E53409BC7BD7B7C2C3A990CB8628F8541B921E43D83D9A79260513A7`).
Full AV validation and feature extraction remain gated on the complete selected
media set.

At 2026-10-19 00:19:40 UTC, the live download journal audit recorded 223/367
selected MP4s (22,974,607,734 completed bytes), 144 not-yet-logged files, and
eight partial files (591,396,864 bytes). The audit found no identity, offset,
path, or on-disk-size mismatches and did not recompute media hashes. PID 16496
remained active. Snapshot:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-download-audit-attempt-01\partial-audit-20261019T001939Z.json`
(SHA-256 `A62DE7DAFCDD2BB4B8790A819464A26948975DDF4560E5EF8F51C8423907D0FE`).
Full AV validation and feature extraction still require all selected media.
At 2026-10-19 00:22:26 UTC, the live download audit recorded 230/367
selected MP4s (23,492,216,471 completed bytes), 137 not-yet-logged files, and
eight partial files (687,865,856 bytes). No identity, offset, path, or disk
size mismatch was found; media hashes were not recomputed. PID 16496 remained
active. Snapshot:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-download-audit-attempt-01\partial-audit-20261019T002226Z.json`
(SHA-256 `4121454D97E8B5B2703BFEBAABC24A6CEA4146043222C2F9FFF18A250AA293D3`).

A second single-window Whisper smoke used selected train session
`R203-7Nov-Belt`, whose full file hash matched the download journal. The pinned
Whisper Tiny encoder produced a finite 384D feature in 0.513 seconds on the
RTX 3080 Laptop GPU (peak allocated/reserved 190,619,136/211,812,352 bytes).
HotUGPU waited through an active Orbitune compute PID, then reserved 3,200 MiB
for the child and released it on completion. No training or test data use.
Result JSON SHA-256 is
`9D20404B6425C4EB4F132D1AD3228870992243D67C5AAC3DEB78AA953BC30B3F`; output
is preserved in
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-extractor-audio-smoke-attempt-02`.
This is encoder plumbing evidence, not task-quality evidence or a real feature
bank.
At 2026-10-19 00:28:06 UTC, the live journal audit recorded 240/367 selected
MP4s (24,556,305,185 completed bytes), 127 unlogged files, and eight partial
files (931,135,488 bytes); no identity, offset, path, or size mismatch was
found, and content hashes were not recomputed. PID 16496 remained active.
Snapshot:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-download-audit-attempt-01\partial-audit-20261019T002806Z.json`
(SHA-256 `EEE008631CD41E158E07F279D0F621C8817B2D2A4A1481D99A3FB507B32E47B1`).
At 2026-10-19 00:38:51 UTC, the live journal audit recorded 263/367 selected
MP4s (27,208,386,839 completed bytes), 104 unlogged files, and eight partial
files (851,443,712 bytes). No identity, offset, path, or size mismatch was
found; content hashes were not recomputed. PID 16496 remained active.
Snapshot:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-download-audit-attempt-01\partial-audit-20261019T003851Z.json`
(SHA-256 `2BD4AA862975254FC0BD0900A69BCFA466D7F2A3297AA3F7D3623AC44009BB75`).
At 2026-10-19 00:48:24 UTC, the live journal audit recorded 287/367 selected
MP4s (29,851,608,731 completed bytes), 80 unlogged files, and eight partial
files (562,036,736 bytes). No identity, offset, path, or size mismatch was
found; content hashes were not recomputed. PID 16496 remained active.
Snapshot:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-download-audit-attempt-01\partial-audit-20261019T004824Z.json`
(SHA-256 `57DFAA1CAC7778E9797B74E24FD36FCCE136AB81DDE57923BC2EAB3ED141ABD5`).
At 2026-10-19 00:57:48 UTC, the live journal audit recorded 309/367 selected
MP4s (32,249,960,620 completed bytes), 58 unlogged files, and eight partial
files (398,458,880 bytes). No identity, offset, path, or size mismatch was
found; content hashes were not recomputed. PID 16496 remained active.
Snapshot:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-download-audit-attempt-01\partial-audit-20261019T005748Z.json`
(SHA-256 `04EFB4DD5F68D2BAFD9BD71B85F67679ADB04DC961C137EDE75B80A51EABBEDA`).
At 2026-10-19 01:05:07 UTC, the live journal audit recorded 335/367 selected
MP4s (34,018,286,048 completed bytes), 32 unlogged files, and eight partial
files (335,544,320 bytes). No identity, offset, path, or size mismatch was
found; content hashes were not recomputed. PID 16496 remained active.
Snapshot:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-download-audit-attempt-01\partial-audit-20261019T010506Z.json`
(SHA-256 `0E6A9AC682A0C859D01FD4A8A450205F54BD59EF288337A678D0FFB2F9159BB2`).
## Full media validation and timeline reconciliation

The selected pitch-shifted archive download completed with 367/367 files and
36,117,365,135 bytes. All 367 local media SHA-256 values matched the immutable
download manifest; train and validation contain 160 and 207 sessions. No test
media or labels were opened. Initial validation stopped on a 63.8 ms audio/video
endpoint difference, just above the old 50 ms rule. A read-only scan of all
selected files found zero start-time offsets, a maximum AV endpoint difference
of 64.6 ms (1.914 video frames), and a maximum annotation-to-audio endpoint
difference of 121 ms. The validator now accepts an AV endpoint difference up
to two measured video-frame intervals and checks annotation duration against
the audio endpoint, preserving the prior 150 ms annotation tolerance. This is
tied to measured stream timing and does not waive start alignment, stream
count, codec, channel, sample-rate, media-hash, or split checks.

Action construction now uses the validated video end rather than rounded
annotation duration. It excluded two validation intervals that ended after the
actual video; neither interval was truncated or relabeled. The output contains
12,954 train / 17,873 validation action events. Thirty-second causal windows
contain 1,550 train / 1,977 validation windows, including 86 / 81 unlabeled
context windows; 1,464 / 1,896 windows are labeled. Validation has 544 positive
and 1,352 negative labeled windows. No test data or media were used.

The validator, event corpus, and temporal-window receipts are preserved under
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-video-pilot-160-207-seed17-pitchshifted-attempt-01`.
Their SHA-256 values are respectively `ED8DBEB89A43EFFFCA709F969DE30AC308189C3CB5A06430014C84F16F25D090`,
`BB08BB9D1BAFBA843A0FBD66D0816F41262EBAA2CF27BA5F59804C679D59ABEA`, and
`7DE0224AE0739EEF67CCA74636D9F37949C7FB7D5EDCB74A406C47E0DAC8DB78`.
The media manifest SHA-256 is `E1BC330C0DB3325090FFC628C5CFCCBBD5DA9BC0D452A9129EC56172A66501D1`.
The corpus still requires frozen-feature extraction and matched readout
training; no HoloAssist task-quality result exists yet.

The pinned MiniLM text bank completed on CPU for all 3,527 train-then-validation
windows (3,527 x 384; 20 unique questions) in 2.55 seconds. Bank SHA-256:
`53BBCE5EF42345DBE7431FD2369766716B16E51105F8F4ADF5208B523492F9A3`.
The bank is at
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-video-pilot-160-207-seed17-pitchshifted-attempt-01\text-features-v1`.
Its ordered sample-ID hash is
`4abb55fb867a4307928f5237c85d253be21039e470bd3278c9777b40c80b203b`; the
receipt binds to the exact window and AV-report hashes above. No fitting or
test data use occurred.

The first VideoPrism launch exposed a software-metadata naming mismatch before
creating an output bank: the pinned local environment provides the
`tensorflow-cpu` distribution, while the extractor queried the `tensorflow`
distribution name. The extractor now resolves either distribution name and a
regression test covers the locked CPU package. No existing environment or
checkpoint was modified. The CPU/JAX VideoPrism run was active as of 10:49 local with 192/3,527
persisted windows. Latest CPU checks: Ruff passes; pytest reports 2,430 passed
and 11 skipped.
## Sources

- [Official HoloAssist overview and download links](https://holoassist.github.io/)
- [Official pitch-shifted video archive](https://hl2data.z5.web.core.windows.net/holoassist-data-release/video_pitch_shifted.tar)
- [Official annotation format, split, synchronization, and license README](https://holoassist.github.io/data_links/README.html)
- [CDLA-Permissive 2.0 FAQ](https://cdla.dev/faq-resources/faq/)
- [HoloAssist ICCV 2023 paper](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_HoloAssist_An_Egocentric_Human_Interaction_Dataset_for_Interactive_AI_Assistants_in_ICCV_2023_paper.html)
- [HoloAssist ICCV 2023 supplementary material](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Wang_HoloAssist_an_Egocentric_ICCV_2023_supplemental.pdf)

## Frozen feature extraction continuation

At the latest check on 2026-10-19, the selected-window frozen-feature work had these states:

- Text: complete, 3,527/3,527. The bank SHA-256 is `53bbce5ef42345dbe7431fd2369766716b16e51105f8f4adf5208b523492f9a3`.
- Audio: complete, 3,527/3,527. The 3,527 x 384 float32 bank SHA-256 is `0eafa0a03898d52647f8b8bf7895564e488c41a826cdd01167b16dcf4e4a13c4`; finite values and exact train-then-validation sample ID order were verified against the frozen windows. Whisper Tiny ran on the RTX 3080 Laptop GPU in 444.63 seconds, peak allocated/reserved memory 192,923,136/213,909,504 bytes, through `scripts.run_with_hotgpu_guard`. The gate waited for active GPU PID 26292 to yield, then reserved 3,200 MiB before starting the extractor; it released its reservation on completion. No training or checkpoint update occurred.
- Video attempt-01: stopped at a clean saved boundary after 256/3,527 rows. Its progress file SHA-256 is `d41c09e283a76ae0d287b317f1086ea593a7f0b9cfa7af4c933fb970fb7b5839`. It used source commit `92829c98623c16328534b20de066ba40618055b5`, VideoPrism weights SHA-256 `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`, and the pinned CPU/JAX environment. The partial remains preserved; it was not loaded as a finished bank.

A decoder optimization was introduced for the next, separate output attempt. Instead of converting every decoded frame to RGB, it converts only frames selected by the same endpoint-inclusive eight-frame rule (plus a fallback frame only if end padding is needed). Its synthetic tests cover high-rate sampling and end-padding parity. It matched the previous pixel arrays exactly on four real train/validation windows and was 1.49–2.24x faster across those samples. This is a local benchmark, not a broad runtime guarantee. The source change passed 8 targeted tests and Ruff. No cache has been packed and no readout has been fitted. Test data remains unopened.
