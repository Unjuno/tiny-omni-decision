# HoloAssist text-free sensor comparison — progress, 2026-10-19

## Why a new attempt is required

The completed HoloAssist temporal readout is not a text-free baseline: the
video-only, audio-only, and joint arms all received the MiniLM embedding of the
task question. Its 207 official validation sessions were used for checkpoint
selection and comparison, so they are not reused here. The prior result remains
unchanged in
[`R2_HOLOASSIST_TEMPORAL_READOUT_RESULTS_20261019.md`](R2_HOLOASSIST_TEMPORAL_READOUT_RESULTS_20261019.md).

## New development corpus

A new immutable plan was created outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\selection.json`.
It uses only the official train member of `data-splits-v1_2.zip`; the validation
and test member payloads were not read. It excludes the old 160 training
sessions and partitions all 1,306 remaining official-train sessions into 1,042
training / 264 development-validation sessions, stratified by task category
without reading correctness labels. All 20 task categories occur in both
splits; rare categories have only two validation sessions and require cautious
interpretation. Session, video, and window identities are disjoint by
construction. Participant disjointness remains unknown because participant
IDs are not exposed.

- Selection seed: `20261019`.
- Selection SHA-256: `66f578b89ddb1757477b6c75e31b9d3fb7320d6832ae3d23dcd72ad67be17498`.
- Pinned media index SHA-256: `ad5a3843e6592d4be8eab1b187f36ff69f4fab90a9be57c06be2c0ce5fd2f8dc`.
- Allow-listed pitch-shifted media required: 117,074,880,152 bytes (117.07 GB).
- Seed is for deterministic grouping only; no model fitting or checkpoint selection has used this split yet.
- The downloader checks range responses, ETag, member size, selection hash, and per-file SHA-256. It writes only to the new attempt directory.

At the first progress snapshot, 21/1,306 videos (1.24 GB) were verified.
Acquisition is active; this is not a completed media corpus or a training run.
The current C: free-space reading was approximately 172 GB after acquisition
had begun; no existing artifacts were deleted.

The latest verified downloader snapshot in this work segment is 38/1,306 files
(2.09 GB; journal mtime 2026-10-19 05:29 UTC). The downloader process remains
live. A train-only source preflight estimates 9,768 30-second train windows
over 77.2 h of media and 2,498 development-validation windows over 19.8 h;
these are metadata-derived estimates before media timing and event filtering,
not final corpus counts.

An annotation-only preflight (after the session split was frozen, before media
validation) found 88,763 valid train action events with 4.79% wrong labels and
22,763 development-validation events with 5.10% wrong labels. Under the
predeclared “any incorrect action in the 30-second window” label, estimated
positive rates are 28.36% (2,652/9,351 labelled train windows) and 30.30%
(719/2,373 labelled development-validation windows). Invalid intervals,
media-end discrepancies, and AV timing checks may change these counts; the
final event/window receipts remain authoritative.

## Implementation now in branch

Branch `codex/r2-holoassist-textfree`, commit `ecca87093e4c610f4258fbc9c6d6e2fd35cdbc60`
adds a text-free causal AV LSTM readout. It receives frozen video and audio
features plus a typed task-category ID shared by the modality arms; it does
not receive natural-language text or a text embedding. The video-only and
audio-only arms mask the other sensor, and CPU tests verify invariance to
changes in that masked input. Per-task macro metrics and a deterministic
10,000-replicate paired session bootstrap (seed 1701; percentile 95% interval)
are now part of the comparison plan; bootstrap uncertainty is descriptive and
selection-biased because validation selects checkpoints. A new
internal-validation event path reads only the official train split member. The
feature packer supports a video/audio cache with no text bank, and the
extractor accepts the new attempt's SHA-bound selection explicitly while
preserving the old pinned-selection default.

The first pushed implementation commit `ecca87093e4c610f4258fbc9c6d6e2fd35cdbc60`
passed 2,443 tests / 11 skipped and GitHub Actions run
[39203925505](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39203925505)
passed. The text-free cache commit
`e0524e8742e193556e32b22dbd3ef8600d870a42` passed 2,445 tests / 11 skipped,
Ruff, and GitHub Actions run
[39204423670](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39204423670).
The class-balance and paired session-bootstrap commit
`dbf54dbbceea581ad06a40814fbbf8cf0c5854d8` passed 2,447 tests / 11 skipped,
Ruff, and GitHub Actions run
[39204939791](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39204939791).
The latest receipt-provenance commit is `418bd359f8d5226a7a572c8a04cf8c8ae241af87`;
its focused 17-test set and Ruff pass locally, with GitHub Actions run
[39205023515](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39205023515)
queued at the time of this report. No model fit has started.

## Remaining gates

1. Complete and hash-check all selected media.
2. Validate audio/video duration, decode the selected recordings, and build
   fresh train and development-validation action/window receipts.
3. Extract only the frozen VideoPrism and Whisper features; do not compute or
   pass text embeddings.
4. Pack, validate split identities and frozen feature hashes, then run
   video-only, audio-only, and joint arms with the pinned seed/config on the
   local machine. Any GPU work must start via HotUGPU reservation so the
   Orbitune supervisor's free-memory launch threshold remains blocked.
5. Compare NLL, Brier, ECE, Accuracy, class-balanced metrics, per-task results,
   validation prevalence, and session-group uncertainty. These are
   development results, not sealed/test results.

No new quality claim, product promotion, quantization, cloud use, or test/audit
evaluation has occurred in this attempt.

## Live acquisition update (2026-10-19 05:35 UTC)

The allow-listed range downloader (PID 23276, started 14:14 JST) is still
running with its pinned selection, four workers, ETag, and per-file hashes.
Its journal contains 54 complete media files (3.02 GB); four in-flight `.part`
files are present. This is about 2.58% of the 117.07 GB selection. Observed
mean throughput is approximately 2.3 MB/s, implying roughly 14 hours remain if
that rate holds; this is only a live estimate. C: had approximately 165.5 GB
free at the observation. No GPU work has started. Orbitune's supervisor and
trainer remain present, so subsequent GPU work will use the repository's
HotUGPU reservation wrapper with the 6,000 MiB reservation and 14,000 MiB free
memory launch-block threshold recorded in `run-preflight.json`.

The branch is clean and synchronized with its remote at
`1f2a796de11032d6bc16ccfe96c21d4b2caa32c1`. GitHub Actions run
[39205159873](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39205159873)
for that commit completed successfully. This live update records acquisition
and CI status only; feature extraction, model fitting, and quality evaluation
remain pending.

Before feature extraction, a config review found stale descriptive metadata: the
new run's validation is a held-out partition of previously unused official
training sessions, not the legacy official validation split, and its text-free
configuration still listed the unused MiniLM encoder. Both entries were
corrected without changing the frozen selection, AV encoder revisions, readout,
sampling, or optimizer settings. A regression test now checks that the config
contains only video/audio encoders and documents the internal validation
source. The focused training/config tests pass (8 passed), and Ruff passes.
The attempt's external `run-preflight.json` now records corrected config SHA-256
`c023a3d4a8a86ddcb3964503241691717af3bd6aa6b9fb2165110596e7ab504f`, previous
config hash, correction reason, and source commit
`607deea5c2f8d1386d82abbb334deb31a4bcfa10` before any feature extraction or
fit.

At 2026-10-19 05:42 UTC, a CPU audit recomputed hashes for all 60 completed
downloads (3,320,250,818 bytes); every session/split, size, selection SHA,
archive ETag, and per-file SHA matched, with zero errors. The receipt is saved
outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T054142Z.json`
(SHA-256 `14b6a057f0b2fc3d999447c32576d7d8e448cc0e429f55dc616371f8b9aca3c6`).
The pinned local VideoPrism NPZ (457,475,296 bytes) and Whisper encoder export
(32,839,984 bytes) also match the exact hashes recorded for extraction. These
are artifact identity checks only, not quality results. The downloader remains
live; GPU feature extraction has not begun.

At 2026-10-19 05:43 UTC, the downloader journal had advanced to 63 completed
files (3,724,124,825 bytes); the saved independent prefix audit currently
covers the first 60 completed entries, so the additional entries are covered
by the downloader's own checks but not yet by a refreshed independent receipt.
After the config correction, the full CPU test suite passed (2,448 passed,
11 skipped in 35.94 s) and `ruff check .` passed. GitHub Actions run
[39205909345](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39205909345)
for source commit `fd83baf36d5aac8fa1e681fec2103aa6f0577f82` completed
successfully.

At 2026-10-19 05:53 UTC, a fresh independent audit covered 80 completed media
files (5,329,640,326 bytes), with zero errors against the frozen selection,
split, size, ETag, and SHA-256. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T055301Z.json`
(SHA-256 `ddb690eab3c0d08ffe3e222495aebf7a28d1f71df8ee0f150a64c619ee3b14db`),
referenced by `run-preflight.json`. The downloader remains active and the full
media manifest has not yet been emitted.

At 2026-10-19 05:54 UTC, the independent prefix check advanced to 81 files
(5,456,400,093 bytes) with no mismatches. The receipt
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T055359Z.json`
(SHA-256 `f8e8d57f2edc2d7ef261c3cebd8c6ab68bd0f254c16e33a3dacfbe8f25aba7fa`)
is now the current prefix receipt in `run-preflight.json`.

At 2026-10-19 05:45 UTC, an updated independent audit recomputed the full
prefix at 64 completed files (3,779,974,279 bytes) with zero identity, size,
ETag, selection-hash, or media-hash mismatches. Its receipt is
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T054501Z.json`
(SHA-256 `ff666c91208adfee4553c7be711701ed1cd7b6091307bb09b23be9e5ff657896`)
and is the current receipt referenced from `run-preflight.json`.

At 2026-10-19 05:48 UTC, a 4 MiB range probe at the offset of the long-lived
Navvis member request returned HTTP 206 with the pinned ETag in 9.58 s
(0.438 MB/s); it wrote to `NUL` and did not alter the active downloader. The
original request remains live, while other workers continue completing files.
At 05:49 UTC, one newly selected train video (`R005-7July-GoPro`) passed a
single-file PyAV decode smoke: H.264 896x504, eight decoded frames, mono AAC
48 kHz with finite non-silent audio. This is a media/runtime smoke only, not
full-corpus validation or model quality. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\new-selection-av-decode-smoke-20261019T0549Z.json`
(SHA-256 `8927bfd18c48a29e77cf36c0880fb584c38723a82613f433b2d8360872beea09`).

At 2026-10-19 05:51 UTC, the long-lived Navvis range request completed and its
full 505,664,476-byte media file was journaled and recomputed against its SHA.
The downloader remains live with 75 completed files (5,108,356,544 bytes).
An independent prefix audit verified all 75 files with zero errors; current
receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T055047Z.json`
(SHA-256 `cc964bd3ee40d08096fa2b90db08b909f4016d550a1b4b8c896d402bb6fcbb1c`).
The 4 MiB probe was diagnostic only; it did not require restarting or modifying
the long-running downloader.

At 2026-10-19 05:52 UTC, the independent audit advanced to 78 completed files
(5,223,952,340 bytes), again with zero errors. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T055203Z.json`
(SHA-256 `113e04c3054aec33c823ee69a044f22df9360ce7525dfdb087a694250aaf8813`),
now referenced by `run-preflight.json`. GitHub Actions run
[39206659213](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39206659213)
for source commit `c60d6a3d2f0db9b32f9494ef0ea6484a4d28a450` completed
successfully.

At 2026-10-19 05:57 UTC, the active long-range request for
`R106-4Aug-Switch` completed after remaining open for several minutes. A
separate 4 MiB probe at that exact member offset returned HTTP 206 with the
pinned ETag in 10.36 s (0.405 MB/s); no media file was written by the probe.
The downloader did not need intervention. The independent prefix receipt now
covers 82 completed files (5,584,575,501 bytes), zero errors:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T055700Z.json`
(SHA-256 `11f95c8a1fa83255b2f5239f704bfe73cdebfaea09ad1f2e24b846c97ababb65`).

At 2026-10-19 05:59 UTC, acquisition reached 84 files (5,859,641,278 bytes).
The independent audit again found zero mismatches; receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T055917Z.json`
(SHA-256 `f2ef022e4b09b68406fb914a3da3d7f1fe01ccdbb9241d399b4627c71b3ffd0a`).
The long range requests on several large Navvis members remained open while
other workers progressed; no active download was stopped or restarted.
GitHub Actions run
[39207225084](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39207225084)
for source commit `c7b2bde002123cdc4917f41eaa9af7f19d7fe9f0` passed.

At 2026-10-19 06:00 UTC, the independently verified prefix reached 85 files
(6,133,002,174 bytes) with zero mismatches. Current receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T060033Z.json`
(SHA-256 `f8b9eb9e4b5a5f17bd9ba5f595e3fe55f0eacf34d212ed04777d5ec994ead918`),
referenced from `run-preflight.json`.

At 2026-10-19 06:01 UTC, an independent recheck covered 86 files
(6,370,016,891 bytes) with zero mismatches. Current receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T060146Z.json`
(SHA-256 `f10ca75ed5dc0f868aa0704e910363adbc7815d536bfe4840378f45a8ef3c358`).

At 2026-10-19 06:05 UTC, acquisition reached 88 completed train videos
(6,718,302,588 bytes). I recomputed SHA-256 for every completed file and
matched each journal row against the frozen selection ID, train split, archive
offset, member size, and pinned ETag; all 88 passed. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T060502Z.json`
(SHA-256 `fe8fbbc134317db768a8ad56f7646bef935e1010dc91bca0a8be132de5a082f6`).
The downloader remained active (PID 23276); no test/audit media was opened.
CI run [39207608187](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39207608187)
for source commit `f1694b2933f086a33176eeab91bc904328b5e221` passed.
