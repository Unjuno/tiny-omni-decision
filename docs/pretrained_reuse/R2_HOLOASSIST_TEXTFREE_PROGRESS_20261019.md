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

At 2026-10-19 06:07 UTC, the independent audit covered 91 completed videos
(7,465,849,342 bytes), with no selection, split, offset, size, or SHA-256
mismatches. The count advanced from 90 while hashing. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T060753Z.json`
(SHA-256 `ee24f298ea802475e6bb8f70fced5c7860e7288b8341bc2b78e27106966c3bf6`).
CI for the preceding report-only commit `a8c202e0ce00e827776a181d8b92d24ca04b484a`
passed (run [39207929069](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39207929069)).

At 2026-10-19 06:09 UTC, the independent prefix audit reached 92 completed
train videos (7,637,328,627 bytes), with zero selection, train-split, offset,
size, or SHA-256 mismatches. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T060942Z.json`
(SHA-256 `19b22f5178458e5b8b59435f20e87ab7ad44a60ad850f3328d34ac6dc13858aa`).
The downloader remains live and the full 1,306-video corpus is still pending.

At 2026-10-19 06:11 UTC, the independently verified prefix reached 94 files
(7,854,610,672 bytes). Selection identity, train split, pinned archive ETag,
member offsets/sizes and every local SHA-256 matched; zero errors. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T061127Z.json`
(SHA-256 `fe5fdc1bf4641424cb89528e2655d75bbce7e1254e76a53407a22e558659a316`).

At 2026-10-19 06:13 UTC, the independently verified prefix reached 97 files
(8,223,868,098 bytes), with zero identity or SHA-256 errors. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T061333Z.json`
(SHA-256 `c90b1602ba8bea3aaf9934b6fe644337e0a90d9beee06d542df23f56ae5f09c7`).

At 2026-10-19 06:16 UTC, the independent integrity audit covered 101 completed
files (8,904,649,164 bytes), with no selection/split/offset/size/ETag/SHA-256
mismatches. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T061636Z.json`
(SHA-256 `3a74f3fe974bddb997b16a93bbf2d272238bd86b9e95cc8e05f8aad625986494`).

At 2026-10-19 06:18 UTC, 102 files (8,985,153,341 bytes) passed the refreshed
independent prefix audit with zero mismatches. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T061807Z.json`
(SHA-256 `a0305dbc6a8678e557e99ebc44d88062570b3e88e5c395fb2dd28740d532ace0`).
GitHub Actions run [39208996114](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39208996114)
for source commit `1fe6bb64314b6063f12e3f69215c5e2ab78f69a9` passed.

At 2026-10-19 06:20 UTC, the selection-aware audit (now checking train and
validation membership by split) covered 107 train files (9,398,670,355 bytes)
with zero hash or metadata errors; no validation files are in the downloaded
prefix yet. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T062047Z.json`
(SHA-256 `4e5154b2cc869e1c53741a9a216e801544456c9d86312b930e3ce47cd7c0af44`).
An earlier ad-hoc audit invocation used the wrong ETag field and emitted a
FAIL receipt; rerunning with the pinned CLI ETag passed. The failed receipt is
preserved, and the active preflight points to this passing receipt.

At 2026-10-19 06:24 UTC, an independent train/validation-aware audit verified
112 downloaded files (10,076,870,554 bytes; all currently from train) with zero
mismatches. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T062410Z.json`
(SHA-256 `c5bbab4166541533380bd24ce2f88c61dd43256b494a404a6c3213a710ea8b01`).

At 2026-10-19 06:25 UTC, the independent selection-aware audit verified 114
files (10,231,973,262 bytes), all train split, with zero mismatches. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T062558Z.json`
(SHA-256 `04b898137da4f295dce2bd2aeae1f0c79f72a228dc9e77be214a810b02e56340`).

At 2026-10-19 06:27 UTC, the independent selection-aware prefix audit
verified 116 files (10,415,658,424 bytes), all train split, with zero
mismatches. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T062749Z.json`
(SHA-256 `ed7b2b1bce36e6b4df8850c35e714103ac08d551f6f851b27060c36dfad0afab`).

At 2026-10-19 06:30 UTC, the independent selection-aware audit verified 118
files (10,691,586,251 bytes), all train split, with zero mismatches. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T063055Z.json`
(SHA-256 `9a2393ec3a00af974c9bdf5315df6d9aa175c79a0c5d76f22aeb5c55693a52f8`).
The active download journal can move ahead by a few files during hashing; this
receipt covers exactly 118 completed entries.

At 2026-10-19 06:33 UTC, an independent selection-aware audit verified 120
completed train files (10,984,233,623 bytes) with zero mismatches. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T063352Z.json`
(SHA-256 `225b7f3e19b45af7dffcc34fc0d0c31d64b3ea2a7522b3fe75f664483e8e66b0`).

At 2026-10-19 06:37 UTC, the independent selection-aware audit covered 122
train media files (11,625,611,393 bytes) with zero mismatches. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T063710Z.json`
(SHA-256 `d9702d31e2c371ae2eb5330761d2c54ee3d79fc5448cad7e18b0c6be76d906f9`).
A 15:36 JST read-only GPU sample showed the 16 GiB RTX 3080 occupied by an
external Orbitune training process (3.6 GiB reported, 36% utilization); no new
GPU work was started. The HotUGPU gate will wait for that work to yield.

At 2026-10-19 06:39 UTC, the selection-aware independent audit verified 126
train files (12,252,290,780 bytes), with zero selection, split, archive, size,
or SHA-256 mismatches. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T063912Z.json`
(SHA-256 `cc7b4f97e818555ba634fa368291343c2b60f9b3df5e8bb843b35a13bfe464bf`).
No validation media has been reached yet; the downloader is processing train
first according to its fixed selection order.

At 2026-10-19 06:46 UTC, the selection-aware SHA-256 audit covered 135 train
files (13,105,348,737 bytes), with zero mismatches. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T064608Z.json`
(SHA-256 `d2c96480098788baac9c706b81140423caaf385d5509fc5a0ae31a292d8cd001`).

A separate GPU encoder smoke on selected train session `R012-7July-Nespresso`
ran the frozen Whisper Tiny encoder over the 0–30s window. HotUGPU waited for
the external compute PID to yield, then reserved 6,000 MiB; the child produced
a finite 384D feature in 0.5445 s, using 190,619,136 peak allocated and
211,812,352 peak reserved bytes. Save/reload matched exactly. The feature and
result are retained at
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-extractor-audio-smoke-attempt-03`
(result SHA-256 `a028b8882b7d1187ed4497ffaf7b3cf50dc93524469412b5b9805eea5060949d`,
feature SHA-256 `ece17d8ed7ef1b96e47f7d2b58108bdf91173f7a3523f8ea77c849ac1ee12767`).
The smoke used the pinned selection/config and model hashes, opened no test data,
and performed no fitting or checkpoint selection. This verifies encoder plumbing
only; it is not a full feature bank or quality result. The first wrapper attempt
used a Python environment without Torch and exited before the child started; the
existing Python 3.11/Torch runtime then completed the smoke without upgrades.

## Live acquisition update (2026-10-19 06:50 UTC)

The fixed downloader remains active as PID 23276 with its original four-worker
command and pinned selection; it has not been restarted or reconfigured. The
journal reached 137 completed train files (13,579,254,871 bytes); all 137 were
rechecked against selection membership, split, archive ETag/offset, expected
size, output path, and recomputed SHA-256 with zero mismatches. The immutable
receipt is
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T064958Z.json`
(SHA-256 `33d4a056c5d0b63c4b5a82b75572a7db8cad163e650637f08e7ff0bdf95a7db8`).
All completed files are still from the selected train partition; validation
media has not yet been reached. The attempt's `run-preflight.json` points to
this passing receipt. The separate OneDrive checkout has user changes and was
not touched; this update is in the clean dedicated worktree. GPU remains an
RTX 3080 Laptop with 15,821/16,384 MiB reported used at the last sample, but
only 1% utilization. The prior one-window Whisper smoke already passed under
HotUGPU; full feature extraction waits until complete media validation.
No training, fitting, test/audit access, or artifact promotion occurred.

## Live acquisition update and VideoPrism smoke (2026-10-19 07:00 UTC)

The pinned downloader remains active on PID 23276 with its original command,
selection, four workers, ETag, and output path unchanged. At 06:59 UTC its
journal contained 154 completed train videos (15,306,439,461 bytes); an
independent full-prefix audit recomputed every SHA and checked selection ID,
split, archive offset/ETag, output path, and exact size with zero mismatches.
The passing receipt is
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T065910Z.json`
(SHA-256 `59801ed0b3026de990a760ad84e17fb17614adf6e23ec19de4e8312d73d3f54f`).
All completed items remain train split; no validation media has been reached.
The attempt's external `run-preflight.json` points to this receipt.

A one-window CPU VideoPrism smoke was attempted on selected train session
`R005-7July-GoPro` using the already-pinned JAX 0.6.2 / TensorFlow 2.18.1
runtime, exact 8-frame / 288px preprocessing, pinned source and checkpoint.
It failed before feature output while JAX serialized its compilation cache key
(`MemoryError` at `_serialize_ir` / `write_bytecode`). A post-failure system
sample showed 47.965 GB committed against a 48.242 GB commit limit (about
277 MB remaining), despite roughly 10 GB physically available. This strongly
suggests system commit pressure, but the exact failure-time counters were not
captured, so root cause remains likely rather than proven. Failure receipt and
script are preserved at
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-extractor-video-smoke-attempt-04`
(receipt SHA-256 `349c1c307ee1f834d3cc2ee90aaf97c94aa98727cb43161729cd8ab761945d12`;
the receipt hash was refreshed after adding the exact script-start time).
No output feature, model update, fit, checkpoint selection, or test/audit read
occurred; no retry or system setting change was made.

The local RTX 3080 was observed alternating between idle and Orbitune training.
No live process was stopped or modified. The prior frozen Whisper one-window
smoke remains a pass through HotUGPU; any further GPU-bound task will start
through that reservation guard, which waits for external GPU use to yield before
reserving VRAM and launching the child. Full-corpus VideoPrism extraction,
validation, feature packing, and readout fitting remain pending.

## HotUGPU and VideoPrism runtime update (2026-10-19 07:04 UTC)

The same one-window VideoPrism smoke that previously hit JAX `MemoryError`
completed when system commit headroom recovered. Through HotUGPU, the wrapper
waited for Orbitune PID 24056 to yield; after a reservation race it secured
6,000 MiB, left 10,215 MiB free, and launched the CPU-JAX child while keeping
the launch threshold blocked. No Orbitune trainer was stopped. With unchanged
selection, config, media, 8 frames, preprocessing, pinned model source and
checkpoint, VideoPrism produced a finite `[1, 768]` feature. Encoder time was
8.463 s, end-to-end smoke time 13.776 s, peak RSS 4,156,661,760 bytes, and
save/reload matched exactly. Result SHA-256
`1901a46575bd1ecd19e3b3f0b709073ad48118b8c360d0a03b57afab0a7f76af`; feature
SHA-256 `92ac2cb06be9974df505f1676985b6d1b66742c5d7f2c7b64d38b4e5b2d1dcd7`.
This is encoder plumbing only, not model quality. It supports (but does not prove)
that the earlier compile failure was caused by near-exhausted system commit.
The first HotUGPU retry remains recorded separately as an invocation mistake:
the wrapper worked, but its child was mistakenly Python311 rather than the
pinned JAX environment and exited before model forward.

The downloader is still PID 23276 with original parameters. At 07:03 UTC the
journal had 162 completed train assets (15,743,353,057 bytes); a fresh full
prefix hash/selection/split/offset/size audit passed all 162 with zero errors.
Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T070346Z.json`
(SHA-256 `d9c8685b74b550e6306dcaa05ca47f599716011a21d004ce8223c9cbc64f957b`).
Three newer journal files were not included in this receipt at the time last
observed. Validation media is not yet reached.

Audio remains a separate GPU-bound step and will use HotUGPU as well. After the
smoke the RTX 3080 returned to 0 MiB / 0% at the sampled instant; Orbitune's
supervisor remains present and may relaunch when the reservation is released.
No fitting, checkpoint selection, test/audit access, or product promotion has
occurred. Full media validation and the actual frozen train/validation feature
banks remain pending.

The preceding documentation commit 13d67f passed GitHub Actions run [39213226246](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39213226246).

## Download prefix audit refresh (2026-10-19 07:08 UTC)

The live journal advanced to 167 completed files while the previous audit was
being checked. A fresh full-prefix SHA-256 audit covered exactly 167 selected
train files (16,503,928,877 bytes), validating selection ID/split, pinned ETag
and archive offset, destination path, exact size, and digest; there were zero
mismatches and no validation media in the completed prefix. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T070750Z.json`
(SHA-256 `b701da8164db0d6d3e6eb6a784f7584759337adf38357616d2b5837490f7a2f7`).
The external `run-preflight.json` now references this audit. The downloader is
still active with its immutable original command; no files or settings were
changed.

## VideoPrism steady-state smoke and audit refresh (2026-10-19 07:11 UTC)

A three-window CPU-JAX extraction ran in one process under HotUGPU reservation
on three selected train recordings (`R012-7July-Nespresso`,
`R035-12July-Nespresso`, `R031-12July-SmallPrinter`). All three features were
finite 768D vectors and exact after save/reload. The pinned 114,365,184-parameter
VideoPrism checkpoint and preprocessing were unchanged. End-to-end time was
20.225s across three windows (6.742s/window including first compile); encoder
loop time was 14.676s and peak RSS 4,159,377,408 bytes. This small throughput
sample includes model initialization/compile and is not a full-run time
projection. Result SHA-256
`0adcd4647b76d3ac32a80dfa8b0253b5e88637b93e42b0de4a0929b53081d866`; features
SHA-256 `47c9d8aec297399afc7e2aa8698ba7abea37fcf137564ae4070a5f864a216ff1`.
No fitting, test/audit access, or checkpoint selection occurred.

The HotUGPU guard released reservations during several high-utilization races,
then acquired 6,000MiB and launched only with 10,215MiB free. After the child
exited, the RTX 3080 returned to 15,117MiB used / 100% utilization in the next
sample, with Orbitune trainer PID 16832 observed. No process was stopped or
modified. This confirms the reservation blocks startup only while our guarded
child holds it; future GPU work must remain inside the gate for its full
GPU-critical duration.

At 07:10 UTC, the live download had 171 completed train files (16,888,173,999
bytes). The refreshed full-prefix audit verified every selection ID, split,
ETag, archive offset, destination path, size, and SHA-256 with zero mismatches.
No validation files have completed. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T071058Z.json`
(SHA-256 `59b6550d11862bad3ecf90ead13fd215f56fb5bcf4d519afcab16b913e4b99c7`).
The attempt preflight now records this receipt and the three-window smoke.

## Live media audit refresh (2026-10-19 07:13 UTC)

The active downloader (PID 23276, original four-worker command) reached 175
completed records / 17,089,550,898 bytes. A fresh full-prefix audit verified all
175 against frozen selection identity, split, ETag, archive offset, destination,
size, and recomputed SHA-256; zero mismatches. They are all selected train
files; no validation media has completed. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T071316Z.json`
(SHA-256 `51a605ca25577979e0d45287bbb647fea067361fab645e87fc68de375921357e`).
`run-preflight.json` points to the receipt. This is about 14.6% of the selected
AV byte total, so full corpus validation and learning remain gated on further
acquisition.

The latest read-only GPU sample shows Orbitune training active at roughly
15.8/16 GiB and 96% utilization. No local feature or fit job was launched while
it occupied the GPU; the previous HotUGPU-gated CPU VideoPrism throughput smoke
is separately recorded above. The latest repository CI passed for commit
`8b62e33` (run [39214399656](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39214399656)).

## Frozen input and media recheck (2026-10-19 07:15 UTC)

The annotation JSON and official train/validation split archive were rehashed
against the source-screen manifest immediately before the full media pipeline:
annotation SHA-256 `cc7898b49958a62fe021ae2ffa53a709c3fd6f45fd3f893960b8aac6d13dfe9c`
and split archive SHA-256 `e10674e7ac32957386d5e88afa60acf3335251b92ba2f8cfb97a736dc85e1621`; both match. Only file hashes were read in this recheck.

The downloader reached 176 complete train files (17,154,179,785 bytes). A full
prefix audit verified all 176 with zero selection, split, archive metadata,
size, path, or SHA mismatches. No validation media has completed. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T071528Z.json`
(SHA-256 `6c536a9e77003a78af3f839fbd2b556b20f276ff200921ee6f999b632241a600`).
`run-preflight.json` now records both the current prefix receipt and pinned
source-input hash recheck. The HoloAssist source annotation and split archive
are unchanged; test labels/media remain unopened.

## Live media prefix refresh (2026-10-19 07:16 UTC)

The active downloader reached 179 completed train assets (17,370,208,026
bytes). A new full-prefix audit validated all 179 selection IDs/splits, the
pinned archive ETag and offsets, file paths and sizes, and recomputed media
SHA-256 values with zero errors. No validation media has completed. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T071639Z.json`
(SHA-256 `19c21f36174860e378287eff2990f461c10bd4277d7eb28ad64716d56f757367`).
The external preflight now points to this full audit. Orbitune was using about
15.8 GiB of VRAM at the last sample, so no new GPU task was launched.

## Live acquisition verification (2026-10-19 07:18 UTC)

The download journal now has 180 completed selected train videos, totaling
17,446,110,088 bytes. An independent full-prefix audit checked selection
identity/split, pinned archive ETag/offset, exact output path and size, and
recomputed every SHA-256; all 180 passed with zero mismatches. No validation
media has completed. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T071853Z.json`
(SHA-256 `c9291c30f1ca8f38a341491d4d8b32c7950c83d56a5a830f769c5187488db34e`).
The attempt preflight points to this receipt. The local GPU was occupied by
Orbitune at the last read; no GPU work was started.

## Live media audit update (2026-10-19 07:20 UTC)

The fixed download process remains active with its original selection and
parameters. During the latest read it reached 182 completed train media files
(17,647,483,758 bytes). A full-prefix independent audit recomputed the hashes
and checked each selection identity, split, ETag, archive offset, path, and
size; all 182 passed with zero mismatches. Validation remains 0 files. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T072055Z.json`
(SHA-256 `0e2eb67293c093d789c267af607f18dac943160fc13ba88346d17a5b369bb09d`).
The attempt preflight points to this audit. RTX 3080 was occupied by Orbitune
at the most recent sample; no training or feature extraction was launched.

## Live media prefix audit (2026-10-19 07:22 UTC)

The unchanged downloader reached 185 completed train videos (18,651,095,324
bytes). The refreshed full-prefix SHA audit verified all 185 selected identities,
split memberships, archive ETags/offsets, exact paths/sizes, and recomputed file
hashes with zero errors. No validation media has completed yet. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T072246Z.json`
(SHA-256 `5150fa363ef38a06aa96c23a63d022ed6aaf4778584aabcb61c3d18f9e7dc5db`).
The external preflight points to the passing receipt. At the latest GPU sample,
usage was 0 MiB / 16,384 MiB; no GPU job was started because full-corpus AV
validation remains gated on the unfinished acquisition.

## Acquisition and CI update (2026-10-19 07:24 UTC)

The downloader remains live and has completed 187 selected train assets,
totaling 18,768,257,715 bytes. All 187 passed a fresh full-prefix hash and
metadata audit with zero errors; validation remains 0. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T072442Z.json`
(SHA-256 `abcaa934b1d78659cf38790d718df01d2737b2dbc660d381649267cc6e8446c8`).
The updated receipt is referenced by the external run preflight. CI for the
previous record commit `a3f3204` passed at run
[39215655105](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39215655105).
The dataset and GPU gates remain unchanged: full media and AV validation first,
then frozen feature extraction and matched validation-only readout evaluation.

## Acquisition verification refresh (2026-10-19 07:26 UTC)

The active journal has 188 completed train recordings (18,797,678,575 bytes).
A fresh full-prefix audit verified all 188 file hashes and selection/split/
archive/path/size metadata with zero mismatches. No validation files have been
completed. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T072631Z.json`
(SHA-256 `b4bb05d522f8647e7bace10560eb75e0c8ec0a683797377fef5f9021f515b50d`).
The external preflight points to this receipt. CI for the previous report
commit `0896e58` completed successfully at run
[39215849783](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39215849783).

## Acquisition and HotUGPU handoff update (2026-10-19 08:11 UTC)

The unchanged PID 23276 downloader remains active with its original four
workers. The journal reached 265 completed train videos (25,528,058,540 bytes);
validation remains at zero. The latest chained audit confirmed that the
previously audited journal byte-prefix is unchanged, then recomputed SHA-256
for the six newly completed files and rechecked selection/split, pinned ETag,
archive offsets, paths and sizes. The cumulative 265-file chain passed with no
errors. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T081116Z-chained.json`
(SHA-256 `822cb644d5c4b469bc98c20fdc2fbba272a8a60d160da84ef0428641cd530f04`).
The external `run-preflight.json` points to this receipt. Training media was
read only for hash verification; no validation files have been downloaded,
and no test or sealed-audit media has been opened.

At 17:11 JST, C: had 144,367,824,896 bytes free (134.45 GiB). The selected
media has 91,546,821,612 bytes remaining at the audited prefix. Applying the
downloader's 1.2x free-space check to that remainder requires about 109.86 GB,
leaving about 34.51 GB measured headroom. This is a point-in-time capacity
check; it does not change the downloader or guarantee future free space.

One intermediate audit attempt was invalid because its checker incorrectly
expected the archive ETag inside `selection.json`; the downloader pins that
ETag in its command line. That check failed before opening any media. The
invalid receipt is preserved and marked superseded in the external preflight;
the corrected full audit above passed.

HotUGPU policy was clarified: any later CUDA feature extraction must be
launched through `scripts/run_with_hotgpu_guard.py`, whose reservation and
free-memory gate prevent the Orbitune supervisor from starting its GPU job
while the guarded child is active. No GPU work is appropriate yet because the
selected corpus is incomplete and validation media has not been acquired.
CI for the preceding status-only commit `4c31f4b` passed at run
[39216975277](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39216975277).
The follow-up audit-record commit `c8790e9` also passed CI at run
[39217234351](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39217234351).
Commit `6c1855d` passed CI at run
[39217626921](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39217626921).
Commit `3fad664` passed CI at run
[39217826460](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39217826460).
Commit `578a33d` passed CI at run
[39218196114](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39218196114).
Commit `0a696b2` passed CI at run
[39218387165](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39218387165).
Commit `ca46251` passed CI at run
[39218679757](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39218679757).
Commit `bf91480` passed CI at run
[39218815925](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39218815925).
Commit `286663c` passed CI at run
[39219028511](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39219028511).
Commit `6272473` passed CI at run
[39219344491](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39219344491).
Commit `83a923b` passed CI at run
[39219561706](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39219561706).
Commit `ae1ca66` passed CI at run
[39219933597](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39219933597).
Commit `a02b7d1` passed CI at run
[39220267847](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39220267847).
Commit `09c4399` passed CI at run
[39220509202](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39220509202).
Commit `49560bd` passed CI at run
[39220792117](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39220792117).

At 2026-10-19 17:16 JST, downloader PID 23276 was still active with the
original pinned selection, ETag, output directory, and four workers. The
download journal had advanced to 270/1,306 selected train videos. A chained
audit independently rehashed the five newly completed files and verified the
prior 265-line journal prefix, selection SHA-256, split, ETag, archive offsets,
paths, and sizes. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T081605Z-chained.json`
(SHA-256 `a525fad074fcea46da3639d326ffaf95ede8dd27f2ed362afe46d66f31559471`),
covering 25,836,984,085 bytes; zero hash errors and no validation/test/audit
media opened. The journal was opened with shared read access because the active
downloader holds it exclusively. C: had 145.99 GiB free at the follow-up
capacity check. A later check during the same active download measured 140.39
GiB free and 38.40 GB headroom above the 1.2x remaining-download estimate.
AV validation and model fitting remain gated on the complete
selected corpus; no GPU work was started in this turn.

At 2026-10-19 17:18 JST, a second chained audit extended the verified prefix
to 276/1,306 train videos (26,418,035,506 bytes). It rehashed six newly
completed media files and verified the exact preceding 270-line journal
prefix, with zero errors. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T081844Z-chained.json`
(SHA-256 `1cad8b195b5275d0dcb80fa0279f17039c52a19e9ef47ff7efbf9aead75ea35c`).
Downloader PID 23276 remained active on the pinned four-worker command. A
capacity snapshot measured 145.72 GiB free with 44.41 GiB headroom above the
1.2x remaining-download estimate. The pushed documentation commit
`9eba0d0` passed CI run
[39221729023](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39221729023).

At 2026-10-19 17:21 JST, the chained audit advanced again to 282/1,306 train
videos (27,060,505,296 bytes), rehashing two appended files with zero errors.
Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T082155Z-chained.json`
(SHA-256 `7124800c17cea5a8f3af5858dfcd1456fb84b594a7eb360068c95629cad1416c`).
The pinned downloader PID 23276 remains active. The latest capacity check
measured 145.22 GiB free, with 44.62 GiB headroom above the conservative 1.2x
estimate. Validation media have not been downloaded or opened. GPU extraction
and model fitting remain gated on completion and integrity verification of
the selected corpus; subsequent CUDA work will be launched with HotUGPU.

The next chained receipt advanced the verified prefix to 295/1,306 train
videos (27,965,684,519 bytes). Eight newly appended files across the latest two
audit links were rehashed; each chain checked its prior journal-byte prefix and
the immutable selection metadata. The latest receipt is
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T082717Z-chained.json`
(SHA-256 `db6fb4cfe9d950140fb6dd8a40d28a1db12dc7dfde1b59c6c2beac88e3742429`);
zero errors. The downloader remains live with its original four-worker
settings. C: had 138.92 GiB free and 39.33 GiB headroom over the conservative
remaining-download estimate. No validation/test media were opened.

At 2026-10-19 17:29 JST, the prefix reached 300/1,306 train videos and
28,197,925,603 bytes. Five new entries passed full file rehash and chained
journal/selection checks with zero errors. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T082918Z-chained.json`
(SHA-256 `7cba58fef0598ac503f69bbcaa382538323b2085f3178bf792e9cfc38b3b53bc`).
The live downloader remains PID 23276 with four partial files. C: had 138.74
GiB free; the conservative remaining-download estimate leaves 39.41 GiB.

At 2026-10-19 17:32 JST, the chained verification reached 306/1,306 train
videos (28,645,206,281 bytes); four appended media files rehashed cleanly.
Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T083254Z-chained.json`
(SHA-256 `253eb074a326cdbaffbe5e418a6130f83820d3f3d74049781297faad8f8d8758`).
PID 23276 remains live with four partial transfers. C: had 138.16 GiB free and
39.33 GiB headroom using the conservative remaining-download estimate.

At 2026-10-19 17:34 JST, the chained audit verified 311/1,306 train videos
(29,013,035,363 bytes). Five appended media SHA-256 values and their pinned
selection metadata passed with zero errors. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T083426Z-chained.json`
(SHA-256 `21de6930b0432984f1d635ff4acfc8cd4e8eb90c7d645f08ab251bb1bf2e7dd0`).
The same four-worker PID 23276 is live; four partial files are still active.
C: had 143.52 GiB free and 45.10 GiB headroom above the conservative estimate.

At 2026-10-19 17:36 JST, the audit chain advanced to 314/1,306 train videos
(29,188,421,605 bytes). Three additional files passed SHA-256 and metadata
checks, with zero errors. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T083638Z-chained.json`
(SHA-256 `7c60cade5a4e9f891f4bf5e77cf13da4511d3520691d9c7419aec65aa2848113`).
PID 23276 remains live with four partials; C: had 143.16 GiB free and 44.94
GiB headroom under the conservative estimate.

At 2026-10-19 17:38 JST, the audited prefix advanced to 319/1,306 train
videos (29,572,668,522 bytes). The five newly completed media files passed
SHA-256 and immutable-selection checks; the chained receipt reports zero
errors:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T083822Z-chained.json`
(SHA-256 `0af26270edd3eaf95d4bed5ce79355cc7971f4a985414af6595711073abd8ae5`).
PID 23276 is still active with four partial transfers. C: had 137.46 GiB free
and 39.67 GiB headroom using the conservative remaining-download estimate.

At 2026-10-19 17:44 JST, the chained audit advanced to 326/1,306 train videos
(30,177,542,231 bytes). Seven files appended since the prior documented
checkpoint were independently rehashed across chained receipts; prior journal
prefixes and pinned selection metadata matched, with zero errors. Latest
receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T084441Z-chained.json`
(SHA-256 `13c8eb2ec553b558749d42451e45cf729cfc0aadd8ef6d8999a224f404d77bec`).
PID 23276 remains active. C: had 142.13 GiB free and 45.02 GiB headroom above
the conservative remaining-download estimate. No validation/test media were
opened.

At 2026-10-19 17:49 JST, the chained audit reached 332/1,306 train videos
(30,853,750,485 bytes). Two newly appended files passed full SHA-256 and
selection-metadata verification, with zero errors. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T084912Z-chained.json`
(SHA-256 `e9ede55ee850f56039247221eb34b5d8252b97fc88fd0f6aa0263a27bb5c249d`).
PID 23276 remains active with four partial transfers. C: had 141.45 GiB free
and 45.09 GiB headroom over the conservative remaining-download estimate.

At 2026-10-19 17:51 JST, the verified prefix advanced to 337/1,306 train
videos (31,199,515,609 bytes). The newly appended file passed SHA-256 and
selection checks; the chained audit reports zero errors. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T085129Z-chained.json`
(SHA-256 `9062a04edbd7e3c4077624f1ff3a394867e041c14a32ea26c906e2ade16d4570`).
PID 23276 remains active with four partial transfers. C: had 141.17 GiB free
and 45.20 GiB headroom above the conservative remaining-download estimate.

At 2026-10-19 17:53 JST, the chained audit advanced to 343/1,306 train videos
(31,850,609,601 bytes). Six newly completed entries passed file hashing and
selection/split/ETag/offset checks with zero errors. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T085312Z-chained.json`
(SHA-256 `fcc41255b0024527a54066de33d6f3bbc77ef88a26917d8177f88305b985d46a`).
PID 23276 remains active with four partials. C: had 135.41 GiB free and 40.17
GiB headroom under the conservative remaining-download estimate.

At 2026-10-19 17:57 JST, the chained download audit reached 349/1,306
train videos (32,243,114,659 bytes). Two newly completed files were independently
SHA-256 checked against the journal and frozen selection; the chained audit
reports zero errors. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T085722Z-chained.json`
(SHA-256 `1534c19b396e957f0ff0b5b0e176fe56a167a9392bb52af4f79c89a143f11a23`).
PID 23276 remains active with its original four-worker command. C: had 140.26
GiB free and an estimated 45.45 GiB headroom after applying the 1.2x factor
to remaining selected bytes. No validation/test media were opened.

At 2026-10-19 17:59 JST, the chained audit verified 352/1,306 train videos
(32,526,454,257 bytes). Three new files (283,339,598 bytes) passed independent
SHA-256, frozen selection, split, ETag, offset, path and size checks. The receipt
reports zero errors:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T085926Z-chained.json`
(SHA-256 `26941db786264512f4af45e3280f4671717be0b42c50840afed4e43b42355a12`).
PID 23276 remains active with four established connections. No validation/test
media were accessed.

At 2026-10-19 18:01 JST, chained media verification reached 355/1,306 train
videos (33,042,720,663 bytes); three appended videos passed SHA-256 and frozen
selection/split/ETag/offset/path/length checks. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T090117Z-chained.json`
(SHA-256 `31d095eda709427ec84ed9027f9ea903b3a88ff9d86b26dafb4f94babdd9dafe`).
An additional CPU PyAV stream-metadata/duration check passed for the three
files in the preceding receipt (not a full-corpus decode):
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T090046Z.json`
(SHA-256 `6cff34cbbd61c114ba5f2371230e87a12dc8e22159671b59742c3dfc03017e6c`).
No validation/test media were opened; PID 23276 continues with the original
command and four workers.

At 2026-10-19 18:03 JST, the chained download audit reached 359/1,306
train videos (33,243,894,585 bytes). Four newly appended files passed SHA-256
and frozen-selection integrity checks. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T090249Z-chained.json`
(SHA-256 `172f47a5e71d91637c106fcdf162b9aa92748e671c4eb39d0a1c72a99f53c713`).
All four also passed the incremental CPU PyAV stream-metadata and annotation
duration check (not full decode):
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T090301Z.json`
(SHA-256 `0871f1825ee16e352f57ee62a772c2b696577f56d3e0c336baa8d6d0a7611b03`).
No validation/test media were opened; acquisition remains on PID 23276.

At 2026-10-19 18:05 JST, 363/1,306 train videos (33,452,472,106 bytes)
were chained-audited. Four new media hashes and pinned split/selection metadata
passed with no errors. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T090450Z-chained.json`
(SHA-256 `8b9d2719688b1acb76344f67240049b828a088e2bd008ce425cb110b9eb067c6`).
The same four files passed incremental CPU PyAV stream and annotation-duration
checks; full-corpus decode remains pending:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T090502Z.json`
(SHA-256 `841b921a4de0489eb6499839a8e816ff3317281033a6f431fedb39e21dd6d27b`).
No validation/test media were opened.

At 2026-10-19 18:07 JST, the chained download prefix reached 367/1,306
train videos (33,725,554,445 bytes). Four newly completed videos passed
selection, split, ETag, offset, path, length and SHA-256 checks. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T090647Z-chained.json`
(SHA-256 `a27b19d130a85e082d8ce1b042d9cc59b95d3306da244aa85a07a6dca537894a`).
Incremental CPU PyAV stream metadata and annotation-duration validation also
passed on those four files; this does not substitute for full decode:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T090708Z.json`
(SHA-256 `3feb50d1c876b2dda8f482d52209dd7f6751adbb845b431312df04266ff1dfa5`).
No validation/test media were opened; the original downloader remains active.

At 2026-10-19 18:10 JST, the chained audit verified 372/1,306 train videos
(34,165,979,329 bytes). Five appended files passed SHA-256 and frozen
selection/split/ETag/offset/path/length checks. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T091039Z-chained.json`
(SHA-256 `a293d00c57268ce85ea1719ab822ed56ede45bb957d2bb2bb5dcc6ce3a23f07e`).
All five also passed incremental CPU AV stream metadata and annotation-duration
checks; this is not the full-corpus media decode gate:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T091049Z.json`
(SHA-256 `d477379709caff65251aa6a101b247a44528f04924b7f7527e75d1a7f848ed60`).
No validation/test media were accessed.

At 2026-10-19 18:12 JST, the chained media audit reached 377/1,306 train
videos (34,699,769,895 bytes). Five appended files passed independent SHA-256
and frozen selection/split/ETag/offset/path/size checks. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T091215Z-chained.json`
(SHA-256 `93e0b994b5057b4ebbb89fbd75dacda6224464fbf504afcaf508e8a66d0f340e`).
All five passed incremental CPU AV stream-metadata and annotation-duration
checks (not full decode):
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T091229Z.json`
(SHA-256 `1fee191847c89be653f4fe0ae72de452a067281eee7a211863eae1b55df7ab17`).
No validation/test media were opened.

At 2026-10-19 18:14 JST, 381/1,306 train videos (34,921,058,076 bytes)
passed the chained media hash and pinned-metadata audit. Four new files were
checked, with zero errors. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T091405Z-chained.json`
(SHA-256 `e73320e5b93510b10ea48ffe36e087f6dced1e90c1ce7185f862339ebf8a6b2c`).
All four also passed incremental AV stream and annotation-duration checks on
CPU; full decode is still a later gate:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T091415Z.json`
(SHA-256 `a2c90b43a49b91f63024903ed92fc0dddf2cce16f49c133d1a48bc2fda900f1e`).
No validation/test media were opened.

At 2026-10-19 18:16 JST, 385/1,306 train videos (35,193,215,742 bytes)
passed the chained hash and frozen-selection audit. Four appended files were
checked with zero errors:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T091633Z-chained.json`
(SHA-256 `551955694c0cff47e8d26489e4e53faa7def3ea1c047f508bd4f303c5ebe6656`).
The four files also passed the incremental CPU AV stream and annotation-duration
check (not full decode):
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T091643Z.json`
(SHA-256 `a5155fcf391ec270ad89f18f2f99671a370b0d3e603e977118c4281e81909ae2`).
No validation/test media were opened.

At 2026-10-19 18:19 JST, the chained audit verified 390/1,306 train videos
(35,639,410,685 bytes). Four appended media files passed their independent
SHA-256 and immutable selection checks. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T091940Z-chained.json`
(SHA-256 `cce9fdc16e9f744569c0792d60b44fc46ac9526b2ffcf3fb8cec5d908cd1ae93`).
All four passed incremental CPU AV-stream and annotation-duration checks (not
full decode):
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T091950Z.json`
(SHA-256 `4dcd15f90a6d370aa5665d64839bd24f8e10503c7dfc0931a85bb77e02d416af`).
No validation/test media were opened.

At 2026-10-19 18:25 JST, 395/1,306 train videos (36,049,307,737 bytes)
passed the chained media audit. Four appended files passed full SHA-256 and
frozen selection/ETag/split/offset/path/size checks:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T092542Z-chained.json`
(SHA-256 `72e1d4890afd212b56cfef26d00f03fcfe97b1a8d24cd9a9935fd86d70afd4eb`).
The four new files also passed CPU AV stream metadata and annotation-duration
checks; full-corpus decoding is pending:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T092555Z.json`
(SHA-256 `1db69086fd152c169d727a489a4f1b7c593d39848afd8a69a75be9167f03e5cb`).
No validation/test media were opened.

At 2026-10-19 18:29 JST, 399/1,306 train clips (36,832,591,795 bytes)
were verified in the chained prefix. Four new files passed SHA-256 and frozen
selection/split/ETag/offset/path/length checks:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T092942Z-chained.json`
(SHA-256 `367018285c664e279107347a2a627f2e90e2fbd0909dbbb91d801ca54e10a47e`).
All four passed the incremental CPU AV stream and annotation-duration check;
full decode remains pending:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T092954Z.json`
(SHA-256 `52073b4f999f7d964c975561b6b00914e8dab82ad52a7176d8a69b5728a50770`).
No validation/test media were accessed.

At 2026-10-19 18:32 JST, the chained media audit reached 405/1,306 train
videos (37,675,561,053 bytes). Five appended clips independently passed SHA-256
and immutable-selection, split, ETag, archive-offset, path and size checks:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T093201Z-chained.json`
(SHA-256 `327dd46300adcad4f6f0f84a487dc1cf04fbe528b2b032898a7a5c80511c2472`).
All five also passed incremental CPU PyAV stream-metadata and annotation-duration
checks (not full decode):
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T093211Z.json`
(SHA-256 `b7678cf6c76249c6aa03213063c5f7a0491ba5f0c29222611dbdb5c44237c871`).
No validation/test media were opened.

At 2026-10-19 18:34 JST, the audited train prefix reached 410/1,306 videos
(37,888,587,062 bytes). Five newly complete files passed the chained SHA-256
and frozen-selection integrity gates:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T093419Z-chained.json`
(SHA-256 `936134acfcb511f35395cba18bd4c7f6125c4fe8a6f29695812bbaa762d24a26`).
The same five files passed incremental CPU AV stream and annotation-duration
checks, but full-corpus decode remains unstarted:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T093430Z.json`
(SHA-256 `085b6cd4da3ecd369e7adb59d005374806af39db9d23992214f6694c69613c9e`).
No validation/test media were opened.

At 2026-10-19 18:37 JST, the chained audit advanced to 415/1,306 train videos
(38,174,753,755 bytes). The five appended media passed individual SHA-256 and
frozen selection/ETag/split/offset/path/length checks:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T093717Z-chained.json`
(SHA-256 `8d28c537fefc695e62b4148f39c0a9bf04373a23f9b77a3d2f1d22a0b90764e2`).
All five also passed the incremental CPU AV stream-metadata and annotation
duration checks; full decode has not started:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T093728Z.json`
(SHA-256 `1bcfaa64d818a80de7be8072e13097ac7e24a6c382ea3ea04209ee5a3c689385`).
No validation/test media were opened.

At 2026-10-19 18:40 JST, the verified prefix reached 419/1,306 train videos
(38,755,693,690 bytes). Four additional files passed independent SHA-256 and
frozen metadata gates. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T094059Z-chained.json`
(SHA-256 `fd7ee05ac0bc64bc4a185ad710bea35af47a3cb6fd40a1b5d02cb1fae3925306`).
The same files passed incremental CPU AV stream and annotation-duration checks;
this is not full media decoding:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T094110Z.json`
(SHA-256 `49ffe30d39b627b5fa092d220957f3d59703c5dcd8e4dea1a2acb9d6a32db0ba`).
No validation/test media were opened.

At 2026-10-19 18:45 JST, the chained verified prefix reached 425/1,306 train
videos (39,381,686,243 bytes). Six appended files passed independent SHA-256
and immutable-selection integrity checks:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T094511Z-chained.json`
(SHA-256 `a601969611cf23368970dbb4a7f1c4be04e93f1c9c6a67f84e63348d30c5b329`).
All six passed incremental CPU AV stream-metadata and annotation-duration
checks; full decode is pending:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T094521Z.json`
(SHA-256 `90f7d57924d83f91f9eed3611baa8ba617bfee73f2de58fbe41f4eed9414ca4f`).
No validation/test media were opened.

At 2026-10-19 18:46 JST, 429/1,306 train videos (39,943,951,996 bytes)
passed the chained media-integrity checks. Four new files were rehashed against
the frozen selection and journal; zero errors were reported:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T094658Z-chained.json`
(SHA-256 `00fbdf6aed0e31c14441a775205c4ad9777deca16aeaf20f722f2332fbf6d4d1`).
The same four passed CPU AV stream metadata and annotation-duration checks; this
is not full decode:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T094711Z.json`
(SHA-256 `8202cba9a0a3064727322674fa79ff2873048a1e4ed3a4fbe7330389d169bc86`).
No validation/test media were opened.

At 2026-10-19 18:49 JST, 435/1,306 train videos (40,260,028,779 bytes)
passed the chained media audit. Six new files passed hash and immutable-source
checks with zero discrepancies:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T094924Z-chained.json`
(SHA-256 `ca0ec5ccbaff88cd45af0e7f1a85611f273a3614499737f16f3aaf7b7eda6807`).
All six passed incremental CPU AV stream-metadata and annotation-duration
checks; the full media decode gate remains pending:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T094935Z.json`
(SHA-256 `dfb67ecf787d38ece70d3a8499839c1b3f3de2276b57544c0edd9f643a17a858`).
No validation/test media were opened.

At 2026-10-19 18:53 JST, the chained media audit reached 442/1,306 selected train videos (40,858,116,498 bytes). One additional file passed independent SHA-256 verification against the prior receipt, immutable selection, split, ETag, archive offset, and planned size:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T0953Z-chained.json` (SHA-256 `846caba99cb773fa0dbae9e5e4cd99484c63274ef891b9579dca6efbd4ffae83`).
The appended train video also passed CPU PyAV stream metadata and annotation-duration checks:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T0953Z.json` (SHA-256 `050b826ac2b5b623b97b164ddf3efc6dba19446d5c4ceec54d56af06749b1528`). This is not a full media decode. Validation, test, and sealed-audit media were not opened. The downloader PID 23276 was still active at the last process check; its original four-worker settings were unchanged.

At 2026-10-19 18:55 JST, the verified train prefix advanced from 442 to 444 videos (41,097,334,945 bytes). Both appended videos passed chained SHA-256/source identity checks and incremental CPU PyAV stream metadata plus annotation-duration checks. Receipts:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T0955Z-chained.json` (SHA-256 `5df7b0b58821165ba1d53a0728cc6967cb8147868d4aee83058a1ae24ff03488`).
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T0955Z.json` (SHA-256 `22d4dcfddb32ac680046a0d0d1ce4af58ff9837bf7f1f8766a84195995959e1f`). The metadata gate does not replace full media decoding. The fixed downloader remains active; no validation, test, or sealed-audit media were opened.

At 2026-10-19 18:57 JST, the chained train-media audit reached 446/1,306 files (41,600,988,546 bytes). Two newly completed files passed SHA-256 and immutable selection/split/ETag/offset/size checks, then CPU PyAV stream-metadata and annotation-duration checks. Receipts: `C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T0957Z-chained.json` (SHA-256 `424ac4d5da44b407db20722097d306ea2b5040f5d9dacd918489681307432785`) and `C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T0957Z.json` (SHA-256 `a7a48e62587632d6ed2c8ceeb17f6e0fa2b4dfc2b3be6c6478578ecc1068e165`). This is stream metadata/duration validation, not full media decoding. No validation, test, or sealed-audit media were opened; downloader settings and PID 23276 were left unchanged.

At 2026-10-19 18:59 JST, the chained SHA-256 train-media audit reached 449/1,306 files (41,821,968,824 bytes). Three new selected videos passed immutable selection/split/ETag/offset/size and file hash checks, followed by CPU PyAV stream-metadata and annotation-duration checks. Receipts: `C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media-prefix-audit-20261019T0959Z-chained.json` (SHA-256 `0811bb5a94f648ed78aad57205c924efdc672b3ad8fe195a8946329139c04fd7`) and `C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\incremental-av-audit-20261019T0959Z.json` (SHA-256 `601938bcfc2a0da7973eceffa2460e92728e880980a4f7058ca7042c5001dae6`). AV validation here checks stream metadata/duration, not full decode. Validation, test, and sealed-audit data remain unopened; downloader PID 23276 and its settings are unchanged.

At 2026-10-19 19:01 JST, the chained train-media integrity audit reached 452/1,306 videos (41,943,970,941 bytes). Three new train media passed the receipt chain, immutable selection/split/ETag/offset/size checks, SHA-256 rehash, and CPU PyAV stream-metadata plus annotation-duration checks. Receipt hashes: `media-prefix-audit-20261019T1001Z-chained.json` `d091f8a5bb6613e7bd55db586ed550e40ac16516f819a69cc87e34e4ca75c689`; `incremental-av-audit-20261019T1001Z.json` `6aa651892318eeae7144277cb8753622bc2689fca395253a57e20ae7d3687715`. These are metadata/timing checks, not full decode. No validation/test/sealed-audit media were opened; downloader PID 23276 remains unchanged.

At 2026-10-19 19:02 JST, the chained audit advanced to 453/1,306 selected train videos (42,011,996,246 bytes). The appended file passed receipt-chain and immutable-source SHA-256 checks plus CPU PyAV stream metadata and annotation-duration validation. Receipt hashes: `media-prefix-audit-20261019T1002Z-chained.json` `316faea65d0868b02b436a906580deacde2d0b7907182937e2d4d89f782d8619`; `incremental-av-audit-20261019T1002Z.json` `d88d5ce03f96a178dc3a5531d19125303503dd451b6b93d01159b3079f5588c6`. This is not full media decode. No validation/test/sealed-audit media were opened; PID 23276 continues with the original download settings.

At 2026-10-19 19:04 JST, the chained media audit verified 454/1,306 official-train videos (42,052,480,079 bytes). One new file passed source/selection/split/offset/size and SHA-256 checks, then CPU PyAV stream metadata and annotation-duration checks. Receipts: `media-prefix-audit-20261019T1004Z-chained.json` SHA-256 `a89d5bca45ca1740f57bd8a13057276b0906354d605af220148f32af113f5fb2`; `incremental-av-audit-20261019T1004Z.json` SHA-256 `88e1447706fa9eb7134522ea23199dab29877de8159b6e5aceb620c012cb452d`. This is not full media decoding. No validation/test/sealed-audit media were opened; the active downloader was not modified.

At 2026-10-19 19:06 JST, the chained train-only audit reached 456/1,306 videos (42,220,467,796 bytes). Two new videos passed exact selection/split/ETag/offset/size checks and SHA-256 rehash; both also passed CPU PyAV stream-metadata and annotation-duration checks. Receipts: `media-prefix-audit-20261019T1006Z-chained.json` SHA-256 `c1bbacb690300e03f1aa65ec8430e15cca24f0d4fb0d12a2c4f81881df3f2206`; `incremental-av-audit-20261019T1006Z.json` SHA-256 `e3fa38c55468e1b7b1b75f74e3302f45cf2ec8c234a19f88779bba6413f92dd9`. Metadata checks are not full media decode. Validation/test/sealed-audit media remain unopened; the active downloader settings were not changed.

At 2026-10-19 19:08 JST, the chained train-only audit reached 459/1,306 videos (42,412,987,383 bytes). The three newly completed files passed the prior-receipt chain, immutable selection/split/ETag/offset/size checks, SHA-256 rehash, and CPU PyAV stream-metadata plus annotation-duration checks. Receipts: `media-prefix-audit-20261019T1008Z-chained.json` SHA-256 `263eb0a8e64cc32606a439da56196dcc9e8445c9556f6280b462ed6c055d6c57`; `incremental-av-audit-20261019T1008Z.json` SHA-256 `20d8bc073f75a9cbfe4bbb31970e2139d1ded8bd2b0b1546bd36a7b373bd76af`. This is not full media decoding. Validation/test/sealed-audit media remain unopened.

At 2026-10-19 19:10 JST, one additional train video passed the chained source/selection/split/offset/size and SHA-256 checks, then CPU PyAV stream metadata and annotation-duration validation. The verified prefix is 460/1,306 videos (42,439,847,930 bytes). Receipt hashes: `media-prefix-audit-20261019T1010Z-chained.json` `ebf70a2e8c15215bcca43b5bd94af678d09bf8c79080e68f1860f12282684cc7`; `incremental-av-audit-20261019T1010Z.json` `96970fb967e1c203e3aa66ffa23d398aad5db36bf2063960881514c18fba5ab9`. This check does not constitute full decoding. No validation/test/sealed-audit media were opened.

At 2026-10-19 19:12 JST, 462/1,306 selected train videos (42,562,843,927 bytes) passed the chained media audit. Two new files passed SHA-256 and frozen selection/split/ETag/offset/size checks, plus CPU PyAV stream metadata and annotation-duration checks. Receipt hashes: `media-prefix-audit-20261019T1012Z-chained.json` `aecfe0f6561aeba085255d8647ca196b31627aeca580a8230b32162393e020a5`; `incremental-av-audit-20261019T1012Z.json` `81c566390ede97bec169b11d9285595d0085ac8609eab93fdc8838fb4a5276cd`. This is not full decode. No validation/test/sealed-audit data were opened.

At 2026-10-19 19:14 JST, the verified train prefix reached 463/1,306 files (42,599,775,812 bytes). One new file passed chained SHA-256 and source-selection/split/ETag/offset/size integrity, followed by CPU PyAV stream-metadata and annotation-duration checks. Receipt hashes: `media-prefix-audit-20261019T1014Z-chained.json` `8664b92c51347271cb08b16986a9d1707942b031d5853041cc5dabdb52e63482`; `incremental-av-audit-20261019T1014Z.json` `997039baceaf046fbc393bedfda439afa36026c1b1be4d700bf61baf14a13a9b`. This does not replace full media decoding. No validation/test/sealed-audit data were opened.

At 2026-10-19 19:16 JST, the independently chained train prefix reached 465/1,306 videos (42,855,010,274 bytes). Two newly completed files passed selection/source/split/ETag/offset/size and SHA-256 verification plus CPU PyAV stream metadata and annotation-duration checks. Receipt hashes: `media-prefix-audit-20261019T1016Z-chained.json` `90a57cca8c31600ed261610eece468bd79b9eb93de93a5fbe8474e043e8793b0`; `incremental-av-audit-20261019T1016Z.json` `93d89a6d927efd8949e3ccc2167bd070007ace94f89ddcb9d25ef0be16b834f4`. No validation/test/sealed-audit media were opened; this metadata check is not full decode.

At 2026-10-19 19:18 JST, the chained official-train prefix reached 467/1,306 videos (43,019,820,009 bytes). Two appended files passed selection/source/split/ETag/offset/size and SHA-256 integrity checks, then CPU PyAV stream metadata and annotation-duration checks. Receipt hashes: `media-prefix-audit-20261019T1018Z-chained.json` `1b38499c21a6655e81af71aa4510dfc5a95e41fc7f49c1f603fe5e359bcbee71`; `incremental-av-audit-20261019T1018Z.json` `b9ef7338bf7b2350d0b17d971697186e3eca7f38789619609afa8b905688e02a`. These checks are not full video decoding; no validation/test/sealed media were opened.

At 2026-10-19 19:12 JST, one newly selected official-train video (`z090-july-08-22-marius_disassemble`) passed a full CPU PyAV demux/decode of both AV streams after its file SHA-256 matched the 467-file chained receipt. Decoded 3,071 video frames and 5,091,328 audio samples in 5.35 seconds; decoded stream and annotation durations met the existing tolerances. The smoke report is `C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\full-decode-smoke-20261019T1011Z.json` (SHA-256 `6256fa7023301a7d835f6e67e5cbc0f64dd9802efdc687a9103ce7ff3d664d60`). This is a single-file pipeline smoke, not corpus-wide decode or model inference. No validation/test/sealed media were read; no GPU was used.

At 2026-10-19 19:13 JST, an independent 1 MiB HTTP range probe against a selected official-train member returned HTTP 206 with the pinned ETag and exact Content-Range; SHA-256 was computed while streaming without writing a media file. It completed in 2.60 seconds (0.404 MB/s). Report: `C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\range-probe-20261019T1013Z.json` (SHA-256 `68512b0eca6f5bcebbee8e26210f902e338ab47d03678dedd7f9265ed40a377a`). This single small probe shows the endpoint is reachable, not the full downloader throughput. No active download request or setting was changed.

At 2026-10-19 19:20 JST, PID 23276 remained present with its original command and unchanged settings. The downloader journal still had 467 rows (last modified 10:07:15 UTC), while 483 complete train MP4s were visible on disk. A read-only diagnostic inspected the 16 selected files absent from the journal: expected selected sizes matched, 15 passed the current PyAV stream-metadata gate, and one (`z092-july-11-22-marius_assemble`) exceeded the AV-duration tolerance by about 0.0009 seconds (video 278.8184 s; audio 278.8860 s). Its complete CPU decode succeeded (8,356 video frames, 13,386,752 audio samples, 13.10 s), which shows decodability but does not waive the existing metadata gate. Diagnostic receipt: `C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-diagnostic-20261019T102014Z.json`, SHA-256 `3d9c4b251a34fbd7aa1cfb141a4fa480d6866579274f47606e5fafb2b0f4657c`; full-decode receipt: `C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\full-decode-diagnostic-20261019T101949Z.json`, SHA-256 `cda9d0fca58cedfd1f0d2347daddd817723d0ac466e4be0fe20ec5b5b6739457`. The local observed hashes of these 16 files are not source-linked verification, so none were appended to the chained download-prefix receipt. The active downloader was not stopped or modified; no GPU, validation/test/sealed-audit media, or model training was used. The journal/file-count discrepancy is unresolved and must be reconciled before feature extraction.

At 2026-10-19 19:22 JST, a 15-second read-only process/network/filesystem comparison showed PID 23276 CPU time increase from 320.125 s to 320.75 s, four established TLS sockets to the archive host, and complete train MP4 count increasing from 485 to 487. The journal remained unchanged at 467 rows. This supports “transfer active, completion journal not caught up”; it does not justify declaring the downloader stopped or treating unlogged files as verified. At this later snapshot 20 complete files were visible beyond the journal count; only the first 16 were covered by the diagnostic receipt above. All unlogged files remain excluded from the verified prefix pending reconciliation. No process controls or GPU jobs were used.

At 2026-10-19 19:25 JST, PID 23276 was still alive and the downloader journal remained at 467 rows. There were 495 complete selected-train MP4s on disk (28 files beyond the journal). A CPU-only delta diagnostic checked the 12 files that appeared after the prior diagnostic; all 12 passed selected-ID/size and AV stream-metadata checks. Across the 28 currently unjournaled files, 27 pass the metadata gate and the same `z092-july-11-22-marius_assemble` item narrowly fails it. Receipt: `C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T102535Z.json`, SHA-256 `6a2a9d92904873fc4a0bba477d86e2eddf0b15d55863139887e3ddda87acab54`. These local observations are not source-linked verification and are not appended to the prefix receipt. The downloader is continuing; its progress journal is not accounting for newly completed files. No changes were made to the downloader, media, training artifacts, or GPU state.

At 2026-10-19 19:28 JST, the same downloader remained active with TLS connections and new files, but `download-progress.jsonl` still contained 467 rows. The complete train-file count reached 499 (32 beyond the journal). A CPU-only diagnostic covered the 4 additional files since the prior snapshot; all passed selection-ID/size and AV metadata checks. Across all 32 unjournaled files, 31 pass that metadata gate and the single marginal duration-gate failure remains. The 4-file receipt is `C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T102831Z.json` (SHA-256 `270fa4414246c38f5516aea9126c02ae6e4d2cd2537aabfb70148a9e6c8489fa`). Local observed hashes do not establish source identity; all unjournaled files remain excluded from the 467-file chained receipt. No downloader, dataset, process, training, or GPU state was changed.

At 2026-10-19 19:31 JST, PID 23276 remained active with four TLS connections. Its open-file list included `download-progress.jsonl` and four active `.mp4.part` destinations; process I/O counters continued increasing. The journal remained at 467 rows while 504 complete train files were visible (37 unjournaled). A CPU-only delta diagnostic covered the five files added since the prior receipt; all five passed selected-ID/expected-size and AV stream-metadata checks. Across the 37 currently unjournaled files, 36 pass metadata and the one marginal duration-gate failure persists. Receipt: `C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T103100Z.json`, SHA-256 `3317f61a498ad9365378a5f6d0a325f55e32fbf7b6b19babe8b7b7903e259082`. This confirms transfer activity, not the cause of the missing journal entries or source integrity for the unlogged files. The journal/source discrepancy remains unresolved; the verified prefix is still 467/1,306. No process controls, GPU job, feature extraction, or training were performed.

At 2026-10-19 19:36 JST, the downloader still ran with active archive connections. Its journal remained at 467 rows while 513 complete train MP4s were visible (46 unjournaled); there was no final media manifest. A CPU-only diagnostic covered nine files newly visible since the prior diagnostic and all nine passed selection-ID/size and AV stream-metadata checks. Across the 46 unjournaled files, 45 pass the AV metadata gate and one (`z092-july-11-22-marius_assemble`) narrowly fails by about 0.9 ms. Receipt: `C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T103634Z.json`, SHA-256 `2afb64484b3db8ffe4ed49fa56f542c587f6056a5a4cba9f2939a37697324f0a`.

To determine whether that timing boundary came from local transfer corruption, one exact selected-train byte range for the flagged video was streamed from the official archive with pinned ETag and Content-Range, without writing a second media file. The 70,796,254-byte remote range SHA-256 exactly matched the local file (`c0a117970456273d727382c978becd898ea135c037ac77238a9783bc5549e191`); throughput for this single range was 442,585 B/s over 159.96 s while the downloader was active. Source-range receipt: `C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\source-range-hash-audit-20261019T103608Z.json`, SHA-256 `ce67a97b12d23768484e021de6bfe5b4a7150aef72e8c2e07ec911bd407f6e48`. This confirms the narrow AV duration discrepancy is present in the pinned source bytes, but it does not justify changing the existing tolerance or mark the other 45 local files source-verified. The chained verified prefix remains 467/1,306; the cause of the journal lag remains unknown. No active process was controlled, and no GPU, model training, validation, test, or sealed-audit media were used.

At 2026-10-19 19:44 JST, PID 23276 was still active and downloading. The media
directory contained 523 complete files (48,592,166,894 bytes) and six partial
files (46,137,344 bytes). The journal still contained 467 rows (43,019,820,009
bytes; SHA-256
`4814de90b0ffff9cb32ac63b38f7aa8f3401d30598e7a1f56b6ddf8d573de8bc`), with
the same length and modification time before and after a shared-read hash.
Fifty-six complete files were absent from the journal; all 56 names were in the
frozen selection and matched its expected byte sizes, but this is not a
content-hash/source-integrity check. Ten files appeared since the 10:36 UTC
metadata-delta receipt and remain uninspected. The observation receipt is
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-integrity-observation-20261019T104425Z.json`
(SHA-256
`c0532b496a46cca2cc48e71f480c84804b85de141596e3b8c8d6a527bff0c79a`). The
receipt also records the active downloader command and confirms no unexpected
session IDs or size mismatches. The downloader's open journal prevented the
standard `Get-FileHash` path from opening it; a read-shared stream produced the
hash above. This observation does not explain why journal appends have not
advanced. No manifest exists yet. CPU-only HotUGPU and HoloAssist media/journal
tests passed (25 tests); GitHub Actions run #2172 for source commit `42c7685`
completed successfully. No process was stopped, no GPU job was launched, and no
feature extraction or training was performed.

At 2026-10-19 19:52 JST, the exact 467-row downloader journal snapshot was
checked against the 80-receipt integrity chain and re-read from disk. The
shared-read snapshot hash matched the pinned latest prefix receipt. Each of the
467 selected-train MP4s was then rehashed against its journal SHA-256 and had
its PyAV stream metadata checked. All 467 passed: H.264/AAC, mono 48 kHz audio,
start alignment, AV duration difference within two video frames, and audio vs
annotation duration difference within 0.15 seconds. Maximum observed AV
mismatch was 1.973 frames; maximum audio/annotation difference was 0.120 s.
This was a full media-hash and stream-metadata audit of the verified prefix,
not full frame/audio decode and not a validation-quality result. No validation,
test, or sealed-audit media was opened. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\verified-prefix-av-audit-v2-20261019T1051Z.json`
(SHA-256 `ed0d32e758db5b23c4b595e20161f31a1839743e0a2e5bf631fda46c3358465e`);
external audit script SHA-256
`51123dab9f10d6e7c6fecbd0fa949ce34d4c823cec55a5901045c7cd9a7fda80`.
The first audit helper attempt had a duration-string parser defect and is
retained as a failed tool attempt; the corrected v2 parser uses the repository's
`HH:MM:SS.fraction` format and produced the passing receipt.

At 2026-10-19 19:53 JST, a separate read-only snapshot showed 547 complete files
(50,215,561,073 bytes), six partial files (96,468,992 bytes), and the unchanged
467-row journal (43,019,820,009 bytes). Eighty complete files were unjournaled;
all 80 names and sizes matched the immutable selection, but their content
hashes were not verified by this observation. Thirty-four of these files are
newer than the prior 46-file AV diagnostic and have not yet had metadata
checked. The snapshot is
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-integrity-observation-20261019T105358Z.json`
(SHA-256 `42af08d5f3a9dfa768fc243d625e84bfd04ed999848c3aa88b5472efac19f072`).
The downloader remains live; no manifest exists. CPU-only HotUGPU/media/journal
tests remain 25/25 passing, and CI #2173 for `863d0b3` is green. The GPU was
observed idle briefly but no GPU task was launched because the incomplete,
partially unjournaled corpus is not ready for feature extraction or training.
No live process was stopped or restarted.

At 2026-10-19 19:55 JST, a CPU-only local diagnostic covered the 82 complete
MP4s absent from the pinned 467-row journal snapshot. It recomputed all 82 local
SHA-256 values, checked IDs/sizes against the frozen selection, parsed AV stream
metadata, and compared audio duration with train/validation annotation duration.
All IDs and sizes matched; 81 passed the AV metadata gates. The sole failure is
`z092-july-11-22-marius_assemble`: AV duration differs by 2.027 video frames,
while audio/annotation duration differs by 0.094 s. Its exact 70,796,254-byte
range had already been fetched from the official archive with pinned ETag and
Content-Range and its SHA matched the local file, so the timing boundary is in
the source bytes. Local hashes for the other 81 files do not establish archive
identity. No media were decoded beyond stream metadata. The report is
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-audit-20261019T1055Z.json`
(SHA-256 `3b56d8cfd349451f17d7d534040619f70066d32467285711246d20bee97a3fa8`); the
external audit script SHA-256 is
`4f9da69888dac470c469f524eb80639646460887c886ddf568f5bbebff0c6712`.

At 2026-10-19 19:56 JST, PID 23276 was still live. There were 550 complete
media files (50,393,667,446 bytes), 467 journal rows (43,019,820,009 bytes),
and no final media manifest. The most recent 82-file diagnostic predates one
newly visible unjournaled file; that file remains unaudited. CI #2174 for
`b1e46fc` completed successfully. No live process was controlled, and no GPU
feature extraction or model fit was started.

At 2026-10-19 19:59 JST, the four newly completed files beyond the previous
82-file local diagnostic were checked independently. All four IDs and sizes
matched the frozen selection; each local SHA-256 was recorded, and all four
passed the AV stream-metadata and annotation-duration gates. Their local hashes
do not establish archive content identity. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1100Z.json`
(SHA-256 `0f18988c0236d9da8e46dcfd89e4190344b7ac0dc7c3b0f12d3a312bdf647807`);
script SHA-256
`882e3af73c6c8c78fd53d0cb40d425de10e1501221644110e0d7462fa7ea1f3d`.

At 2026-10-19 19:58:55 JST, downloader PID 23276 was still live with archive
connections. The media directory contained 554 complete files (50,721,770,225
bytes), while the journal remained at 467 rows (43,019,820,009 bytes; unchanged
mtime 10:07:15 UTC). Thus 87 complete files were absent from the journal. The
82-file diagnostic plus this four-file delta cover 86; the last file visible in
the live snapshot has not yet been audited. No media manifest exists. GPU usage
was 15,821/16,384 MiB at 9% utilization, so no HotUGPU reservation was
attempted. CI #2175 for `6455750` completed successfully. No process was stopped
or restarted; feature extraction and training remain unstarted.

At 2026-10-19 20:01 JST, local diagnostics covered all 89 complete files absent
from the 467-row journal. Together, the 82-file audit, 4-file delta, and
3-file delta recomputed each local SHA-256 and checked frozen selection ID,
expected size, AV stream metadata, and annotation timing. All 89 IDs/sizes
matched; 88 passed metadata gates. The sole metadata failure remains
`z092-july-11-22-marius_assemble`, whose pinned exact archive range was already
verified to have the same SHA-256 as the local file. Thus its ~2.027-frame AV
duration difference is source data. The other 88 local hashes have not been
compared to official archive ranges. Delta receipt for the final three files:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1102Z.json`
(SHA-256 `47690dde2bfb328357baa0ff0d9bb25413201e1f9fca4431ba98510192f36161`);
external audit script SHA-256
`1b3faeb5af0eac8aa6768e595a55d4bf127dfe28b885dceb735e56668ff22356`.

At 2026-10-19 11:01:30 UTC, downloader PID 23276 remained live with four
archive connections and its journal open. It had 556 complete MP4s
(50,866,823,097 bytes), six partials, and 467 journal rows (43,019,820,009
bytes; last write still 10:07:15 UTC); no final manifest existed. The verified
prefix remains 467/1,306. GPU was idle at this sample, but no GPU task was
started because the corpus is incomplete and most local hashes lack archive
comparison. CI #2176 for `4bd12c2` completed successfully. No live process was
stopped or restarted, and no feature extraction or training was performed.

At 2026-10-19 11:05 UTC, a CPU-only delta audit covered the three files that
appeared after the 11:02Z receipt. All three local SHA-256 values were
recomputed; their IDs and sizes matched the frozen selection, and AV stream
metadata plus annotation-duration checks passed. These are local integrity
diagnostics only; their hashes have not been compared with official archive
ranges. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1115Z.json`
(SHA-256 `ef476aad564c1f559165027f164ce61fca6ae1f93a8f6049c65eee3b96695445`).

At 2026-10-19 11:05 UTC, PID 23276 was still the sole matching downloader
process, with four established archive connections. The output contained 561
complete MP4s (51,647,096,776 bytes) and six partial files (163,577,856
bytes). The journal was still 467 rows / 218,310 bytes, with last-write time
10:07:15 UTC. A 15-second read-only sample saw 561 files at both ends and no
growth in the six partials; the process CPU advanced by only 0.06 seconds in
that interval. This indicates the downloader is currently making little or no
observable progress, but does not identify whether the bottleneck is the
remote archive, a stalled worker, or another I/O condition. The journal lag is
unresolved; do not treat later files as journal-verified or resume/restart this
live process without explicit approval. Two files that appeared after the
previous 559-file sample were subsequently checked by the 11:20Z receipt below.

The RTX 3080 was occupied at 93% utilization with 15,821/16,384 MiB in use at
the same observation. No GPU work was started. Any future GPU-bound child must
be launched through `scripts/run_with_hotgpu_guard.py` with the repository
preflight so it waits for a cooperative yield and reserves memory before the
child process starts. The C: volume had 116.13 GiB free. GitHub Actions run
#2177 for commit `52908e3e83517a54c54819fca0ed7c5443648bac` was still queued;
the worktree remained clean and its branch matched `origin`.

At 2026-10-19 11:06:17 UTC, all 94 complete files absent from the 467-row
journal had local hashes and AV metadata checks recorded: the previous 89-file
set, the three-file 11:15Z delta, and this two-file delta. All IDs and sizes
match the frozen selection; 93 pass the metadata gates and the one failure is
the already source-range-verified timing-boundary asset
`z092-july-11-22-marius_assemble`. The remaining unjournaled files' local
hashes are not archive-authenticated. Receipt for the two-file delta:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1120Z.json`
(SHA-256 `273ff7c917804b995f0c6ec2a04977b617a64191804513c0ac7773fdd743110b`).

At 11:06:17 UTC the downloader remained live, with 561 complete files
(51,647,096,776 bytes), six partials, and the unchanged 467-row journal (last
write 10:07:15 UTC). No final manifest exists. No GPU job or model training
was started; R2 feature extraction remains gated on a reconciled, complete
media manifest and the required task-quality checks.

At 2026-10-19 11:08:26 UTC, two additional complete files were observed and
passed the same CPU-only local audit. Their IDs/sizes matched the frozen
selection and AV metadata/timing checks passed. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1130Z.json`
(SHA-256 `ae53dadf8bd8ca2448d0a153bb22aa0d2d37ed9043e9950ce5ba4dcb8562fd4a`).
Across the 96 complete files still missing from the 467-row journal, all 96
now have a recorded local digest and metadata check: 95 pass, and the sole
failure is the already archive-range-verified timing-boundary file. The other
95 local digests are not official-archive authenticated.

At 11:08:26 UTC, PID 23276 remained live and had four established archive
connections. There were 563 complete files (51,929,026,344 bytes); the journal
remained at 467 rows, with unchanged 10:07:15 UTC mtime. Thus two files became
visible during the recent observation window while journal reconciliation
remains unresolved. No process was interrupted, no GPU child was launched,
and no model work began. The next GPU-bound step still requires a validated,
reconciled corpus and the HotUGPU reservation wrapper.

At 2026-10-19 11:10 UTC, the CPU-only regression command
`python -m pytest -q tests/test_hotgpu_preflight.py tests/test_run_with_hotgpu_guard.py tests/test_holoassist_download_progress_audit.py tests/test_holoassist_selected_media_download.py`
passed all 19 tests. This covers HotUGPU launch/wait-gate behavior and the
selected-media download/journal audit paths; no GPU job was started. GitHub
Actions run #2180 for commit `c51c61c288ea1762ff7e0f0f1f7e4df87c86b623`
completed successfully. The downloader was still alive at the 11:09 UTC
observation with 563 files and four established archive connections; its
journal remained 467 rows, last written at 10:07:15 UTC. This does not resolve
the unjournaled-progress inconsistency.

At 2026-10-19 11:10:35 UTC, two more complete files appeared and passed the
CPU-only local delta audit (selection ID/size, local SHA-256, AV stream and
annotation timing). Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1140Z.json`
(SHA-256 `53fcf2395e00f8205136f0867047f3e2b5de4cfa4b7e043c7b0f949b045452a9`).
The unjournaled set is now 98 files: 97 pass the metadata gates and one is the
known source-verified boundary failure. The 97 passing local hashes remain
uncompared with source ranges.

The downloader was alive at 11:10:35 UTC with 565 complete files
(52,255,143,559 bytes); the journal remained at 467 rows with its 10:07:15 UTC
mtime. No GPU process, feature extraction, or training was started. The latest
branch CI at the preceding source snapshot (#2180) was green; #2181 for
`fe6bba9` was in progress at the time of observation.

At 2026-10-19 11:14 UTC, the repository journal-audit CLI was corrected after
reproducing its documented historical precondition failure. `selection.json`
does not contain `archive_etag`; download code receives the archive ETag as a
separate pinned CLI argument. The audit CLI now requires `--archive-etag` and
compares each journal row with that value. Its regression fixture now uses the
actual selection shape. This was an audit-tool contract defect, not a newly
identified media failure; the earlier no-ETag error is superseded by the
existing corrected prefix receipts and by this tested CLI path.

The corrected CPU-only CLI ran against the live selection, media directory and
journal with the pinned ETag and returned `PASS_PARTIAL_HOLOASSIST_DOWNLOAD_AUDIT`:
467 journal rows / 43,019,820,009 journal-verified bytes, 1,306 selected files,
104 complete unlogged files on disk and six partial files. The command checks
identity/path/size/journal structure; it does not rehash media. Separately,
local-delta receipts now cover all 104 unlogged complete files: 103 pass the
AV metadata gates and one remains the exact source-range-verified timing-boundary
failure. The 103 passing local hashes are not archive-authenticated.

The focused regression suite (HotUGPU gates, HoloAssist journal audit and
selected-media download) passes 19 tests; Ruff passes on the changed auditor
and test. No GPU work or model fitting was started, and PID 23276 remains
untouched. The commit containing this repair and the observation will receive
its own CI run after push.
At 2026-10-19 11:15 UTC, the live repository auditor (with the pinned ETag)
again passed its partial-download integrity gate: 467/1,306 journal rows, 107
complete unlogged files on disk, six partials, and 839 selected files not yet
journal-completed. It intentionally does not hash media. The local-delta audit
then covered the three files newly visible since the 11:55Z receipt; all three
passed ID/size, local hash, AV metadata and timing checks. The cumulative 107
unjournaled completed files now have local diagnostics: 106 metadata passes and
the single known, exact-range-verified timing-boundary failure. The other 106
hashes remain unverified against archive ranges. Delta receipt:
C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1200Z.json
(SHA-256 6d2038051c655ccca7ae1243951bea0021afd828c57baed5f8ed9101d207bf01).

The 11:15 UTC observation found 574 complete files (53,121,978,680 bytes) and
a still-live PID 23276. Media writes continued, but the 467-row journal's
mtime remained 10:07:15 UTC. The audit-tool fix therefore resolves only its
ETag input contract; the live journal lag remains a separate unresolved
behavior. No restart or process intervention was performed. CI #2183 for
e7eab64 was still running at the last check.
At 2026-10-19 11:19 UTC, the live repository audit still passed its partial
journal gate with the explicit pinned ETag: 467/1,306 journal rows, 109
complete unlogged files on disk, six partials totaling 754,974,720 bytes, and
839 expected files not yet present in the journal. The new two-file delta since
the 12:00Z receipt passed local SHA, selection ID/size, AV metadata and
annotation-timing checks. Cumulative unjournaled-file diagnostics now cover 109
files: 108 metadata passes and the same one timing-boundary failure; the 108
local hashes are not source-authenticated. Receipt:
C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1220Z.json
(SHA-256 7ef3c4e7878af4b5508de5df8bdb098a93aac73ffbbd88f6d5bf33b56c6f556a).

A 20-second read-only process sample captured six growing .part files while
PID 23276 remained alive. Complete media count was 576 at 11:19:10 UTC
(53,324,598,946 bytes), with the journal still at 467 rows / last write
10:07:15 UTC. This confirms active transfer, but does not explain why journal
rows have not advanced. No process or GPU intervention was made.
At 2026-10-19 11:20:06 UTC, the live selection contained 578 complete train
videos (53,870,747,487 bytes) and no validation videos yet. The source plan has
1,042 train / 264 validation videos (117,074,880,152 bytes total); 114.11 GiB
was free on C:, exceeding the estimated 75.85 GB safety requirement for the
remaining selected bytes at this observation. The journal auditor passed its
partial gate: 467 journal rows, 111 complete files on disk but not journaled,
six partials, and 839 selected items not journal-completed. A two-file local
delta passed; cumulative unjournaled diagnostics cover 111 files, with 110
metadata passes and the one known boundary failure. The 110 passing local
hashes are not source-authenticated. Receipt:
C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1230Z.json
(SHA-256 0252cd098cba5c031430d12019287f9932bb1d1ced17cc047033f334f1c21774).

The GPU was 86% utilized with 15,821/16,384 MiB in use at this observation;
no GPU job was started. The corrected audit code remains covered by the green
CI run #2183 for e7eab64; the later documentation-only commits did not change
that code.
At 2026-10-19 11:21 UTC, one new complete train file appeared and passed
the local delta checks (SHA-256, frozen selection ID/size, AV metadata and
annotation timing). Receipt:
C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1240Z.json
(SHA-256 d5559ee6d5a9a410c5b0b6201710c32e64da05e045538a961e718b5fe9cf8873).
The latest live journal audit reports 467 journaled rows, 112 complete
unlogged files, six partials, and 839 selected rows not yet journal-completed.
Cumulative local diagnostics cover all 112 complete unlogged files: 111
metadata passes and the one source-range-verified timing-boundary failure.
The 111 passing local hashes do not prove archive identity. PID 23276 remains
live with four archive connections. GPU was idle at the 11:21:02 UTC sample;
no GPU task was launched.
At 2026-10-19 11:23 UTC, two more complete train files appeared; the
CPU-only local audit passed for both. Receipt:
C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1250Z.json
(SHA-256 ae366fb86a68b3c468106496d6f9cd5c8a26839635fa167d1000e89fbddf23b7).
The machine-readable live journal audit is
C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1123Z.json
(SHA-256 654a6f8d5029d1b7538824ae156c8444a87c1713de18523765d29f03044069f2):
467 journaled rows, 114 complete unlogged files, six partials, 839 selected
records not yet journal-completed. The 114 unjournaled files now have local
diagnostics: 113 pass AV metadata gates, one is the known source-range-verified
timing-boundary failure. Local hashes do not authenticate the other 113
against the archive. No live process was stopped or restarted; no GPU job was
started.
At 2026-10-19 11:24 UTC, the downloader remained live with four archive
connections. A new two-file local delta passed selection ID/size, local SHA,
AV metadata and annotation timing checks:
C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1300Z.json
(SHA-256 fa27b0344502a95258b764c0bac1a40aedebfb9cd6334e84fb3f7cab56e05c56).

The persisted live journal audit
C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1124Z.json
(SHA-256 06f7cafeacb951d7f14323bd8a685f6e993d5098d87e5c7a86b2b99c68c5d8a5)
reports 467/1,306 journal rows, 116 complete unlogged files, six partials,
and 839 selected records not yet journal-completed. The 116 unjournaled files
have local diagnostics: 115 metadata passes and one known source-range-verified
timing-boundary failure. The 115 local hashes are not source-authenticated.
GPU showed 54% utilization / 15,821 MiB at the live snapshot; no GPU work was
started and no live process was modified.
At 2026-10-19 11:25 UTC, the live audit receipt
C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1125Z.json
(SHA-256 4716e034c150cb57bd4c0a81f281ce24213e3ba37277688e526b97bf6f626013)
records 467 journaled rows, 118 complete unlogged files, six partials, and 839
selected records not yet journal-completed. The two-file CPU-only delta audit
passed; cumulative local diagnostics cover 118 unjournaled complete files:
117 metadata passes and one known source-range-verified timing-boundary failure.
The 117 local hashes are not archive-authenticated. Delta receipt:
C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1310Z.json
(SHA-256 c05726f4b34afc731b070348f11df0bfffacbcfa11a44ed6c533c7645d81802b).
The downloader remained alive, with no process intervention and no GPU job.
At 2026-10-19 11:26 UTC, one more complete file appeared and passed the
local delta checks (selection ID/size, SHA-256, AV metadata and annotation
timing). Receipt:
C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1320Z.json
(SHA-256 0fa57e613647984911707e7df4b50670f9ff9306e5361c110087450b4878055b).
The live journal audit receipt
C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1126Z.json
(SHA-256 9aef1792407a95a4bd22e9e3ca7b39ecc5838868112fe73db6c2723c6dbd9490)
shows 467 journaled rows, 119 complete unlogged files, six partials, and 839
selected rows not yet journal-completed. Cumulative diagnostics cover all 119:
118 metadata passes, one known source-range-verified timing-boundary failure.
The 118 passing local hashes are not archive-authenticated. PID 23276 remained
alive with four connections. GPU was 46% utilized / 15,309 MiB; no GPU work
started and no process was modified.

At 2026-10-19 11:27 UTC, PID 23276 and its PowerShell parent were still
running the pinned-ETag four-worker download. The persisted journal audit
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1127Z.json`
(SHA-256 `f7afa1b6f15b6df35791f9f3da92a37fd1c7ec11ca8cdbeb1aedccbfec547c7`)
reports 467 journaled records out of 1,306, 588 complete files on disk (121
not yet journaled), six partial files (264,241,152 bytes), and 839 selected
records not yet journal-completed. A CPU-only diagnostic of the two newly
appeared complete files passed local ID/size, AV metadata, and annotation
timing gates. Its receipt is
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1340Z.json`
(SHA-256 `08b10a530c7600807157b3ae45fa6bb1ce170721baf50374bb47048405a294e1`).
Cumulative local diagnostics now cover the 121 unjournaled files: 120 metadata
gate passes and the one known source-range-verified timing-boundary failure.
The 120 passing local SHA-256 values are not archive-authenticated. The
journal was unchanged at the snapshot; the newest complete file was written at
11:26:27 UTC and a partial at 11:26:53 UTC. The observed GPU sample was idle
(0% utilization, 0/16,384 MiB); this is only a point-in-time reading. No GPU
job was started and the downloader was not modified or stopped. Future GPU
work will use the repository HotUGPU guard so unrelated GPU workloads do not
start concurrently; corpus completeness and integrity gates still precede
training.

At 2026-10-19 11:29 UTC, the downloader was still alive and held four
established HTTPS connections to the pinned archive endpoint. The latest
persisted live audit
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1129Z.json`
(SHA-256 `120efbccda111f5cd7a92c0f351c263edc26218752be6e3c8dd4aff3d5002d52`)
still reports 467/1,306 journal rows and 122 complete unjournaled files;
six partial files had grown to 612,368,384 bytes. One additional complete
file passed the CPU-only local diagnostic, bringing cumulative unjournaled
coverage to 121 local metadata passes plus the known timing-boundary failure.
Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1350Z.json`
(SHA-256 `97b87127a451b5c881e5d5b882a3a7fdc0fde1a25209b9944d7e65e843f80047`).
These local hashes do not establish archive-content authenticity. GPU usage
was 0% at the sampled instant, while it varied substantially in earlier
samples; no GPU training was started. Use HotUGPU's wait/reservation behavior
before any GPU task so a transient idle sample does not permit a competing
Orbitune workload to start alongside it.

At 2026-10-19 11:31 UTC, the live audit receipt
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1131Z.json`
(SHA-256 `4804d36671611218c8441b4f5f40f01082f506c2b722fb9cf0e4f3f3f6cd0aad`)
reports 467/1,306 journal rows, 127 complete unjournaled files, and six
partials totaling 486,539,264 bytes. Five newly complete files passed local
selection ID/size, recomputed SHA, AV metadata and annotation timing checks.
Cumulative local diagnostics cover 127 unjournaled files: 126 metadata passes
and the same one timing-boundary failure already source-range verified. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1410Z.json`
(SHA-256 `85b3bdc8daf0916dd65ac517470d4504730d0452811e8f6b4f6e1b446e360ff3`).
The local hashes do not authenticate those records against the archive. The
journal's last write remains 10:07:15 UTC while complete files continue to
appear; the cause is not established. The downloader process and its four
HTTPS connections remained live. GPU readings oscillated from 83% / 15,821 MiB
to idle within adjacent snapshots, confirming HotUGPU must wait for a safe
reservation instead of launching on a momentarily idle reading.

At 2026-10-19 11:33 UTC, another CPU-only delta audit covered four newly
completed local files; all four passed the local gates. Its receipt is
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1420Z.json`
(SHA-256 `925128b0399a405f15bfdece6fcc44fe3edb048f1fd4162bba4174da88d8871c`).
The pinned live journal audit
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1132Z.json`
(SHA-256 `0ea7af0222bbda23af8d30411ac85a3aafb1ae0af8bb85b97db60492f89e6495`)
now sees 131 unjournaled complete files (130 local metadata passes and one
known source-range-verified timing-boundary failure), while the official
journal remains at 467/1,306 rows. Six partials totaled 264,241,152 bytes at
that audit. PID 23276 was still alive at the follow-up process check; GPU then
showed 15,821/16,384 MiB used at 12% utilization. A code review of
`run_with_hotgpu_guard.py` confirms it waits while another compute PID exists
or utilization is at least 5%, allocates a reservation, rechecks the free
memory threshold, and starts the child only after the reservation succeeds.
No training child has been launched: corpus integrity remains the gating item.

At 2026-10-19 11:36 UTC, PID 23276 remained alive with four established
HTTPS connections. The journal still had 467 rows (last write 10:07:15 UTC),
while 134 complete media files were present without journal entries. A new
three-file local delta passed all metadata gates; cumulative diagnostics cover
133 passes and the one known timing-boundary failure. Delta receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1435Z.json`
(SHA-256 `2b3884d91a20a3471292ca325fd8cd8b819370e5439ab57ba226397eb3e1cf90`).
Live audit:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1135Z.json`
(SHA-256 `59918fe2d4f7658ad71a5b402c8185a017e58f1f3d38e75ce2c72b1de43dd0a8`),
with six partials at 285,212,672 bytes. The growing complete-file set and
stale journal are consistent with a downloader main-thread/future error while
executor workers finish queued media, but this remains an unverified
hypothesis because the process output is not attached. The process remains
untouched; wait for its natural exit before considering any recovery.

At 2026-10-19 11:38 UTC, PID 23276 remained live with four HTTPS connections.
The journal was still at 467 rows (mtime 10:07:15 UTC); the on-disk complete
set had grown to 603 files, including 136 unjournaled files. Six partials
occupied 352,321,536 bytes. Two new files passed local diagnostics, bringing
cumulative coverage to 135 metadata passes and one known timing-boundary
failure. Delta receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1450Z.json`
(SHA-256 `52bf354b3bbe36ff353b58ed952619b16005722499580da7a97028e0896ab69a`);
live receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1137Z.json`
(SHA-256 `208abd51c20d9959a4be0ce75dd4400d73d72978d1c79dd4aa3eea56f67c616e`).
GPU was idle at this sample, but no feature extraction started because the
full corpus manifest and integrity validation remain prerequisites.

At 2026-10-19 11:40 UTC, PID 23276 was still active with four HTTPS
connections. Complete files increased to 607; the journal remained at 467 rows
(last write 10:07:15 UTC), leaving 140 complete files unjournaled. The latest
live audit is
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1140Z.json`
(SHA-256 `2e79096c7c2025020090d626dee4888f018d3339982d55a9bbef9f9642a1a4fc`),
with six partial files totaling 331,350,016 bytes. Four newly completed files
passed local gates; cumulative unjournaled diagnostics are now 139 passes and
one known timing-boundary failure. Delta receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1500Z.json`
(SHA-256 `0154522d78949b8024914d5fdb92eb101bfba0e6d5a18c0f4ea512b0816455f1`).
As before, local hashes are not an independent source archive comparison. GPU
was idle at this sample, but recent samples have varied sharply, so any GPU
child remains HotUGPU-gated.

At 2026-10-19 11:41 UTC, a read-only `psutil` inspection of PID 23276
confirmed the downloader process was running, had the progress journal open,
held three `.part` files open, and had four established archive HTTPS
connections. The journal handle being open does not prove the completion loop
is healthy: it is also consistent with the executor context waiting for
workers after a future error. The exception/main-thread state is still
unknown; no debugger or process intervention was used.

At 2026-10-19 11:43 UTC, a 60-second verified wait showed PID 23276 still
live with four HTTPS connections. Five more complete files appeared; all five
passed local diagnostics. The live audit
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1142Z.json`
(SHA-256 `93770932ac3e39c5873423e6431fc20c2b9ee9633154fec63884732646264157`)
records 467 journal rows, 146 unjournaled complete files, and five partials
totaling 192,937,984 bytes. Cumulative local diagnostics now cover 145 of the
146: 144 metadata passes and the known timing-boundary failure. Delta receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1510Z.json`
(SHA-256 `c8a84f19542dd54263f976cb93ca3a997f285e63dab1ff7929d69ffb31499382`).
One newly unjournaled file is not yet covered by the local delta audit. The
journal remains stale; the Python process and HTTP workers remain active.

At 2026-10-19 11:45 UTC, a verified 60-second wait showed PID 23276 still
running with four HTTPS connections. The completed set advanced to 616 files;
467 remain journaled, leaving 149 unjournaled complete files. Four new files
passed local diagnostics, bringing cumulative local coverage to 148 passes
and one known timing-boundary failure. The latest delta receipt is
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1520Z.json`
(SHA-256 `5d2229c885556a09b0f852b3aa9e8f5bdcfceba976030dc1128b6b40d0b15d4c`);
the live audit is
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1144Z.json`
(SHA-256 `c3c697689ad9fbd3e02a2ba8503843bf428e987e7b5299b7e3118db9c71ec3e8`).
The journal remains at 467 rows, last written 10:07:15 UTC, and six partials
totaled 331,350,016 bytes. GPU use was 15,821/16,384 MiB at 38% on the follow-up
sample; no GPU work was launched.

At 2026-10-19 11:45 UTC, two additional complete files appeared and passed
local diagnostics. The pinned audit now reports 151 unjournaled complete files
and six partials totaling 314,572,800 bytes; 150 unjournaled files have local
metadata passes, with one known timing-boundary failure. Receipts:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1530Z.json`
(SHA-256 `aa5dae88cddfc6e196679f1fd26b9ca344b60228e45163d376486c446739d07d`) and
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1145Z.json`
(SHA-256 `340ab9ffc6f5fe1cda5851226bc98d1e52d3bd849e9a2ff30b7a2540ec030711`).
Code inspection of the downloader confirms `--resume` rejects a destination
MP4 absent from the progress journal. If the process exits with this divergence,
do not rerun `--resume` directly; preserve the original journal and reconcile
files through a separate verified recovery artifact first. PID 23276 remained
live with four connections; no intervention was made.

At 2026-10-19 11:46 UTC, PID 23276 remained active with four HTTPS
connections. The complete set reached 621 files while the download journal
remained at 467 rows; 154 complete files are therefore unjournaled. A three-
file local diagnostic delta passed, bringing cumulative local coverage to
153 metadata passes and one known timing-boundary failure. One of the 154
unjournaled files remains unaudited. The live audit
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1146Z.json`
(SHA-256 `5b39faccbf2bbe9c5389c22692ab3f0f2a982d48f6cf1bc063da2e0ae60016cf`)
and delta receipt
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1540Z.json`
(SHA-256 `6084b4721ee92748006492396ef9f1998b2504cdacc3d4aad41e1aeb37812102`)
were saved externally. Six partials occupied 247,463,936 bytes at the audit.
GPU utilization was 74% at the state check, so no GPU task was started.

At 2026-10-19 11:48 UTC, the downloader remained live with four HTTPS
connections. The complete set reached 624 files; 157 are not yet represented in
the unchanged 467-row journal. Three newly complete files passed local
diagnostics, so cumulative coverage is 156 passes and one known
source-range-verified timing-boundary failure. Delta receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1550Z.json`
(SHA-256 `4e1031b7177b6d9d1f482c0b3a35160e1f621ebb4813b8b1f69f221e8244b0d4`);
live receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1148Z.json`
(SHA-256 `d58df232605a862a0caf159ec04d71a8eae3e188a863c75d3d99e9ae2dd4d402`).
Six partial files totaled 369,098,752 bytes. GPU was idle at the latest point
sample, but task start remains gated by corpus integrity and HotUGPU.

At 2026-10-19 11:50 UTC, the HoloAssist/HotUGPU focused CPU suite passed
(86 tests) and Ruff passed for the selected extraction, corpus, audit, and
HotUGPU scripts. No code was changed by this verification. PID 23276 remained
alive with four HTTPS connections; one additional completed MP4 passed local
diagnostics. The live receipt
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1150Z.json`
(SHA-256 `cde22f6e8bf0e54131061ff7b90dc61351f714fbe77de4c710fbd08a6e95ee0f`)
records 625 complete files on disk, of which 158 are not in the 467-row journal,
and six partials totaling 591,396,864 bytes. Cumulative local diagnostics cover
157 of the 158 unjournaled files: 156 metadata passes and one known
source-range-verified timing-boundary failure. The new delta receipt is
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1600Z.json`
(SHA-256 `567e77cb272a4e5b850367a7ecc5b4f03be1f0d25b09305c3735f15b67ea82d0`).
GPU was idle on the last sample, but feature extraction remains gated on a
complete, reconciled media corpus.

At 2026-10-19 11:50 UTC, PID 23276 remained alive with four HTTPS connections.
Two new complete files passed local diagnostics. The live audit
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1151Z.json`
(SHA-256 `52d272f4795e07f47257338f42fa50bf5601f251936ba469d519fdf53e97e4cf`)
reports 628 complete media files on disk, 161 missing from the 467-row journal,
and six partials totaling 100,663,296 bytes. Cumulative local diagnostics cover
160 unjournaled files: 159 metadata passes and the known timing-boundary
failure. Delta receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1610Z.json`
(SHA-256 `95fb825163a7e7d6a72b781d143a8785136b23feb40de7b4d06da5408782f5f9`).
GPU usage was 15,821/16,384 MiB at the follow-up sample, so no GPU work was
launched.

At 2026-10-19 11:51 UTC, PID 23276 remained active with four archive
connections. The latest live receipt
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1152Z.json`
(SHA-256 `82642afc86ac146a3a38ae1cd52ff47fcc1ff988aa460075926b214b6db89036`)
records 629 complete files and 162 unjournaled files; 161 partial-media
records are locally diagnosed as passing and one is the known timing-boundary
failure. The new two-file local delta is
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1620Z.json`
(SHA-256 `24af8f4c5107d0dd56ddc1c45b6622282689387f50ec036802a4476c625bf6ec`).
The parent PowerShell has no open log/output or attempt-artifact handle, and
Codex exposes no attached terminal for this process, so the downloader's
exception output remains unavailable. It remains live and untouched.

At 2026-10-19 11:52 UTC, PID 23276 remained live with four HTTPS workers.
The latest persisted audit
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\live-progress-audit-20261019T1153Z.json`
(SHA-256 `87e9b9b6bbe445d5e15ab3aa316afca01c62b76634e1c09a1ebf3e57929ef9bb`)
records 631 complete files on disk, 164 unjournaled, and six partials totaling
150,994,944 bytes; the official progress journal remains at 467 rows. Two
additional local delta files passed ID/size, local SHA, AV metadata, and
annotation timing checks. Cumulative local diagnostics cover 163 passes and
the one known timing-boundary failure. Delta receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1630Z.json`
(SHA-256 `2702baaa463efd18c64746c62ec1fc676afc794d271620c7fd587f6f79626f7e`).
The local hashes are not source-archive authenticated. No process was modified.

At 2026-10-19 11:55 UTC, an external read-only provenance snapshot was saved at
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\downloader-provenance-live-20261019T1154Z.json`
(SHA-256 `E6ACA3A4FACEC376179213ED3D6C478330B027473EDB8FD3E3D5E6C5B4BD5836`).
It captured a stable snapshot of 637 finalized MP4s: all were selected, matched
the expected byte sizes, and had Windows creation times after PID 23276 started.
The fresh media directory was created 0.220 seconds after that process started.
The process command showed the pinned archive ETag and no `--resume`; the current
downloader source blob matches the recorded source commit
`c60ef8c6c02a345f89ab79c3836b671de3780ed8`. In that code path, HTTP 206,
Content-Range, ETag, and expected size are checked before atomic `.part` to
`.mp4` rename, while per-file SHA and journal append happen after rename. This
supports source-range provenance by execution path for unjournaled files, but
there is no persisted per-file response-header receipt; treat this as a strong
inference, not direct per-file HTTP evidence. The snapshot did not recompute
media hashes.

A follow-up read-only check at 11:56 UTC confirmed PID 23276 was still running
(the apparent absence in one intermediate process query was transient). There
are now 639 MP4s, six partials totaling 46,137,344 bytes, and the unchanged
467-row journal (last written 10:07:15 UTC). All 639 files matched a selected
train/validation ID and expected size. The downloader and original journal
remain untouched. GPU utilization was 0% at one instantaneous sample; this is
not authorization to start GPU work. Any future GPU task remains subject to the
HotUGPU guard and corpus reconciliation.

At 2026-10-19 11:54 UTC, the latest two newly finalized files also passed the
existing local-diagnostic gates (selection ID/size, stable local SHA-256, AV
stream metadata, and annotation duration). Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1650Z.json`
(SHA-256 `3A0F1C5373268DADE88123998BA2EE06070F9DAA6E646626ADE1873347544D75`).
The receipt is bound to selection SHA-256
`66f578b89ddb1757477b6c75e31b9d3fb7320d6832ae3d23dcd72ad67be17498` and
journal snapshot SHA-256
`4814de90b0ffff9cb32ac63b38f7aa8f3401d30598e7a1f56b6ddf8d573de8bc`.
Cumulative local diagnostics then cover 168 unjournaled files: 167 pass and the
one known source-range-confirmed timing-boundary case fails the unchanged local
timing threshold. The 167 local hashes remain unauthenticated against the
archive. At 11:57 UTC the same live process remained active; 639 complete files
matched selected IDs and expected sizes, six partials remained, and the official
journal still had 467 rows. Do not promote this local count to a verified
archive prefix or begin extraction until the journal/source-integrity gate is
reconciled.

At 2026-10-19 12:00 UTC, a further CPU-only delta audit hashed and checked
seven selected, finalized files not covered by the prior receipt set. All seven
passed local ID/size, stable SHA, AV metadata, and annotation-timing gates.
Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1700Z.json`
(SHA-256 `44A30597680AB2070C3DE4C9D2165E50EE2A55A10D51B93A737B528954050A94`).
A separate identity-set reconciliation at 12:00 UTC found 642 finalized MP4s,
467 journaled files, and 175 unjournaled files; the existing local-diagnostic
receipts now cover exactly those 175 unique unjournaled identities. Cumulative
local outcomes are 174 passes plus the previously known timing-boundary failure.
No additional media were hashed by the identity-set reconciliation. This does
not extend independent archive hash authentication beyond the 467-row journal.
PID 23276 remained active, six partials totaled 46,137,344 bytes, and the
original live journal remained at 467 rows; neither downloader nor data was
modified. Feature extraction remains gated on completing and reconciling the
pinned source-integrity chain.

At 2026-10-19 12:01 UTC, two more newly finalized files passed the same
CPU-only local gates. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1710Z.json`
(SHA-256 `9AE44C24EC74F4C66FF29D5A812DEA931BECF44558F7DA048E5C7BA9D6DF991A`).
A fresh identity-set reconciliation found 644 selected finalized MP4s, 467
journal rows and 177 unjournaled files. The receipt set covers exactly 177
unique unjournaled IDs, with no uncovered or unselected file; cumulative local
diagnostic outcomes are now 176 passes and the one known timing-boundary
failure. The pinned live journal SHA remains
`4814de90b0ffff9cb32ac63b38f7aa8f3401d30598e7a1f56b6ddf8d573de8bc`.
The downloader PID remained active; six `.part` files totaled 46,137,344 bytes.
This is identity and local-media validation only; source-authenticated archive
hash coverage remains the 467 journaled rows. No data or process was modified.

At 2026-10-19 12:02 UTC, two incremental CPU audits checked three additional
finalized files; all three passed the existing local media gates. Receipts:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1720Z.json`
(SHA-256 `EFBC7D7FEE78FB8E353652C41718EC1E039F76CD8831109FCFDCD6A6463CB694`) and
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1730Z.json`
(SHA-256 `15D19198E8D2D2D61C12D63645DF63FF0AE0D1E9FA1CE5EA5043937C9CDBA970`).
The second identity-set reconciliation found 646 finalized selected videos,
467 source-hash-authenticated journal rows, and 179 unjournaled videos. Existing
local diagnostic receipts cover exactly those 179 unique IDs, with no uncovered
or unselected files; 178 local diagnostic passes and the single known
source-range-confirmed timing-boundary failure remain. Six partial files remain
(46,137,344 bytes). PID 23276 is still running; no process or media was
modified. Local hashes do not extend archive authentication beyond the journal.

At 2026-10-19 12:03 UTC, three additional selected media files passed the
existing CPU-only local diagnostics. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1740Z.json`
(SHA-256 `8902238F08BCB0026715FE170D6ECECF622A241FFA11BEED536D3A4A52BD1116`).
A subsequent identity-set scan found 649 finalized files, 467 journal rows, and
182 unjournaled files; all 182 IDs are now covered by local diagnostic receipts
with no unselected or uncovered file. Cumulative local outcome is 181 passes
and the one known timing-boundary failure. The process was alive during both
checks. Partial-file count varied between concurrent scans (five then six), so
it is a point-in-time transfer count, not a stable total. The journal remains
source-authenticated only for its 467 rows; no archive authentication is
inferred from local hashes. No data or process was modified.

At 2026-10-19 12:05 UTC, one additional finalized selected MP4 passed the
local diagnostic gates. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1750Z.json`
(SHA-256 `EA2361E9C396F7CE96A486720900DE346F1D7FC48BC694B6DD8BC5E8181681AB`).
The current identity-set snapshot found 650 finalized files, 467 journal rows,
and 183 unjournaled IDs. All 183 are covered by local diagnostic receipts with
no unselected or uncovered IDs: 182 local passes and the single known
source-range-confirmed timing-boundary failure. Six partial files totaled
46,137,344 bytes at this scan. PID 23276 remained active. Archive authentication
still covers only the official journaled prefix; no process/data was changed.

At 2026-10-19 12:06 UTC, one additional finalized selected MP4 passed the
local diagnostic gates. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1800Z.json`
(SHA-256 `59BA2EFB021C532C20DEC7C22775AF573D8ABBC3E9F59C033D5CC5C5D7D6A23D`).
Identity-set reconciliation found 651 finalized files, 467 journaled and 184
unjournaled; receipts cover all 184 unique IDs, with no unselected/uncovered
files. Cumulative local diagnostics are 183 passes plus one known timing-
boundary failure. Six partials totaled 46,137,344 bytes; PID 23276 remained
active. This does not extend source-archive authentication past the 467 journal
rows. No process or media was modified.

At 2026-10-19 12:06 UTC, one additional finalized selected MP4 passed local
diagnostics. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1800Z.json`
(SHA-256 `8DA9F133AF81A49052D30D40808D08637056293607A284A45FCD70EF019DD483`).
The first receipt-set reconciliation used an incomplete filename filter and
under-counted prior receipts; it was discarded. Re-running with all 10:50–18:10
receipts found 652 finalized selected files, 467 journal rows and 185
unjournaled files, all covered by the local receipts with no unselected IDs.
Local cumulative outcomes are 184 passes plus the one known timing-boundary
failure. Six partial files totaled 46,137,344 bytes at the final scan. PID
23276 was active. This local reconciliation does not extend archive hash
authentication beyond the 467 journal entries.

At 2026-10-19 12:07 UTC, two additional finalized selected files passed the
CPU-only local gates. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1820Z.json`
(SHA-256 `93741710F8354BDB87761A7AF23155720081FE5C6D1A864185926A45A2E56BB1`).
The latest identity-set reconciliation found 654 finalized selected videos,
467 journaled and 187 unjournaled. Local receipts cover all 187 unique IDs,
with no unselected/uncovered files; cumulative local diagnostic outcomes are
186 passes plus the one known timing-boundary failure. Six partials totaled
46,137,344 bytes and PID 23276 remained active. Archive hash authentication
still covers only journaled rows; no process or media was modified.

At 2026-10-19 12:08 UTC, the newly finalized `z129-aug-11-22-gopro` passed
local ID/size, stable SHA-256, AV stream metadata, and annotation-timing gates.
Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1840Z.json`
(SHA-256 `4B475CDF826A67E4DE56DC0CA0E599AEF088AF439458456C1441C5A00DC68C37`).
The reconciled snapshot found 656 finalized selected files, 467 journal rows,
and 189 unjournaled files. Local diagnostic receipts cover all 189 unique
unjournaled IDs (188 pass plus the known timing-boundary failure), with no
unselected IDs. Six partials totaled 46,137,344 bytes at the scan, and PID
23276 remained active. This does not create source hash receipts for unjournaled
files; process and media remain untouched.

At 2026-10-19 12:09 UTC, one new selected MP4 (`z129-aug-11-22-marius_assemble`)
passed the local diagnostic gates. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1850Z.json`
(SHA-256 `29A895D44C5E386F2BEC2BC72E1B179FFFDE83C3A26A40636ED59BFDBBE8950A`).
The file-set reconciliation found 657 finalized files and 190 unjournaled files;
all 190 unjournaled IDs are covered by local diagnostic receipts, and no
unselected IDs were present. Cumulative local outcome is 189 passes plus the
one known timing-boundary failure. A read-only SHA attempt against the live
progress journal was denied because another process held the file; no retry
was made. The pinned 467-row journal snapshot remained the reconciliation
basis, and the live journal's observed size/mtime were unchanged. PID 23276
remained active; six partials totaled 46,137,344 bytes. No process or data was
modified; local receipts are not archive-hash authentication.

At 2026-10-19 12:11 UTC, two additional finalized files passed local
diagnostics. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1211Z.json`
(SHA-256 `E6DDA777D15C4429E50E7AC20A344D9BE887C3F5C4EC30F4509103D4C4F0998B`).
Reconciliation through the explicit cumulative report list found 660 finalized
selected MP4s, 467 journal rows and 193 unjournaled files. Local receipts cover
all 193 unique unjournaled IDs, with 192 local passes and the one known
source-range-confirmed timing-boundary failure; no unselected IDs were found.
Six partials totaled 46,137,344 bytes. PID 23276 was active.

**Downloader journal-lock interpretation correction:** source inspection of the
identical pinned/current script blob (`f3b3f60f38ebab1c662741452b2f16ab7f17ab07`)
shows `download-progress.jsonl` is opened at line 218 around the whole executor
at lines 219–237, so another process's inability to read it is expected while
the downloader is alive. A `future.result()` exception at line 225 would exit
the append loop while `ThreadPoolExecutor.__exit__` waits for already-submitted
work, which could explain growing finalized files with a stale journal. This is
a code-supported hypothesis only; the process exception output is unavailable,
so the actual failure remains UNKNOWN. The artifact names ending `T1700Z` through
`T1850Z` were sequential labels, not their actual UTC capture times; use each
JSON `created_at_utc` and filesystem timestamps. New receipts now use actual UTC
names. No source-authentication claim is added for the 193 unjournaled files;
no process or media was changed.

At 2026-10-19 12:12 UTC, two additional finalized selected files passed the
local diagnostic gates. Receipt (named using its actual UTC capture time):
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1212Z.json`
(SHA-256 `B15EA048353DBDB49A3C1294830620512F5F1565B88E722D0766BF8763EC8C45`).
The reconciled snapshot found 662 finalized MP4s, 467 entries in the pinned
journal snapshot, and 195 unjournaled IDs. The cumulative receipt chain covers
all 195 (194 local passes, the known timing-boundary failure), with no
uncovered IDs. Six partials totaled 46,137,344 bytes. PID 23276 remained active;
the live journal is still held open by the downloader and its hash was not
retried. Archive authentication remains bounded to the pinned journal rows.

At 2026-10-19 12:14 UTC, three newly finalized selected files passed the
local gates across two delta receipts:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1213Z.json`
(SHA-256 `2E9B3A76CAB6359F65CEC138F96EC89AC1AF17A967D0370437EC9FCAC26E27C2`) and
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1214Z.json`
(SHA-256 `772EA284E3D5C9D36269204C5B55DEF4E43C49447683855ACD9E18930A3EEC2B`).
The latest identity reconciliation found 665 finalized MP4s, 467 pinned journal
rows and 198 unjournaled files. The cumulative receipt chain covers every
unjournaled ID, with 197 local passes and the one known source-range-confirmed
timing-boundary failure; no uncovered IDs. Six partials totaled 46,137,344
bytes. PID 23276 remains active. Only the pinned journal rows are archive-hash
authenticated; no running process or media was changed.

At 2026-10-19 12:14 UTC, one additional finalized selected MP4 passed the
local gates. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1215Z.json`
(SHA-256 `6D2EA75AFDB3B4B2F8107FAD08EFC0111AA49DE090BEA5F3009213592EA8D562`).
The latest identity reconciliation found 666 finalized selected files, 467
journal rows and 199 unjournaled IDs. The cumulative receipt chain covers all
199 (198 local passes and the known timing-boundary failure), with no uncovered
IDs. Six partials totaled 46,137,344 bytes. PID 23276 remained active; source
archive authentication remains limited to the pinned journaled files.

At 2026-10-19 12:16 UTC, two additional selected MP4s passed the existing
local gates in receipts:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1216Z.json`
(SHA-256 `0BE7865A45DBEF3CD28F032CD8BAF116DA557D38034679FE47FA38ED03CF61E3`) and
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1217Z.json`
(SHA-256 `16933358CD259F831E742C72C507F808078BD94162710F8A01A8738689BA7D12`).
The reconciled snapshot found 668 finalized selected files, 467 journal rows,
and 201 unjournaled IDs. Receipts cover all 201 (200 local passes and one
known timing-boundary failure), with no uncovered IDs. Six partials totaled
46,137,344 bytes. PID 23276 remained active; archive authentication remains
limited to the 467 journaled entries.

At 2026-10-19 12:17 UTC, two additional selected media files passed the
local diagnostic gates. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1218Z.json`
(SHA-256 `7401F486B7EFEC784093316C1DB8C25EE0BF11C595E58D914FF6B0B569AC2F31`).
The latest reconciliation found 670 finalized selected files, 467 journal rows,
and 203 unjournaled files. Existing receipt chains cover all 203 unique IDs:
202 local passes and the one known timing-boundary failure, with no uncovered
IDs. Six partials totaled 46,137,344 bytes. PID 23276 remains active; archive
hash authentication still covers only the 467 journaled rows.

At 2026-10-19 12:18 UTC, four additional finalized files passed local gates
across three new delta receipts:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T1219Z.json`
(SHA-256 `347C7C25878CB6A49056194D6FE0F4CFF8B455363865E7AA24BD39E49E4A6CE4`),
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T121816Z.json`
(SHA-256 `F3938DE8566FB975357978EC33DAAEB0C8641E751E3930D793497094FF3D069A`),
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T121832Z.json`
(SHA-256 `17B7F7D9F89B3B36EF050B8526E8C661F9781A8705C358B69C07A8D936B136FA`),
and `C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T121848Z.json`
(SHA-256 `E2BA141F46ABB26DBE18A34700128ACE16B06BBDEE772C2C4F4EE7D70E6131DA`).
The reconciled snapshot found 676 selected finalized files, 467 journal rows, and
209 unjournaled IDs. The cumulative receipt chain covers all 209, with 208 local
passes and the known timing-boundary failure; no uncovered IDs. Six partials
totaled 46,137,344 bytes. PID 23276 remained active. An existing legacy file
named `unlogged-media-delta-20261019T1220Z.json` had its own earlier
`created_at_utc`; it was preserved and not overwritten or substituted for a
current-time receipt. Archive authentication remains limited to journaled rows.

At 2026-10-19 12:21 UTC, five more finalized media files passed the local
diagnostic gates in two incremental receipts:
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T122034Z.json`
(SHA-256 `900daf4be73944c061091786dc391b6f0cd9915ef7f2c7e7420fa8220073e632`)
and
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\unlogged-media-delta-20261019T122121Z.json`
(SHA-256 `1968bd7545faf46b355eff8610cc4ca2bae75a88c742a57b1599753d8c9e3120`).
Together they add five previously unseen media IDs; all pass local hashing and
metadata checks. The pinned journal snapshot still has 467 rows, so archive
byte authenticity remains established only for those rows. A partial audit of
the pinned snapshot at 12:20:54Z observed 680 finalized files and 213 files not
represented by that snapshot; the live tree then continued changing. At
12:21:40Z the live media tree had 681 MP4 files and six `.part` files totaling
318,767,104 bytes. PID 23276 was still present at 12:21:44Z. These live counts
are observations during acquisition, not a final manifest; do not infer
completion or restart the downloader from them. Local diagnostic coverage is
214 unjournaled IDs (213 local passes and the known timing-boundary failure),
subject to a fresh reconciliation after acquisition stops.

At 12:22:47Z, one incremental audit script omitted the immediately prior
receipt from its already-seen set, so its four-row delta included two IDs that
had been checked before. This receipt was preserved and not counted as four
new IDs. Subsequent audits included the missing receipt and deduplicated IDs.
The corrected cumulative diagnostic coverage at 12:24 UTC is 219 unjournaled
IDs (218 local passes and the one known timing-boundary failure), matching the
latest pinned-snapshot audit: 467 journal-authenticated files plus 219 files
not represented in that snapshot. The audit output
`audit-live-snapshot-20261019T122332Z.json` has SHA-256
`A00265B99D674FCA7FD1B7E6F601FC7F928F57DFF53207F3BADFF3CC7F6D7AB2` and
records six partial files totaling 432,013,312 bytes. One additional media
file passed local checks in
`unlogged-media-delta-20261019T122400Z.json` (SHA-256
`e25e28456af5f2654f3f04eb7a07969d91d3ff7c4e9cb17ff2c55133cbc2b2a2`).
PID 23276 remained active at 12:24 UTC. Acquisition is still in progress;
archive authentication is still limited to the 467 pinned journal rows.

At 12:25:28Z the live media tree had 689 MP4 files and six partial files.
The pinned-journal audit therefore found 222 finalized files without archive
journal authentication; deduplicated local diagnostic coverage matches all
222 (221 pass and the one known timing-boundary failure). Three newly observed
IDs passed local checks in
`unlogged-media-delta-20261019T122528Z.json` (SHA-256
`0217296d4e8bc701231f080a0a5b718626d3a11fc9a5d00d6eda7efc1f4b79ce`). PID
23276 remained active. The partial-file byte total is transient while workers
write; this observation measured 75,497,472 bytes. CI for the documentation
commit `2b07dbd25de3d000aec0af30022e41f62a403bc9` completed successfully.


At 12:27:12Z, the latest pinned-snapshot audit recorded 692 finalized MP4s,
467 journal-authenticated and 225 not represented by the pinned journal. Its
SHA-256 is `8DCCD6EF528C48EE7EB2027060F623833AFBE1E028B991A3BEFC6DEA0E7858FD`.
Deduplicated local diagnostics cover all 225 unjournaled IDs (224 pass, one
known timing-boundary failure). Three newly found IDs passed in
`unlogged-media-delta-20261019T122659Z.json` (SHA-256
`2f199f0a779688ac93b3dbf9b5b327c30499926d899a5eef370459c3e87a26a7`). Six
partial files totaled 260,046,848 bytes at the audit snapshot. The downloader
was confirmed alive at 12:26:59Z; the subsequent process query returned no row,
so its state after that time is UNKNOWN and must be checked again before any
assumption about termination. The latest repository commit is `2176f31` and
its CI was queued at the time of this note.

At 12:29:20Z, a fresh audit of the pinned journal snapshot recorded 467
journal-authenticated files and 228 finalized media files outside that
snapshot, matching 695 MP4s in the live tree. Snapshot SHA-256:
`1292FF0453A937E01EA0EB125F46ADAF910068DC38279443220F74E736CED5F7`.
Deduplicated local diagnostic receipts cover all 228 unjournaled IDs (227
passes; one known timing-boundary failure). The latest two delta receipts added
three IDs total, all locally passing; one receipt rechecked two IDs also found
by the immediately preceding receipt, and coverage counts are deduplicated by
session ID. Six `.part` files remained; their total size fluctuates as workers
write (260,046,848 bytes at the pinned audit). PID 23276 remained alive at
12:29 UTC. CI for commit `95ec716782a6692d2129554a1eba0b84e604bdde` completed
successfully.

At 12:31:47Z the pinned-journal audit recorded 467 journal rows and 231
finalized media files not represented by that journal (audit SHA-256
`89BC4C0BD1D3CDAB5BBDDB48DF1FBB6B4E089B04DD029CC23BF8FDDD5CE39167`). The
live file tree contained 698 MP4s, all selected from the 1,042-record train
partition; none of the 264 validation assets had finalized yet, and no
unselected IDs were found. Deduplicated local receipts cover all 231
unjournaled train IDs (230 local passes and one known timing-boundary failure).
A new ID passed local checks in
`unlogged-media-delta-20261019T123147Z.json` (SHA-256
`06734761e07da45e2c5d20423199e02cc1e241e66527638e0314983b63c8e92f`). Six
partial files remained (406,847,488 bytes at the audit instant); PID 23276
was active. Archive hash authentication still covers only the 467 pinned
journal rows.

At 12:34:12Z the pinned-journal audit saw 700 finalized MP4s: 467 in the
pinned journal and 233 not represented there. Audit SHA-256:
`6D3323D7776C54EADA70FB2EBA23E23D515378984475D6AF98D1E2A63DBC9A83`.
Deduplicated local diagnostics cover all 233 unjournaled IDs (232 pass; one
known timing-boundary failure). The live tree held 700/1,042 train MP4s and
0/264 validation MP4s, with no unselected IDs. Two new local passes are in
`unlogged-media-delta-20261019T123330Z.json` (SHA-256
`00dccceae2374ff09260ea102160c005528a32cf3f8e02e111f5214cc9c0729d`). Six
partial files totaled 562,036,736 bytes during the audit. PID 23276 was still
active. CI for repository commit `789968eca7f80dcb6416e922f070bdd2036e25c9`
completed successfully. Only the pinned 467 journal rows are source-hash
authenticated.

At 12:35:24Z the pinned-journal snapshot audit observed 704 finalized selected
files, 467 journal-authenticated and 237 unjournaled. Audit SHA-256:
`8D9E1394B35DDB02540C8E2F26E504D392411261D8B9C85CB8E88D7EAEBFEE71`.
Its transient partial count was five (423,624,704 bytes). The tree changed while
workers were running: at 12:35:48Z it showed 704 MP4s and six partial files.
All 237 unjournaled IDs now have deduplicated local diagnostic coverage (236
pass; the known timing-boundary failure remains). The newly added local pass is
`unlogged-media-delta-20261019T123548Z.json` (SHA-256
`ecfcf56e78f5b0b2eb82068480390a1863b0356e3900992493c41d271bed3d0d`). PID
23276 remained active. The latest prior documentation commit `5cb2f84` CI is
green; current unjournaled items remain unauthenticated against the source
archive.

At 12:39:01Z a robust reconciliation scanned every `unlogged-media-*.json`
receipt in the attempt directory rather than relying on the incremental
scripts' manually extended receipt lists. The reconciler output
`live-receipt-reconciliation-20261019T123843Z.json` has SHA-256
`04cedf695de50ec6d2b0dfaa356d23a15ba124d0c76253d030a7c151d28290ef`. It
matched 711 current MP4s to the frozen selection: 711 train, zero validation,
zero unselected, and no selected-size mismatches. The pinned journal remains
467 rows; 244 currently unjournaled media IDs are all covered by deduplicated
local receipts (243 passes and the one known timing-boundary failure). The
reconciler collapsed 23 repeated receipt rows and found no conflicting local
hashes or uncovered IDs. Two files discovered after the prior delta chain were
locally checked in `unlogged-media-delta-20261019T123843Z.json` (SHA-256
`e1d7d50540007f7de117e757bb0b4087e70a8d390eabbc5a6e5c4d03feb91881`). This
local identity/size/media check does not authenticate those 244 files against
the source archive. PID 23276 remains active; validation downloads and the
full media gate remain incomplete.

At 12:41:21Z, an all-receipts reconciliation confirmed 716 finalized selected
MP4s, all in train, with no validation files yet. Output:
`live-receipt-reconciliation-20261019T124114Z.json` (SHA-256
`dae09001d86f1183e7c24c28433d3be0c65b0b8c39e2a50e1e0a61a5ec4ee8a6`). Of
these files, 467 are represented in the pinned journal and all 249 unjournaled
IDs have deduplicated local diagnostic receipts (248 pass, one known
timing-boundary failure). It reports zero unselected IDs, size mismatches,
receipt hash conflicts, or uncovered unjournaled IDs; repeated receipt rows
are deduplicated. Three new local passes were recorded in
`unlogged-media-delta-20261019T124056Z.json` (SHA-256
`ea987125addd72c9e2ba4bc1916720072fd2129622b467836438121b27757f79`). Six
partial files remain, PID 23276 is active, and the prior commit `2ead8bd` CI
is green. Unjournaled media still lacks archive source-hash authentication.

At 12:44:58Z, the all-receipts reconciliation output
`live-receipt-reconciliation-20261019T124443Z.json` (SHA-256
`c9e092a854f6eff044677bbf7667293040817941f7899ddcfa05401a306f34bf`)
matched 721 finalized MP4s to selected train IDs (721/1,042 train; 0/264
validation; zero unselected files, wrong sizes, conflicting receipt hashes or
uncovered unjournaled IDs). It covered 254 unjournaled IDs with local
receipts (253 local passes and the known timing-boundary failure). Four newly
observed files passed local checks in
`unlogged-media-delta-20261019T124443Z.json` (SHA-256
`01d32a86a38da84742ec2b989c86910f0f799d41abf882a990e144f4e48b9f2a`).
The download process PID 23276 is alive, has the progress journal open, holds
four `.part` files open and has four established HTTPS connections to the
archive endpoint. The live journal's length/mtime remain 218,310 bytes and
2026-10-19 10:07:15Z despite the later finalized files. The cause is UNKNOWN;
this is a provenance/reproducibility concern, so the 254 locally checked files
are not claimed to have source-archive hash authentication. Do not stop or
restart the active downloader on this evidence alone. The journal snapshot
pinned at 10:55Z still authenticates 467 files. CI for `19c3920` passed.

At 12:46:19Z, all-receipt reconciliation
`live-receipt-reconciliation-20261019T124600Z.json` (SHA-256
`65cbf62913cd9c6f830967750ac5f91a3ddcaf5920a38ada06c1592e7daba39e`)
matched 723 MP4s to selected train records (723/1,042 train; 0/264 validation)
with no unselected files, size mismatches, receipt-hash conflicts or uncovered
IDs. The pinned journal remains 467 rows; local receipts cover 256 unjournaled
files (255 passes, the known timing-boundary failure unchanged). Two newly
observed videos passed local checks in
`unlogged-media-delta-20261019T124600Z.json` (SHA-256
`81f60327dfcfa34905a8b4f402a74a42fdfc3b11132d9987a1c87d0bad0be484`). Seven
partial files were present, and PID 23276 remained active. The journal's size
and mtime are still unchanged at 218,310 bytes / 10:07:15Z while the process
holds it open; the cause remains UNKNOWN. CI for `ed35df0` completed
successfully. Unjournaled local checks still do not authenticate source bytes.

At 12:48:20Z the all-receipts reconciliation
`live-receipt-reconciliation-20261019T124806Z.json` (SHA-256
`0291ea94153b9141ded3ea641cd9803440335361c87f77e58ffb1cb857ee40e1`)
matched 727 finalized train MP4s (727/1,042 train; 0/264 validation) to the
frozen selection. It found no unselected files, size mismatches, receipt hash
conflicts or uncovered unjournaled IDs. The journal still authenticates 467;
260 files have local receipts (259 pass, one known timing-boundary failure).
Two new local passes are in
`unlogged-media-delta-20261019T124753Z.json` (SHA-256
`48d3b0958f9b752828f3e67bb7be433b711cd24a75299b60297eab4f725200d0`).
At 12:48 UTC PID 23276 remained running, held the progress journal and four
`.part` files open, and had four established HTTPS connections. The journal
remained 218,310 bytes with mtime 10:07:15Z.

A code-path hypothesis for the stale journal is now recorded, not treated as
confirmed: `scripts/download_holoassist_selected_media.py:224-229` iterates
`as_completed` but calls `future.result()` without per-future exception
handling. A failed future would unwind the result/journal loop; the executor
context then waits for remaining submitted tasks, which may continue writing
media without the parent loop recording their results. This matches the
observed combination, but the actual exception and whether this exact path was
taken are UNKNOWN because the original terminal output is unavailable. Keep
PID 23276 untouched; verify its terminal state before any restart or recovery.
CI for commit `0068294` completed successfully.

At 12:51:25Z, all-receipt reconciliation
`live-receipt-reconciliation-20261019T125110Z.json` (SHA-256
`dcc07efc682f53b1693d89be71c88dbe2553854042f52273469e88bb401036b4`)
matched 730 current MP4s to the frozen selection: 730 train, zero validation,
zero unselected, zero wrong-sized, and zero uncovered unjournaled IDs. The
pinned journal remains 467 rows, leaving 263 unjournaled files with local
receipts (262 passes and the one known timing-boundary failure). One additional
local pass is recorded in `unlogged-media-delta-20261019T125110Z.json` (SHA-256
`4bbe00110bddec8437a46b57c7c25b6386ff0626c08aca1d27922d22fc990392`). PID
23276 remained present. The fixed journal snapshot is still the only source
hash-authenticated segment. CI for `fe9c07c` completed successfully.

At 12:53:10Z, the all-receipts reconciliation
`live-receipt-reconciliation-20261019T125233Z.json` (SHA-256
`d6cbc1c5a6a1afa2581175d69fda60ca771511338cc0da428d800a1aa74a369b`)
matched 733 finalized MP4s to the frozen selection (733 train, zero validation,
zero unselected, zero size mismatches, and zero uncovered unjournaled IDs).
The pinned journal remains 467 rows; local receipts cover all 266 unjournaled
files (265 pass and the one known timing-boundary failure). Three latest
locally passing rows are in `unlogged-media-delta-20261019T125233Z.json` (SHA-256
`2d93ebb1ef36cd7ee679b558310f74d724c8861ad7647a70f137b65d3d98c539`). Seven
partial files remain. PID 23276 is still present and the live progress journal
is unchanged at 218,310 bytes / 10:07:15Z, so source authentication remains
limited to the pinned 467 rows. CI for commit `5028f51` completed successfully.

At 12:55:03Z, the all-receipts reconciliation
`live-receipt-reconciliation-20261019T125500Z.json` (SHA-256
`6517556d09320372117df0aac5e1cb8f87527709f3ae85187106257161b059b7`)
matched 738 finalized MP4s to the frozen selection (738 train, zero validation,
zero unselected, zero size mismatches, and zero uncovered unjournaled IDs).
The pinned journal remains unchanged at 467 rows; all 271 unjournaled files
have local receipts (270 pass, one known timing-boundary failure). Five new
files locally passed metadata diagnostics in
`unlogged-media-delta-20261019T125500Z.json` (SHA-256
`f6153fdaeddb73f7056c95c106d874587dad57f84c6255968e5d7b7df93a3c90`). The
receipt directory contained no conflicting local hashes. The reconciler saw
seven partial files, while a later live count saw six, confirming these counts
are race-sensitive. PID 23276 remains active and the live journal remains
218,310 bytes with mtime 10:07:15Z. Therefore source-content authentication is
still limited to the 467 pinned journal rows; local receipts are not official
archive hashes. CI for `5430e9b` completed successfully. Acquisition is
incomplete; do not intervene in the downloader while it remains live.

At 12:58:58Z, the all-receipts reconciliation
`live-receipt-reconciliation-20261019T125900Z.json` (SHA-256
`9ab0090d2a2b7061d9d5772372c2d223a1a9b1922c8f43082c2c460c7e82286f`)
matched 741 finalized MP4s (all train; zero validation) to the frozen selection.
It found no unselected files, wrong sizes, uncovered unjournaled IDs, or local
receipt hash conflicts. The 274 files outside the pinned journal are covered by
local receipts: 273 pass, one is the known timing-boundary failure. Three new
local diagnostics passed in `unlogged-media-delta-20261019T125900Z.json`
(SHA-256 `03781cdaf010991d748f3976a19725aefedcc6fae818a868fe569b7d9a4d0745`).
PID 23276 remains active. The 467-row pinned journal is unchanged, so official
source-content authentication remains limited to those 467 rows. Seven partial
files totaling 128 MiB were present. CI run 39255891055 for `17b3d0c` completed
successfully.

At 13:00:06Z, full-receipt reconciliation
`live-receipt-reconciliation-20261019T130020Z.json` (SHA-256
`b22143fb7aaf432adb836f8db95e11cfa7d5f7aeded804d588d46872fd7d04cd`) matched
744 finalized MP4s, all from train, to the frozen selection. There are zero
validation MP4s, unselected files, size mismatches, uncovered unjournaled IDs,
or local receipt hash conflicts. Local receipts cover all 277 IDs outside the
pinned journal (276 pass, one known timing-boundary failure); only 467 files
are source-hash authenticated by the journal. Three new local passes are in
`unlogged-media-delta-20261019T130020Z.json` (SHA-256
`fa61b7703c99b911d93cf17a361d382541aede5ab1f9fe5c418096635f08b93f`). PID
23276 remains active; seven partial files total about 72 MiB, down from 128 MiB
at 12:59. CI for `7e0c392` was still in progress at the last check.

At 13:01:51Z, the all-receipts reconciliation
`live-receipt-reconciliation-20261019T130120Z.json` (SHA-256
`7910b181e5dc43fde123f198e7e5698639a93018e0b61484f29fc4a9eab93b74`) matched
749 finalized MP4s, all train, to the frozen selection. Validation remains
0/264. No unselected files, wrong sizes, uncovered unjournaled IDs, or local
receipt hash conflicts were found. All 282 files outside the pinned journal
have local receipts (281 pass; one known timing-boundary failure). Five new
local checks passed in `unlogged-media-delta-20261019T130120Z.json` (SHA-256
`a54868ce0c43513815b52f370338d1629adc86adf5b0138845b8d8affff34f29`). Source
archive hash authentication remains limited to 467 journal rows. PID 23276 is
still active with seven partial files; transfer size changes were observed
during polling. CI run 39256499356 for `1e7ac47` completed successfully.

At 13:03:21Z, full-receipt reconciliation
`live-receipt-reconciliation-20261019T130315Z.json` (SHA-256
`b3cecdb9533e7213f1c597ffa5ab12e88dbc081314cbff6b7a6e277c7f03c240`) matched
751 finalized MP4s, all train, to the frozen selection. Validation remains
0/264. No unselected files, size mismatches, uncovered journal-external IDs,
or local receipt hash conflicts were found. The 284 IDs outside the pinned
journal have local receipts (283 pass; one known timing-boundary failure).
Two new local checks passed in `unlogged-media-delta-20261019T130315Z.json`
(SHA-256 `883c0e4f85809bcacb602efec4b755779592fe7f4d34d9b3502f514e6f9114be`).
The pinned journal remains at 467 rows, the only source-hash-authenticated
segment. PID 23276 remains active with seven partials. CI run 39256769531 for
`0b876b8` completed successfully.

At 13:05:24Z, full-receipt reconciliation
`live-receipt-reconciliation-20261019T130520Z.json` (SHA-256
`d21af66879794d4d5ce9670d3867e8594d0bcd957a08ad906ed7b592348209a1`) matched
757 finalized MP4s, all train, to the frozen selection (validation 0/264).
There are no unselected files, size mismatches, uncovered journal-external IDs,
or local receipt hash conflicts. All 290 journal-external files have local
receipts (289 pass; the same known timing-boundary failure). Six newly found
files passed local checks in `unlogged-media-delta-20261019T130520Z.json`
(SHA-256 `37bcaebd8e14506a9df6d09972c041f5b962c7f9202d85a19efb1c0178269dde`).
The 467-row pinned journal remains the only source-hash-authenticated set. PID
23276 remains active with seven part files. CI run 39256769531 for `cb3e850`
completed successfully.

At 13:07:00Z, all-receipts reconciliation
`live-receipt-reconciliation-20261019T130700Z.json` (SHA-256
`c9c256a141121c14263f4ac62a2cd7f5acf69a0810e0f665652d3c553634e57f`) matched
760 finalized MP4s, all train, to the frozen selection (validation 0/264). No
unselected files, wrong sizes, uncovered journal-external IDs, or local receipt
hash conflicts were found. Local receipts cover all 293 journal-external IDs
(292 pass; one known timing-boundary failure). Three newly observed media files
passed local diagnostics in `unlogged-media-delta-20261019T130700Z.json`
(SHA-256 `2c0d4e6e438b29b64226b75c8ac17947455b1d855f1f0453e9ba40c75c78fc1e`).
The journal remains at 467 source-hash-authenticated rows. PID 23276 remains
active. CI run 39257236413 for `f5c4f8e` completed successfully.

At 13:07:53Z, all-receipts reconciliation
`live-receipt-reconciliation-20261019T130800Z.json` (SHA-256
`cc9fe750258b54bb239e4519991feaec4f992c6d610a30ffa673057972a071d0`) matched
764 finalized MP4s, all train, to the frozen selection. Validation remains
0/264. There are no unselected files, size mismatches, uncovered journal-
external IDs or local receipt hash conflicts. Local receipts cover all 297
files outside the pinned journal (296 pass; one known timing-boundary failure).
Four new media files passed local checks in
`unlogged-media-delta-20261019T130800Z.json` (SHA-256
`98e26b3e2eee6567df7e62d2b70a7bc1551c0b0b29cc9d21b509325572e26540`). Source
hash authentication remains limited to the unchanged 467-row journal. PID 23276
remains active with seven partials.

At 13:09:12Z, all-receipts reconciliation
`live-receipt-reconciliation-20261019T131000Z.json` (SHA-256
`2e069833e5a1fd9715553a6773196285b94ec32b6900d278846f9b3f84858e61`) matched
767 finalized MP4s, all train, to the frozen selection. Validation remains
0/264. No unselected files, size mismatches, uncovered IDs or receipt hash
conflicts were found. Local receipts cover all 300 files outside the pinned
journal (299 pass; one known timing-boundary failure). Three new locally passing
files are recorded in `unlogged-media-delta-20261019T131000Z.json` (SHA-256
`c9bd9899c647ec47b3ff9db7435328d1ad3e03e7e8a496e1e7e2068395c087f1`). Official
source-hash authentication remains limited to 467 journal rows. PID 23276
remains active. A subsequent live count was 768. CI run 39257573719 for
`581ae49` completed successfully.

At 13:09:12Z, full-receipt reconciliation
`live-receipt-reconciliation-20261019T131000Z.json` (SHA-256
`2e069833e5a1fd9715553a6773196285b94ec32b6900d278846f9b3f84858e61`) matched
767 finalized MP4s, all train, to selection. Validation remains 0/264. The 300
IDs outside the pinned 467-row journal have local receipts (299 pass, one known
timing-boundary failure); there are no unselected files, size mismatches,
uncovered IDs or receipt hash conflicts. Three new local passes are recorded in
`unlogged-media-delta-20261019T131000Z.json` (SHA-256
`c9bd9899c647ec47b3ff9db7435328d1ad3e03e7e8a496e1e7e2068395c087f1`). PID
23276 remains active. A later live count was 768. CI for `9ed49d8` was shown as
queued in the public Actions UI; the unauthenticated REST endpoint was rate
limited.

At 13:11:38Z, all-receipts reconciliation
`live-receipt-reconciliation-20261019T131400Z.json` (SHA-256
`a24f955777a4ebed6a26ecf22649b68801b59020f6263ebc75d83fc12fb4c22d`) matched
774 finalized MP4s, all train, to selection (validation 0/264). There are no
unselected files, size mismatches, uncovered journal-external IDs, or receipt
hash conflicts. Local receipts cover all 307 journal-external files (306 pass;
one known timing-boundary failure); seven newly observed media passed local
checks in `unlogged-media-delta-20261019T131400Z.json` (SHA-256
`374d221596112872b99d7e480f67a05d6f284f073a739b1b8644a1e97a8d777f`). The
pinned journal remains unchanged at 467 rows, the only source-hash-authenticated
segment. PID 23276 remains active. The public Actions page for `9460805` still
showed pending when checked.

At 13:13:37Z, full-receipt reconciliation
`live-receipt-reconciliation-20261019T131340Z.json` (SHA-256
`b5bebdf5f29631e9b00a98b145ef25f7bced56a5038f679e19fef95dc09011cb`) matched
781 finalized MP4s, all train, to selection (validation 0/264). There are no
unselected files, size mismatches, uncovered journal-external IDs, or local
receipt hash conflicts. The 314 IDs outside the pinned 467-row journal have
local receipts (313 pass; one known timing-boundary failure). Seven new local
passes are in `unlogged-media-delta-20261019T131340Z.json` (SHA-256
`ec58b0dbdab9331f4c35e7e89d5badf64b3f678a5a664439258fa658e5f78cdb`). PID
23276 remains active. The public Actions run 39258112400 for `b8db4a9` is now
green.

At 13:14:59Z, full-receipt reconciliation
`live-receipt-reconciliation-20261019T131500Z.json` (SHA-256
`a654ea78419eaa26552c5fb9c68c96ccc502a2ff9b0d691fb6e2d3bb7eda9e5a`) matched
784 finalized MP4s, all train, to selection (validation 0/264). No unselected
files, size mismatches, uncovered journal-external IDs, or local receipt hash
conflicts were found. All 317 files outside the pinned journal have receipts
(316 pass; one known timing-boundary failure). Three newly checked files passed
local diagnostics in `unlogged-media-delta-20261019T131500Z.json` (SHA-256
`d1b5b42a7a8d9f6f97608a7b788f6e24696b656ae97fdb2b0f5cfba6e1726575`). The
pinned journal is unchanged at 467 source-hash-authenticated rows. PID 23276
remains active. CI run 39258384882 for `1da21c2` completed successfully.

At 13:17:10Z, full-receipt reconciliation
`live-receipt-reconciliation-20261019T131720Z.json` (SHA-256
`52e147e272e4ed284a1c34ff8f64a32f77886aeacabff3752de561a0f5684bf2`) matched
787 finalized MP4s, all train, to selection (validation 0/264). No unselected
files, size mismatches, uncovered journal-external IDs, or local receipt hash
conflicts were found. Local receipts cover all 320 IDs outside the pinned
journal (319 pass; one known timing-boundary failure). Three new files passed
local diagnostics in `unlogged-media-delta-20261019T131720Z.json` (SHA-256
`68c3ddb6440cab4db2a7a46e017cfac5752522f7eb1ebb6f91de6b6d46138d79`). Source
hash authentication remains limited to 467 pinned journal rows. PID 23276
remains active. Public Actions run 39258678240 for `33b1418` completed
successfully.

At 13:20:03Z, all-receipts reconciliation
`live-receipt-reconciliation-20261019T132000Z.json` (SHA-256
`227858fc36c69a0c4c6fdcf6a3e1eac51e3fccfaa4ad9425c8ffa01a114c63d4`) matched
790 finalized MP4s, all train, to the frozen selection (validation 0/264). No
unselected files, size mismatches, uncovered journal-external IDs or receipt
hash conflicts were found. Local receipts cover 323 journal-external files
(322 pass; one known timing-boundary failure). Three new media files passed
local checks in `unlogged-media-delta-20261019T132000Z.json` (SHA-256
`90c8fdaa7ab3dd1f24c3507e5e5233ff1519305f0f98e53c6f1be2fb30c9b9ab`). The
pinned journal remains at 467 source-hash-authenticated rows. PID 23276 remains
active. CI run 39258893455 for `1394430` completed successfully.

At 13:21:42Z, all-receipts reconciliation
`live-receipt-reconciliation-20261019T132200Z.json` (SHA-256
`453f5111de5d32f26859a107539920649d499aa132be35a0805f97b843083c4c`) matched
793 finalized MP4s, all train, to selection (validation 0/264). No unselected
files, size mismatches, uncovered journal-external IDs, or local receipt hash
conflicts were found. The 326 journal-external files have local receipts (325
pass; one known timing-boundary failure). Three new local passes are recorded
in `unlogged-media-delta-20261019T132200Z.json` (SHA-256
`3567fa12bbbefa4cff91db14687e1d9153634681234c6477c7e8b8eb6361966d`). The
unchanged pinned journal authenticates 467 source hashes. PID 23276 remains
active with four established HTTPS connections at a subsequent poll. CI run
39259299140 for `f9e2079` completed successfully.

At 13:22:27Z, full-receipt reconciliation
`live-receipt-reconciliation-20261019T132600Z.json` (SHA-256
`bebb79197d99c832030e7a9981b4ca3747bc09d88763ba4453aa2ce8f435d55f`) matched
796 finalized MP4s, all train, to the frozen selection (validation 0/264). No
unselected files, size mismatches, uncovered journal-external IDs or receipt
hash conflicts were found. Local receipts cover 329 files outside the pinned
journal (328 pass; one known timing-boundary failure). Three newly observed
files passed local checks in `unlogged-media-delta-20261019T132600Z.json`
(SHA-256 `40e88f4ab47bc914119da7c93c548d7c73bb0f1af7b31ddac97cce74800cd314`).
The 467-row journal is still the only archive-source-hash-authenticated set.
PID 23276 remains active. CI run 39259537046 for `fb9b41f` completed
successfully.

At 13:24:01Z, all-receipts reconciliation
`live-receipt-reconciliation-20261019T132400Z.json` (SHA-256
`eca33f1e1c84d8885d176ba48bcd3db7b19954c3ed02cbe27dce5cafd8210856`) matched
798 finalized MP4s, all train, to selection (validation 0/264). It found no
unselected files, size mismatches, uncovered journal-external IDs, or local
receipt hash conflicts. Local receipts cover all 331 journal-external files
(330 pass; one known timing-boundary failure). Two new media files passed local
checks in `unlogged-media-delta-20261019T132400Z.json` (SHA-256
`7e0fec036126a3b4d730a50c274082e0f2b91f5acf5abd26e8bd22053b6f0ae0`). The
pinned journal still authenticates 467 source hashes. PID 23276 remains active.
CI run 39259653563 for `fea7fc7` completed successfully.

At 13:25:58Z, full-receipt reconciliation
`live-receipt-reconciliation-20261019T132550Z.json` (SHA-256
`0790c427df6ef9fe4ab4d9f916d107b3ea19569f7031d204afec153ccc5ab081`) matched
801 finalized MP4s, all train, to selection (validation 0/264). No unselected
files, size mismatches, uncovered journal-external IDs, or local receipt hash
conflicts were found. All 334 files outside the pinned journal have local
receipts (333 pass; one known timing-boundary failure). Three new media files
passed local diagnostics in `unlogged-media-delta-20261019T132550Z.json`
(SHA-256 `efbe6c7ffbbf0df2b59133f28cd97b080a511fe28a97fc61d856eebfd03d24f5`).
The 467-row pinned journal remains the only archive-source-hash-authenticated
set. PID 23276 remains active. CI for `e089d89` was green.

At 13:27:28Z, all-receipts reconciliation
`live-receipt-reconciliation-20261019T132740Z.json` (SHA-256
`8291c4ba95264fc5de30d7e688ee13099d550d473886282ef809282aea4ca3e0`) matched
806 finalized MP4s, all train, to selection (validation 0/264). There are no
unselected files, size mismatches, uncovered journal-external IDs, or local
receipt hash conflicts. Local receipts cover all 339 files outside the pinned
journal (338 pass; one known timing-boundary failure). Five new local passes
are in `unlogged-media-delta-20261019T132740Z.json` (SHA-256
`378c1487bad7429394b07acaad2809fb86f241faef33a69f8aa6f8088dca3f2b`). The
pinned journal remains the only source-hash-authenticated set and has 467 rows.
PID 23276 remains active. CI run 39260131456 for `7c2426d` completed
successfully.

At 13:29 UTC a read-only poll again found PID 23276 alive, with four established
HTTPS sockets. Over the observed 20-second process-I/O sample, CPU increased
0.1875 seconds and process disk read/write counters did not change; the broader
interval showed no media change for about 45 seconds. Counts remained 808 MP4s
and seven partials totaling 100,663,296 bytes. This supports only a transient
quiet interval; the live handle remains authoritative and the downloader must
not be treated as finished or restarted.

At 13:31:33Z, all-receipts reconciliation
`live-receipt-reconciliation-20261019T133130Z.json` (SHA-256
`3df139e24d7e60656ea6ed41a6bbc55fef24b9938bc623c72595f7d466e1a0b9`) matched
811 finalized MP4s, all train, to selection (validation 0/264). No unselected
files, size mismatches, uncovered journal-external IDs, or local receipt hash
conflicts were found. Local receipts cover all 344 journal-external files
(343 pass; one known timing-boundary failure). Five new files passed local
checks in `unlogged-media-delta-20261019T133130Z.json` (SHA-256
`c7f5320cae658b86e14184c279bc2e5ac450bc9cca5ebe616933c02279f1fec5`). The
pinned journal remains the only source-hash-authenticated set and has 467 rows.
PID 23276 remains active with four HTTPS connections. CI run 39260671335 for
`dfe96cd` completed successfully.

At 13:33:13Z, full-receipt reconciliation
`live-receipt-reconciliation-20261019T133300Z.json` (SHA-256
`ec99d48437672fc772354f1009c53984280f22706fb3a498acbe4f0fe0b80d0b`) matched
813 finalized MP4s, all train, to the frozen selection (validation 0/264).
There were no unselected files, size mismatches, uncovered journal-external
IDs, or local receipt hash conflicts. The 346 files outside the pinned journal
have local receipts (345 pass; one known timing-boundary failure). Two new local
passes are recorded in `unlogged-media-delta-20261019T133300Z.json` (SHA-256
`736b84086984f4803b4633a0d67ff4e30f20da87ddad0d425f32567556d7ec00`). Official
source-hash authentication remains limited to the unchanged 467-row journal.
PID 23276 remains active. CI run 39260948250 for `0f1f174` completed
successfully.

At 13:35:28Z, full-receipt reconciliation
`live-receipt-reconciliation-20261019T133540Z.json` (SHA-256
`572d0c672f7c8e56f27ffdfec74d61517efc57bdd26531880a54401ae5aa4a3e`) matched
815 finalized MP4s, all train, to the frozen selection (validation 0/264). No
unselected files, size mismatches, uncovered journal-external IDs, or local
receipt hash conflicts were found. The 348 files outside the pinned journal
have local receipts (347 pass; one known timing-boundary failure). Two newly
checked files passed in `unlogged-media-delta-20261019T133540Z.json` (SHA-256
`2da6618855a746cf391dcadbc476fdb28d48b5ab551f58826c040f945e6b9e9f`). The
467-row pinned journal remains the only archive-source-hash-authenticated set.
PID 23276 remains active; seven partials were present. CI run 39261188551 for
`2deadae` completed successfully.
