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
