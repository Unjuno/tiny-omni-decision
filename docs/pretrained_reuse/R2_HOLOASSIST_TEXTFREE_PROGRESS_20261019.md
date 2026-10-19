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
