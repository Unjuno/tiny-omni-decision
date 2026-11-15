# FSD50K annotation-Decision acquisition and verification — 2026-11-15

## Scope

This is a narrow Path A audio task: predict whether the frozen FSD50K dev
annotation assigns one of four supplied broad labels. A negative means “not
annotated in this release,” not verified acoustic absence. Only the frozen,
uploader-grouped dev manifest is eligible. The official eval labels and audio
remain unopened and unused.

## Frozen inputs

- Dataset revision: `10.5281/zenodo.4060432`.
- Preflight: `C:/CodexArtifacts/pretrained-reuse-r2/fsd50k-annotation-presence-v1-preflight-20261115-attempt-01/`.
- Preflight manifest SHA-256: `0bbe5db5e6167a4c041e349889920f8da0f1a0703ee00fa63ff12a59a0531b6d`.
- Split counts: train 12,845 / calibration 2,053 / validation 3,981.
- Uploader counts: train 2,089 / calibration 290 / validation 572.
- All 129 previously selected IDs and their 128 uploader groups are excluded.
- The archive central-directory SHA-256 is
  `a52c8e77acf8c01e5108a3872461a65f2a583b2ddc3e0bb6554e51dc7ae12c5f`.
- The frozen sample IDs resolve to 18,879 WAV members, 12,352,131,640 expanded
  bytes, and a planned maximum range transfer of 8,867,277,632 bytes. C: had
  over 279 GB free when acquisition began; the downloader preserves a 2 GiB
  reserve.

## Source-duration discrepancy

The FSD50K paper describes all clips as 0.3–30 seconds, uncompressed PCM
16-bit, 44.1 kHz mono. ZIP central-directory sizes and seven bounded WAV-header
reads show that the frozen cohort contains **39** files over 30 seconds by
that declared encoding: 20 train and 19 validation, from 17 uploader groups;
none are in calibration. The largest inspected header corresponds to about
550.25 seconds. These are official dev members and have allowed per-clip CC BY
3.0 or CC0 licenses. They are retained in the frozen manifest; this stage
does not truncate or silently remove them. Whole-clip feature extraction must
process them safely or fail closed before fitting. Full decoded-audio duration
validation remains part of the media verification.

## Acquisition attempts

| Attempt | Result | Evidence |
|---|---|---|
| `...media-...-attempt-01` | FAIL before saving any WAV | Windows Python lacks `os.statvfs`; the archive index also revealed a selected member larger than the original 5 MiB guard. Only the directory index was read. |
| `...media-...-attempt-02` | FAIL before saving any WAV | ZIP local filename length and extra-field length were read in reverse order. The new output contains an empty receipt and the failure record. |
| `...media-...-attempt-03` | FAIL at 2026-11-15 02:38 JST | 445 WAVs / receipt rows; terminal `result.json` records `ValueError('range length mismatch')`. Official eval access flags remain false. |
| `...media-...-attempt-04` | FAIL at 2026-11-15 03:37 JST | Fresh output resumed the verified 445-file attempt-03 prefix. Six short-range attempts failed at bytes 902,885,557–903,241,699 (`337,620` received of `356,143` expected). It ended with 888 receipt rows / WAVs and 651,253,684 bytes. |
| `...media-...-attempt-05` | FAIL at 2026-11-15 03:46 JST | Fresh output copied the 888 verified attempt-04 files, then hit the same byte range after six attempts (`347,216` received of `356,143`). No additional complete receipt row was added; 888 WAVs were preserved. |
| `...media-...-attempt-06` | RUNNING at 2026-11-15 04:35 JST | New output resumed the independently reverified attempt-04 prefix. It passed the former failure boundary and reached 1,400 durable rows / 1,401 visible WAVs; cumulative range bytes 819,180,903 and durable output bytes 1,053,367,300. |

The old reader raised a generic `range length mismatch` after its retry limit;
it did not preserve the failing response body length or member ID. A read-only
probe of the next member after the last durable receipt (`119193.wav`, volume 2,
offset 700,301,253, compressed size 337,333 bytes) succeeded on a fresh request.
This supports a transient or inconsistent range response as the likely cause,
but the exact failed response remains UNKNOWN. The failed attempt was not
restarted in place. All 445 committed receipt rows were validated against
manifest identity, WAV bytes, SHA-256 and ZIP CRC, and the exact partial WAV set
matched.

The reader retries short bodies from the same offset up to six total attempts,
uses `Accept-Encoding: identity`, validates `Content-Range`, and charges each
retry against the frozen total budget. The resume path now accepts only the
identified attempt-03 or attempt-04 failure, revalidates every partial row,
copies into a distinct output directory, confirms the same central-directory
hash, and resumes after the exact verified receipt prefix. Unit tests exercise
short-body retry and reject corrupted partial media.

Attempt-04 stopped after six consecutive short range responses for bytes
`902885557-903241699`: 337,620 bytes arrived where 356,143 were expected. Its
terminal result records 888 rows and a cumulative range transfer of
505,246,601 bytes, within the frozen transfer budget. A read-only full-prefix
verification checked all 888 receipt rows against frozen manifest
order/identity, recomputed every WAV SHA-256 and ZIP CRC, confirmed the exact
WAV path set, and matched 651,253,684 bytes. The attempt-04 result SHA-256 is
`e26b39ac9afa39fb9653bbd23e4d7cd5991d11f5419edd224c271db67abe4662`; its
receipt SHA-256 is
`7b594d268b2b9268a01cff2f900c9fa82d23838cab0625498c5746a0e9654aad5`. The
runner SHA-256 is the pinned in-use code hash
`8b7f5b08aecb109db6072f5bcc035e42defff57defa875239c1d62568ef0ad4b`.

Attempt-05 independently reproduced the same short response at the same
requested range, receiving 347,216 of 356,143 bytes on its sixth attempt. It
stopped with the unchanged 888-file receipt prefix and 651,253,684 media bytes;
its cumulative range counter was 511,346,228 bytes and result SHA-256 was
`c1c2334c34e0b3a5e30209f6a8370b615671208b2d684a26fcc875799802f1b9`. This
repeat at the identical offset indicates that retrying the full range alone is
insufficient. The reader now has a bounded 128 KiB chunk fallback for a
persistent short body, still charged to the frozen shared transfer budget. A
deterministic test reproduces the short large response and verifies recovery
via exact subranges; the actual source prefixes remain unchanged. Attempt-06
uses a distinct output, resumes attempt-04's verified 888 rows, and has passed
the previous failure boundary. At 06:51 JST it remains RUNNING under PID 5452,
with 3,000 durable receipt rows, 3,021 visible WAV files, 2,149,175,898 durable
media bytes, and 1,655,317,968 requested range bytes in the current attempt.
`progress.json` is checkpointed
every 50 rows; visible WAVs beyond its count are not treated as durable. The
process has shown recent file updates and established HTTPS connections, but
the observed pace varies enough that no completion time is promised. It remains
incomplete pending terminal PASS plus the full acquisition verifier.

## Pinned Path A audio encoder recheck

The selected EfficientAT DyMN-0.4 checkpoint is present at
`C:/CodexArtifacts/pretrained-reuse-r1/efficientat-audit-v1/dymn04_as.pt`;
its size is 8,086,973 bytes and its locally recomputed SHA-256 is
`49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`. The
upstream source checkout is clean at commit
`a425fdce92572e602a1d5634799bd9f1f2efa806`. The existing R1 release-license
follow-up records explicit upstream commercial checkpoint permission and
admits this as an ALLOW research candidate; task quality remains open. The
reusable extractor pins the same source and checkpoint, requires the measured
PyTorch/torchaudio `2.6.0+cu124` pair, and records 1,961,239 model parameters.
The active Python runtime was rechecked as Python 3.11.9, PyTorch
`2.6.0+cu124`, torchaudio `2.6.0+cu124`, and CUDA runtime 12.4. These checks
establish artifact identity and a licensed candidate, not audio-task accuracy.

## Frozen-feature Decision implementation

`scripts/train_fsd50k_annotation_decision.py` now implements the predeclared
384-D frozen feature route using four typed binary questions and the shared
`NativeFeatureDecisionScorer(384, 6, 64)` (25,089 readout parameters; 1,986,328
encoder-plus-readout parameters). It fits on train only for a fixed 20 epochs,
fits one scalar temperature on calibration only, selects the epoch by
calibrated validation macro NLL, records per-label Accuracy/NLL/Brier/ECE and
train-validation metrics, and saves separate best and final checkpoints plus
selected validation predictions. The loader verifies the frozen manifest,
complete acquisition report, feature hashes/order, encoder revision/weight
hash, temporal policy, and disjoint source/group identities. Five focused CPU
tests, including a 20-epoch synthetic save/reload smoke, pass. This is pipeline
plumbing only; no corpus-scale feature extraction, FSD fit, or task-quality
metric has been performed yet.

`scripts/extract_fsd50k_annotation_features.py` implements the pinned
whole-clip extraction path with hash-bound identity, 32-clip atomic shards,
strict resume checks, final feature/row receipts, and CUDA peak-memory
recording. Its synthetic CPU tests cover PCM decoding, whole-clip resampling
and shape validation, and receipt resume checks. The focused suite for media
acquisition, verification, Decision fitting, and feature extraction passes 26
tests, including verified shard-resume and corrupted-shard rejection; Ruff and
`git diff --check` pass. These checks do not replace
corpus-scale extraction or real task metrics.

The full local suite passes 2,931 tests with 11 skipped. The first Actions run
on the feature-extractor commit failed because the CI `dev` extra intentionally
does not install PyTorch; lint passed, while collection failed for the two
PyTorch-dependent test modules. They now use `pytest.importorskip("torch")`.
The follow-up Actions run on commit `59a3a35e9edc416495a4ee5b617046f7bdf0f75b`
passed its lint, test, and manifest-validation jobs.

One receipt-verified FSD training clip (`119192`, 4.9067 seconds, 44.1 kHz
mono) was decoded, resampled with the pinned torchaudio runtime to 32 kHz, and
passed through the exact EfficientAT DyMN-0.4 checkpoint on CPU. The complete
clip yielded finite `[1,384]` features in 0.1484 seconds; no crop or window
selection was applied. This verifies the real input-to-feature path for one
clip, not feature repeatability over the corpus or task performance. An initial
attempt relying on an empty `CUDA_VISIBLE_DEVICES` did not hide CUDA on this
Windows runtime and stopped before forward; the successful rerun explicitly
pinned the encoder to CPU and did not perform GPU inference.

Attempt-03 uses a distinct output path and a 64 MiB per-member safety cap, above
the observed 48,532,044-byte maximum. Its range-transfer budget remains bounded
by the planned 8,867,277,632 bytes. The two implementation defects were fixed
before attempt-03 and covered by tests. Focused tests currently pass: 13; Ruff
passes. The live acquisition script was not changed after launch.

## Verification gate before feature extraction

[`validate_fsd50k_annotation_acquisition.py`](../../scripts/validate_fsd50k_annotation_acquisition.py)
checks the passing terminal receipt against the exact frozen manifest, verifies
all sample IDs, split/uploader/source/license fields, file sizes, SHA-256 and
ZIP CRCs, requires the exact WAV file set, rejects cross-split exact byte
duplicates, and requires official eval access flags to remain false. Four
fixture tests cover success, corrupted media, exact cross-split duplicates,
and eval/unlisted-file rejection. This verifier must pass before any
features are generated.

The acquisition process does not use the GPU. PID 5452 was active at the
2026-11-15 06:51 JST process snapshot. At that time the RTX 3080 showed 46%
utilization and 8,270 MiB allocated by another workload, which was left
untouched.
No corpus-scale FSD50K features, model predictions, training,
calibration, validation scoring, or official eval access has yet occurred. The
focused CPU tests for acquisition, complete-media validation, extraction, and
Decision fitting pass (22 tests).

At 2026-11-15 07:41:40 JST, the same attempt-06 process (PID 5452) was still
active. Its durable progress and flushed receipt both reported 3,600/18,879
files, with 1,954,595,046 cumulative range bytes and 2,490,104,322 durable
output bytes. There were 3,614 visible train WAVs at the snapshot; this excess
is in-flight/uncommitted output and is not counted as acquired. `result.json`
did not exist. The previous durable checkpoint was 3,550 at 07:37:52, so the
observed checkpoint interval was 50 files in about 3m48s; this is a local
throughput observation, not a completion-time guarantee. Attempt-06 remains
RUNNING and is not yet eligible for the full verifier.

At 2026-11-15 07:50:16 JST, PID 5452 remained active and attempt-06 advanced
to 3,750/18,879 durable files. The flushed receipt matched; cumulative range
bytes were 1,992,357,711 and durable output bytes 2,578,810,480. One additional
WAV was visible beyond the durable count. This checkpoint followed 3,700 at
07:46:25 (50 files in 3m45s). No terminal result was present.

At 2026-11-15 07:55:18 JST, the same process remained alive at 3,800/18,879
durable files, with matching receipt count, 2,021,011,371 cumulative range
bytes, and 2,619,890,266 durable output bytes. Four WAVs were visible beyond
the durable count. This followed 3,750 at 07:50:10 (50 files in 5m08s). No
terminal result was present; a TCP connection was not observable in this
instantaneous snapshot, so no process-state conclusion is drawn from that
alone.

At 2026-11-15 08:10:45 JST, attempt-06/PID 5452 was still active at
4,000/18,879 durable files with a matching receipt. The progress snapshot
recorded 2,124,835,664 cumulative range bytes and 2,752,079,108 durable output
bytes; 11 WAVs were visible beyond the durable count. This followed 3,900 at
08:03:27 (100 files in about 7m18s). No terminal `result.json` was present.

At 2026-11-15 08:24:09 JST, PID 5452 remained active at 4,200/18,879 durable
files with a matching receipt, 2,205,238,779 cumulative range bytes, and
2,853,378,868 durable output bytes. The 4,200 checkpoint file was written at
08:23:04, following the 4,000 checkpoint at 08:10:11 (200 files in about
12m53s). No terminal `result.json` was present; this remains acquisition
progress only, not a verification or quality result.

At 2026-11-15 08:42:12 JST, PID 5452 remained active at 4,450/18,879 durable
files with a matching receipt, 2,319,658,314 cumulative range bytes, and
3,007,568,088 durable output bytes. The checkpoint was written at 08:41:04,
250 files after the 4,200 checkpoint at 08:23:04 (18 minutes). No terminal
`result.json` was present.

At 2026-11-15 09:05:22 JST, PID 5452 remained active at 4,750/18,879 durable
files with a matching receipt, 2,463,477,615 cumulative range bytes, and
3,205,138,552 durable output bytes. The checkpoint was written at 09:04:25,
300 files after the 4,450 checkpoint at 08:41:04 (23m21s). No terminal
`result.json` was present.

At 2026-11-15 09:15:03 JST, PID 5452 remained active at 4,850/18,879 durable
files with a matching receipt, 2,551,005,470 cumulative range bytes, and
3,314,349,076 durable output bytes. The checkpoint was written at 09:14:30,
100 files after 4,750 at 09:04:25 (10m05s). No terminal `result.json` was
present. C: had 259.49 GiB free at 09:11; no disk-space intervention is
needed at this checkpoint.

At 2026-11-15 09:25:33 JST, PID 5452 remained active at 5,000/18,879 durable
files with a matching receipt, 2,623,825,857 cumulative range bytes, and
3,413,573,630 durable output bytes. The checkpoint was written at 09:25:31,
150 files after 4,850 at 09:14:30 (11m01s). No terminal `result.json` was
present.

At 2026-11-15 09:40:01 JST, PID 5452 remained active at 5,150/18,879 durable
files with a matching receipt, 2,732,503,385 cumulative range bytes, and
3,560,987,566 durable output bytes. The checkpoint was written at 09:39:28,
150 files after 5,000 at 09:25:31 (13m57s). No terminal `result.json` was
present.

At 2026-11-15 09:53:41 JST, PID 5452 remained active at 5,300/18,879 durable
files with a matching receipt, 2,826,244,326 cumulative range bytes, and
3,687,636,602 durable output bytes. The checkpoint was written at 09:52:48,
150 files after 5,150 at 09:39:28 (13m20s). No terminal `result.json` was
present.

At 2026-11-15 10:12:03 JST, PID 5452 remained active at 5,500/18,879 durable
files with a matching receipt, 2,948,016,357 cumulative range bytes, and
3,839,586,256 durable output bytes. The checkpoint was written at 10:11:31,
200 files after 5,300 at 09:52:48 (18m43s). No terminal `result.json` was
present.

At 2026-11-15 10:28:29 JST, the same PID 5452 remained active at
5,700/18,879 durable files with a matching flushed receipt, 3,048,472,095
cumulative range bytes, and 3,974,172,664 durable output bytes. The checkpoint
was written at 10:28:14, 200 files after 5,500 at 10:11:31 (16m43s). No
terminal `result.json` was present. This is acquisition progress only; the
full media verifier has not run.

At 2026-11-15 10:36:36 JST, PID 5452 remained active at 5,800/18,879 durable
files with a matching flushed receipt, 3,095,281,865 cumulative range bytes,
and 4,038,544,900 durable output bytes. The checkpoint was written at
10:36:12, 100 files after 5,700 at 10:28:14 (7m58s). No terminal
`result.json` was present. This remains partial acquisition, not verified
media.

At 2026-11-15 10:45:26 JST, PID 5452 remained active at 5,900/18,879 durable
files with a matching flushed receipt, 3,136,642,734 cumulative range bytes,
and 4,093,635,256 durable output bytes. The checkpoint was written at
10:44:32, 100 files after 5,800 at 10:36:12 (8m20s). No terminal
`result.json` was present. This remains partial acquisition, not verified
media.

At 2026-11-15 10:53:41 JST, PID 5452 remained active at 6,000/18,879 durable
files with a matching flushed receipt, 3,184,517,767 cumulative range bytes,
and 4,159,268,510 durable output bytes. The checkpoint was written at
10:53:26, 100 files after 5,900 at 10:44:32 (8m54s). No terminal
`result.json` was present. This remains partial acquisition, not verified
media.

At 2026-11-15 11:02:04 JST, PID 5452 remained active at 6,100/18,879 durable
files with a matching flushed receipt, 3,224,404,337 cumulative range bytes,
and 4,216,572,284 durable output bytes. The checkpoint was written at
11:01:10, 100 files after 6,000 at 10:53:26 (7m44s). No terminal
`result.json` was present. This remains partial acquisition, not verified
media.

At 2026-11-15 11:10:11 JST, PID 5452 remained active at 6,200/18,879 durable
files with a matching flushed receipt, 3,269,988,466 cumulative range bytes,
and 4,285,833,404 durable output bytes. The checkpoint was written at
11:09:03, 100 files after 6,100 at 11:01:10 (7m53s). No terminal
`result.json` was present. This remains partial acquisition, not verified
media.

At 2026-11-15 11:20:43 JST, PID 5452 remained active at 6,300/18,879 durable
files with a matching flushed receipt, 3,331,150,751 cumulative range bytes,
and 4,354,605,342 durable output bytes. The checkpoint was written at
11:20:41, 100 files after 6,200 at 11:09:03 (11m38s). No terminal
`result.json` was present. This remains partial acquisition, not verified
media.

At 2026-11-15 11:31:39 JST, PID 5452 remained active at 6,400/18,879 durable
files with a matching flushed receipt, 3,394,297,759 cumulative range bytes,
and 4,436,244,092 durable output bytes. The checkpoint was written at
11:30:52, 100 files after 6,300 at 11:20:41 (10m11s). No terminal
`result.json` was present. This remains partial acquisition, not verified
media.

At 2026-11-15 11:49:58 JST, PID 5452 remained active at 6,600/18,879 durable
files with a matching flushed receipt, 3,514,292,367 cumulative range bytes,
and 4,586,941,486 durable output bytes. The checkpoint was written at
11:49:56, 200 files after 6,400 at 11:30:52 (19m04s). The RTX 3080 snapshot
was 8,270/16,384 MiB used at 53% utilization; C: had 259.62 GiB free. No
terminal `result.json` was present. This remains partial acquisition, not
verified media.

At 2026-11-15 12:00:42 JST, PID 5452 remained active at 6,700/18,879 durable
files with a matching flushed receipt, 3,575,000,630 cumulative range bytes,
and 4,659,238,090 durable output bytes. The checkpoint was written at
11:59:56, 100 files after 6,600 at 11:49:56 (10m00s). No terminal
`result.json` was present. This remains partial acquisition, not verified
media.

At 2026-11-15 12:09:32 JST, PID 5452 remained active at 6,800/18,879 durable
files with a matching flushed receipt, 3,622,142,573 cumulative range bytes,
and 4,717,869,438 durable output bytes. The progress and receipt checkpoint
were written at 12:08:43, 100 files after 6,700 at 11:59:56 (8m47s). The
latest WAV was updated at 12:09:32. No terminal `result.json` was present.
This remains partial acquisition, not verified media.

At 2026-11-15 12:16:07 JST, PID 5452 remained active at 6,900/18,879 durable
files with a matching flushed receipt, 3,655,250,382 cumulative range bytes,
and 4,761,723,128 durable output bytes. The progress and receipt checkpoint
were written together at 12:16:07, 100 files after 6,800 at 12:08:43 (7m24s).
No terminal `result.json` was present. This remains partial acquisition, not
verified media.

At 2026-11-15 12:23:00 JST, PID 5452 remained active at 7,000/18,879 durable
files with a matching flushed receipt, 3,684,681,526 cumulative range bytes,
and 4,814,074,244 durable output bytes. The progress and receipt checkpoint
were written together at 12:23:00, 100 files after 6,900 at 12:16:07 (6m53s).
No terminal `result.json` was present. This remains partial acquisition, not
verified media.

At 2026-11-15 12:28:32 JST, PID 5452 remained active at 7,100/18,879 durable
files with a matching flushed receipt, 3,699,793,753 cumulative range bytes,
and 4,837,612,846 durable output bytes. The progress and receipt checkpoint
were written together at 12:28:32, 100 files after 7,000 at 12:23:00 (5m32s).
The latest WAV was updated at 12:29:36. No terminal `result.json` was present.
This remains partial acquisition, not verified media.

At 2026-11-15 12:36:21 JST, PID 5452 remained active at 7,200/18,879 durable
files with a matching flushed receipt, 3,745,354,077 cumulative range bytes,
and 4,898,773,576 durable output bytes. The progress and receipt checkpoint
were written together at 12:36:21, 100 files after 7,100 at 12:28:32 (7m49s).
At 12:37:40, 7,220 WAV files were visible while the durable receipt remained
7,200; only receipt-confirmed rows are counted. No terminal `result.json` was
present. This remains partial acquisition, not verified media.

At 2026-11-15 12:44:34 JST, PID 5452 remained active at 7,300/18,879 durable
files with a matching flushed receipt, 3,787,295,005 cumulative range bytes,
and 4,940,984,862 durable output bytes. The progress and receipt checkpoint
were written together at 12:44:34, 100 files after 7,200 at 12:36:21 (8m13s).
No terminal `result.json` was present. This remains partial acquisition, not
verified media.

At 2026-11-15 12:52:35 JST, PID 5452 remained active at 7,400/18,879 durable
files with a matching flushed receipt, 3,834,322,121 cumulative range bytes,
and 4,993,619,558 durable output bytes. The progress and receipt checkpoint
were written together at 12:52:35, 100 files after 7,300 at 12:44:34 (8m01s).
No terminal `result.json` was present. This remains partial acquisition, not
verified media.

At 2026-11-15 13:01:47 JST, PID 5452 remained active at 7,500/18,879 durable
files with a matching flushed receipt, 3,883,343,180 cumulative range bytes,
and 5,055,465,476 durable output bytes. The progress and receipt checkpoint
were written together at 13:01:47, 100 files after 7,400 at 12:52:35 (9m12s).
No terminal `result.json` was present. This remains partial acquisition, not
verified media.

At 2026-11-15 13:10:49 JST, PID 5452 remained active at 7,600/18,879 durable
files with a matching flushed receipt, 3,940,977,045 cumulative range bytes,
and 5,131,618,152 durable output bytes. The progress and receipt checkpoint
were written together at 13:10:49, 100 files after 7,500 at 13:01:47 (9m02s).
No terminal `result.json` was present. This remains partial acquisition, not
verified media.

At 2026-11-15 13:18:02 JST, PID 5452 remained active at 7,700/18,879 durable
files with a matching flushed receipt, 3,973,108,515 cumulative range bytes,
and 5,175,485,082 durable output bytes. The progress and receipt checkpoint
were written together at 13:18:02, 100 files after 7,600 at 13:10:49 (7m13s).
No terminal `result.json` was present. This remains partial acquisition, not
verified media.

At 2026-11-15 13:26:08 JST, PID 5452 remained active at 7,800/18,879 durable
files with a matching flushed receipt, 4,018,439,340 cumulative range bytes,
and 5,233,624,790 durable output bytes. The progress and receipt checkpoint
were written together at 13:26:08, 100 files after 7,700 at 13:18:02 (8m06s).
No terminal `result.json` was present. This remains partial acquisition, not
verified media.

At 2026-11-15 13:34:54 JST, PID 5452 remained active at 7,900/18,879 durable
files with a matching flushed receipt, 4,072,127,730 cumulative range bytes,
and 5,302,676,970 durable output bytes. The progress and receipt checkpoint
were written together at 13:34:54, 100 files after 7,800 at 13:26:08 (8m46s).
No terminal `result.json` was present. This remains partial acquisition, not
verified media.

At 2026-11-15 13:46:21 JST, PID 5452 remained active at 8,000/18,879 durable
files with a matching flushed receipt, 4,137,349,658 cumulative range bytes,
and 5,377,624,880 durable output bytes. The progress and receipt checkpoint
were written together at 13:46:21, 100 files after 7,900 at 13:34:54 (11m27s).
No terminal `result.json` was present. This remains partial acquisition, not
verified media.

At 2026-11-15 13:54:28 JST, PID 5452 remained active at 8,100/18,879 durable
files with a matching flushed receipt, 4,182,637,344 cumulative range bytes,
and 5,443,471,888 durable output bytes. The progress and receipt checkpoint
were written together at 13:54:28, 100 files after 8,000 at 13:46:21 (8m07s).
No terminal `result.json` was present. This remains partial acquisition, not
verified media.

At 2026-11-15 14:01:55 JST, PID 5452 remained active at 8,200/18,879 durable
files with a matching flushed receipt, 4,215,483,776 cumulative range bytes,
and 5,493,477,616 durable output bytes. The progress and receipt checkpoint
were written together at 14:01:55, 100 files after 8,100 at 13:54:28 (7m27s).
No terminal `result.json` was present. This remains partial acquisition, not
verified media.

At 2026-11-15 14:10:10 JST, PID 5452 remained active at 8,300/18,879 durable
files with a matching flushed receipt, 4,258,564,125 cumulative range bytes,
and 5,552,011,898 durable output bytes. The progress and receipt checkpoint
were written together at 14:10:10, 100 files after 8,200 at 14:01:55 (8m15s).
No terminal `result.json` was present. This remains partial acquisition, not
verified media.

## Next gates

1. Continue monitoring attempt-06's live PID/session; never infer completion
   from an observation timeout.
2. On terminal PASS, run the full receipt/media verifier and exact cross-split
   content checks.
3. Extract the pinned EfficientAT features over the entire clip, including all
   39 duration outliers; abort on decoder, memory, non-finite, or repeatability
   failures.
4. Fit the predeclared 25,089-parameter typed readout on train only, fit one
   scalar temperature on the uploader-held calibration split, and select by
   macro validation NLL. Preserve seed, learning history, predictions, and
   per-label Accuracy/NLL/Brier/ECE.

This remains one audio annotation-prediction component. It does not close
image, text, video, multimodal necessity, package-size, or release-quality
gates.

At 2026-11-15 14:19:06 JST, PID 5452 remained active and attempt-06 reached
8,400/18,879 durable files with a matching flushed receipt, 4,306,707,124
cumulative range bytes, and 5,616,577,008 durable output bytes. This was 100
files after the 8,300 checkpoint at 14:10:10 (8m56s). 8,409 WAV files were
visible at the following observation, but the durable count remains 8,400. No
terminal `result.json` exists. Acquisition is still partial and unverified.

At 2026-11-15 14:32:03 JST, PID 5452 remained active and attempt-06 reached
8,500/18,879 durable files with matching progress/receipt state, 4,401,810,340
cumulative range bytes, and 5,733,632,434 durable output bytes. This was 100
files after the 8,400 checkpoint at 14:19:06 (7m12s). No terminal
`result.json` exists; the media remains partial and unverified.

At 2026-11-15 14:35:49 JST, PID 5452 remained active and attempt-06 reached
8,550/18,879 durable files with matching progress/receipt state, 4,419,472,601
cumulative range bytes, and 5,756,770,192 durable output bytes. This was 50
files after the 8,500 checkpoint at 14:32:03 (3m46s). No terminal
`result.json` exists; the media remains partial and unverified.

At 2026-11-15 14:40:19 JST, PID 5452 remained active and attempt-06 reached
8,600/18,879 durable files with matching progress/receipt state, 4,444,874,910
cumulative range bytes, and 5,795,898,768 durable output bytes. This was 50
files after the 8,550 checkpoint at 14:35:49 (4m30s). No terminal
`result.json` exists; the media remains partial and unverified.

At 2026-11-15 14:45:51 JST, PID 5452 remained active and attempt-06 reached
8,650/18,879 durable files with matching progress/receipt state, 4,478,861,790
cumulative range bytes, and 5,839,173,226 durable output bytes. This was 50
files after the 8,600 checkpoint at 14:40:19 (5m32s). No terminal
`result.json` exists; the media remains partial and unverified.

At 2026-11-15 14:51:42 JST, PID 5452 remained active and attempt-06 reached
8,700/18,879 durable files with matching progress/receipt state, 4,513,634,914
cumulative range bytes, and 5,877,856,480 durable output bytes. This was 50
files after the 8,650 checkpoint at 14:45:51 (5m51s). No terminal
`result.json` exists; the media remains partial and unverified.

At 2026-11-15 14:57:21 JST, PID 5452 remained active and attempt-06 reached
8,750/18,879 durable files with matching progress/receipt state, 4,555,156,750
cumulative range bytes, and 5,938,176,212 durable output bytes. This was 50
files after the 8,700 checkpoint at 14:51:42 (5m39s). No terminal
`result.json` exists; the media remains partial and unverified.

At 2026-11-15 15:01:11 JST, PID 5452 remained active and attempt-06 reached
8,800/18,879 durable files with matching progress/receipt state, 4,573,863,735
cumulative range bytes, and 5,966,894,436 durable output bytes. This was 50
files after the 8,750 checkpoint at 14:57:21 (3m50s). No terminal
`result.json` exists; the media remains partial and unverified.

At 2026-11-15 15:06:30 JST, PID 5452 remained active and attempt-06 reached
8,850/18,879 durable files with matching progress/receipt state, 4,602,561,541
cumulative range bytes, and 6,002,910,566 durable output bytes. This was 50
files after the 8,800 checkpoint at 15:01:11 (5m19s). No terminal
`result.json` exists; the media remains partial and unverified.

At 2026-11-15 15:11:01 JST, PID 5452 remained active and attempt-06 reached
8,900/18,879 durable files with matching progress/receipt state, 4,631,755,346
cumulative range bytes, and 6,038,542,420 durable output bytes. This was 50
files after the 8,850 checkpoint at 15:06:30 (4m31s). No terminal
`result.json` exists; the media remains partial and unverified.

At 2026-11-15 15:15:55 JST, PID 5452 remained active and attempt-06 reached
8,950/18,879 durable files with matching progress/receipt state, 4,661,386,782
cumulative range bytes, and 6,073,450,472 durable output bytes. This was 100
files after the 8,850 checkpoint at 15:06:30 (9m25s). No terminal
`result.json` exists; the media remains partial and unverified.

At 2026-11-15 15:18:49 JST, PID 5452 remained active and attempt-06 reached
9,000/18,879 durable files with matching progress/receipt state, 4,671,074,470
cumulative range bytes, and 6,090,414,504 durable output bytes. This was 50
files after the 8,950 checkpoint at 15:15:55 (2m54s). No terminal
`result.json` exists; the media remains partial and unverified.

At 2026-11-15 15:22:12 JST, PID 5452 remained active and attempt-06 reached
9,050/18,879 durable files with matching progress/receipt state, 4,685,867,521
cumulative range bytes, and 6,112,359,650 durable output bytes. This was 50
files after the 9,000 checkpoint at 15:18:49 (3m23s). No terminal
`result.json` exists; the media remains partial and unverified.

At 2026-11-15 15:25:43 JST, PID 5452 remained active and attempt-06 reached
9,100/18,879 durable files with matching progress/receipt state, 4,697,136,845
cumulative range bytes, and 6,129,162,244 durable output bytes. This was 50
files after the 9,050 checkpoint at 15:22:12 (3m31s). No terminal
`result.json` exists; the media remains partial and unverified.
