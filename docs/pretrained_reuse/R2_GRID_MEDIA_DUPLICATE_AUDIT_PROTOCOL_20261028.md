# GRID exact media duplicate audit (2026-10-28)

Status: **audio exact-payload scan passed; video archive acquisition and the
full audio/video media audit remain in progress. No new training result exists.**
This is a prerequisite integrity check for any future frozen-feature Path A
fit on the corrected direct-source split.

## Frozen inputs and scope

- Corrected split summary:
  `C:/CodexArtifacts/pretrained-reuse-r2/grid-development-split-direct-source-20261028-attempt-02/summary.json`,
  SHA-256 `547e84167267428712ba5c7859869cd1eb7d558ee3502d9908f89223a77a3410`.
- Source bundle index SHA-256:
  `fb54bb49a0f9425cd6206508e99a1d2fef25220444e3853af11aac75431861ed`.
- Eligible split manifests: train 23,993 rows, validation 7,998 rows.
- The source gate already established separate speaker/source identities,
  zero code and normalized-target overlap, and zero known-observed-code
  overlap. Physical cross-corpus speaker independence is still unknown.
- GRID remains internal research only: the pinned Zenodo record says CC BY 4.0;
  Sheffield describes research use. Product distribution is not cleared.

The new audit hashes every `.wav` and `.mpg` member payload in all 32 official
per-speaker archives. It compares eligible train/validation rows by exact
SHA-256 within each modality. It does not decode or perceptually normalize
re-encoded media. A clean result therefore rules out exact payload duplicates,
not perceptual near-duplicates. Archive members are streamed in memory and
never extracted to a second corpus tree. All original and acquired archives
remain preserved outside Git.

## Acquisition measurement and active run

Total video ZIP bytes in the pinned summary are 13,160,236,536 (12.26 GiB).
The existing `s30.zip` is reused read-only after rechecking its exact size,
publisher MD5 `8958f25b49de7a7827a76abbdc7fedb2` and SHA-256
`69e7c2ef14b1bd4f2d0012a8a13fe8e9375c4f11a03c4de9f03f4071f48f12fd`.
The remaining 31 archives are being written under the new external attempt
directory:
`C:/CodexArtifacts/pretrained-reuse-r2/grid-video-archives-duplicate-audit-20261028-attempt-01/`.

Throughput probes used validated HTTP 206 ranges against the official `s10.zip`
file and saved their receipts in the attempt directory:

| Method | Samples | Mean / aggregate speed |
|---|---:|---:|
| API URL, sequential 4 MiB ranges | 3 | 0.255 MiB/s mean; 0.253 MiB/s aggregate |
| Record URL, sequential 4 MiB ranges | 3 | 0.270 MiB/s mean |
| API URL, four concurrent 8 MiB ranges | 4 | 1.177 MiB/s aggregate |

The full-file run uses four workers, checks every completed archive against the
publisher MD5, records SHA-256, and preserves partials for hash-gated Range
resume. The first four workers had a roughly 160-second start delay; after
startup, the last 12 per-connection intervals averaged 0.353 MiB/s, or an
estimated 1.411 MiB/s aggregate. That implied about 2.4 hours for the remaining
11.9 GiB at the measurement snapshot. This is a forecast, not a completion
claim; the live progress log and actual elapsed time are authoritative.

At 2026-10-28 11:33 JST, the first four archives (s10–s13; 1,608,089,900
bytes) completed with publisher MD5 and SHA-256 checks passing in 18 minutes
21 seconds from run start. Their measured aggregate rate was about 1.39
MiB/s. The remaining 27 downloads total about 10.4 GiB; at that observed
whole-run rate the forecast is approximately 2 hours 8 minutes. The s14 and
s15 transfers had started, and the overall acquisition was still live. Do not
interpret this interim rate as a guaranteed ETA.

At 2026-10-28 11:36 JST, the next four connections (s14–s17) had produced 11
post-startup interval samples averaging 0.324 MiB/s per connection, or 1.294
MiB/s across the four workers. The remaining 10.4 GiB forecast was then 2
hours 17 minutes. The interval set includes three recent intervals each for
s14–s16 and two for s17; sampling and transfer continue.

At 2026-10-28 11:52 JST, seven archives had passed verification. A fresh
three-sample 30-second measurement (0.800, 0.800 and 1.067 MiB/s aggregate)
averaged 0.889 MiB/s. This calculation includes completed archive sizes plus
current partial bytes so completion of an archive does not look like negative
throughput. About 9.19 GiB remained, implying approximately 176 minutes at
that short-window mean. This is a variable network estimate; the acquisition
process remained active and no restart was attempted.

At 2026-10-28 11:56 JST, eight archives had passed verification. The next
three 30-second cumulative-size samples were 1.867, 1.067 and 0.800 MiB/s,
averaging 1.244 MiB/s. About 8.91 GiB remained, corresponding to roughly 122
minutes at that short-window mean. The downloader remained live; no restart
or worker-count change was made.

At 2026-10-28 11:59 JST, the cumulative-size samples were 1.333, 1.867 and
1.867 MiB/s, averaging 1.689 MiB/s. Eight archives were verified and about
8.62 GiB remained, for an approximately 87-minute forecast at this window's
mean. The download remained active; forecast changes reflect variable
throughput and are not a completion claim.

At 2026-10-28 12:02 JST, the next three cumulative-size samples were all
1.867 MiB/s. Eight archives were verified, about 3.95 GiB had been downloaded
including partials, and 8.31 GiB remained. The short-window forecast was
approximately 76 minutes. PID 28572 and the progress log were both confirmed
live at this observation; no GPU or model training was used.

At 2026-10-28 12:05 JST, ten archives were verified, with about 4.28 GiB
downloaded and 7.98 GiB remaining. Three 30-second cumulative-size samples
(1.867, 1.600 and 1.522 MiB/s) averaged 1.663 MiB/s, for an approximately
82-minute forecast. The progress log was updated at 12:05:19 JST.

At 2026-10-28 12:08 JST, eleven archives were verified. The next three
cumulative-size samples were 1.067, 1.333 and 1.345 MiB/s (mean 1.248 MiB/s).
About 7.78 GiB remained, yielding an approximately 106-minute forecast. The
progress log was updated at 12:08:18 JST; the estimate varies with network
throughput.

At 2026-10-28 12:15 JST, the three cumulative-size samples were 1.333, 1.600
and 2.133 MiB/s, averaging 1.689 MiB/s. Twelve archives were verified, about
4.96 GiB had been acquired, and 7.29 GiB remained; the short-window ETA was
about 74 minutes. The progress log was updated at 12:15:02 JST.

At 2026-10-28 12:17 JST, the next three cumulative-size samples were 2.400,
1.867 and 1.333 MiB/s, averaging 1.867 MiB/s. Twelve archives were verified,
about 5.28 GiB had been acquired, and 6.98 GiB remained; the short-window
forecast was approximately 64 minutes. The progress log was updated at
12:17:50 JST.

At 2026-10-28 12:24 JST, 13 archives were verified and cumulative acquisition
was about 5.98 GiB, leaving 6.28 GiB. Using the 1.771 MiB/s mean from the
12:21 three-sample measurement gives an approximately 61-minute forecast.
The progress log was updated at 12:24:24 JST.

At 2026-10-28 12:27 JST, 16 archives had passed publisher checks; about 6.20
GiB was acquired and 6.05 GiB remained. At the 12:21 measured mean of 1.771
MiB/s, the forecast was about 58 minutes. Progress was updated at 12:27:35.

At 2026-10-28 12:30 JST, 16 archives remained verified, cumulative acquisition
was about 6.48 GiB, and 5.77 GiB remained. The next three 30-second samples
were 2.133, 2.133 and 2.400 MiB/s (mean 2.222 MiB/s), giving a short-window
forecast of about 44 minutes. The progress log was updated at 12:30:37 JST.

At 2026-10-28 12:37 JST, 17 archives were verified; cumulative acquisition
was 7.36 GiB with 4.90 GiB remaining. The 12:36 three-sample mean of 2.200
MiB/s implied about 38 minutes remaining. The progress log was updated at
12:37:54 JST.

At 2026-10-28 12:41 JST, 19 archives were verified; cumulative acquisition
was 7.70 GiB with 4.56 GiB remaining. Using the 12:36 measured mean of 2.200
MiB/s gives about 35 minutes remaining. The progress log was updated at
12:41:35 JST.

At 2026-10-28 12:46 JST, 20 archives were verified; cumulative acquisition
was 8.15 GiB with 4.11 GiB remaining. Three 30-second cumulative-size
samples (2.400, 1.600 and 2.133 MiB/s) averaged 2.044 MiB/s, for a forecast
of about 34 minutes. The progress log was updated at 12:46:35 JST.

At 2026-10-28 12:57 JST, 23 archives were verified; cumulative acquisition
was 9.19 GiB and 3.07 GiB remained. At the latest 1.687 MiB/s three-sample
mean, the forecast was about 31 minutes. The progress log was updated at
12:57:06 JST.

At 2026-10-28 13:00 JST, 24 archives were verified; cumulative acquisition
was 9.44 GiB with 2.82 GiB remaining. At the 1.687 MiB/s measured mean, the
forecast was about 29 minutes. The progress log was updated at 13:00:06 JST.

## Audio-only interim result

Before the video archives finish, all 32 pinned audio TARs were re-hash
verified and all 32,000 WAV members were streamed and hashed. The 31,991
eligible train/validation rows had **zero exact-payload duplicate groups
between splits and zero within either split**. This clears only the exact WAV
payload gate; it does not rule out re-encoded audio duplicates or establish
cross-corpus speaker independence. No audio was decoded for quality, no model
was loaded, and no sealed audit was accessed.

- Report:
  `C:/CodexArtifacts/pretrained-reuse-r2/grid-audio-duplicate-audit-20261028-attempt-01/report.json`.
- Report SHA-256:
  `ce95ee4679dff2a18dc855f44aa031f0e3a43bd65587ed5e500f048cff330f71`.
- Runtime: 11.437 seconds.
- The report pins the split summary, bundle index, train/validation manifests,
  and per-speaker audio TAR SHA-256 values.

## Real ZIP member-hash smoke

The first real-archive smoke found 2,000 `.mpg`-suffixed entries in `s10.zip`:
1,000 actual clips plus 1,000 `__MACOSX/._*.mpg` AppleDouble metadata
sidecars. The pinned source-inventory parser already excludes those sidecars;
the media scanner was corrected to match it, and a unit test now covers the
case. The pre-fix attempt is preserved under
`grid-video-member-hash-smoke-20261028-attempt-01`.

The corrected streaming scan passed on all 1,000 actual s10 clips, with 1,000
distinct exact payload hashes. The ZIP's publisher MD5 and SHA-256 matched the
acquisition receipt. It took 3.827 seconds and wrote the per-code hashes to
`C:/CodexArtifacts/pretrained-reuse-r2/grid-video-member-hash-smoke-20261028-attempt-02/result.json`
(SHA-256 `773eca17c17e1456e8c3c6a10bfa50b149bbc6024ff797164c20ab65350710c8`).
The runner snapshot SHA-256 is
`5e081559be4cce03a21885b3c14d55a59dd8c185069e509e93d485ef20bb536a`.
This validates one train speaker's ZIP structure, not the full split-level
video duplicate gate.

## Partial video split diagnostic

At 2026-10-28 12:11 JST, all 12 then-completed speaker archives were rechecked
against the pinned publisher MD5 and byte size, then streamed without
extraction. The 12,000 actual MPG payloads had 12,000 unique SHA-256 hashes,
with zero exact cross-split duplicate groups and zero within-split duplicate
groups in this subset. The audit covers speakers s2, s10–s20 (excluding s21)
and includes both train and validation speakers. The report also records each
archive SHA-256 and source/config hashes:

- Report: `C:/CodexArtifacts/pretrained-reuse-r2/grid-video-partial-media-audit-20261028-attempt-01/report.json`.
- Report SHA-256: `97cdc86c116a4a6bbcf1f80b85cd6eabe78dbd8952bed210691e63c7f81d16a4`.
- Runtime: 36.367 seconds.

This is a partial diagnostic only. It says nothing about duplicates involving
the remaining 20 speaker archives; the required full 32-speaker audit remains
pending and GRID training remains gated on that full result.

An incremental attempt at 2026-10-28 12:22 JST added newly verified `s22.zip`
and rechecked the 13 completed train/validation archives. All 13,000 actual
MPG payloads again had unique hashes, with zero exact cross-split or within-
split duplicate groups. The 20 uncompleted speakers remain outside this
diagnostic, so the full gate remains pending.

- Report: `C:/CodexArtifacts/pretrained-reuse-r2/grid-video-partial-media-audit-20261028-attempt-02/report.json`.
- Report SHA-256: `b7e5f329ee8e7c31e2655696c5a5e245b2de7fd2f744edf38b0beba31a97b6ca`.
- Runtime: 41.293 seconds.

At 2026-10-28 12:27 JST, a further incremental scan covered 16 publisher-
verified archives: s2, s10–s20, and s22–s25. All 16,000 actual MPG payloads
had unique exact SHA-256 hashes; there were zero exact cross-split and
within-split duplicate groups. This remains partial coverage and does not
clear the full 32-speaker gate.

- Report: `C:/CodexArtifacts/pretrained-reuse-r2/grid-video-partial-media-audit-20261028-attempt-03/report.json`.
- Report SHA-256: `5890ac1b9f002ae43fb551beb86ccca0cecfc1895722638d956b000a6e5f71a4`.
- Runtime: 50.410 seconds.

At 2026-10-28 12:37 JST, an incremental scan of 17 publisher-verified
archives (s2, s10–s20, and s22–s26) hashed 17,000 actual MPG payloads. All
17,000 hashes were unique; there were zero exact cross-split and within-split
duplicate groups. The remaining 15 speakers still require the full audit.

- Report: `C:/CodexArtifacts/pretrained-reuse-r2/grid-video-partial-media-audit-20261028-attempt-04/report.json`.
- Report SHA-256: `fa9cd77f8ea5c192046bb64e3ddb8f63583ab30513042481d77c3c0146f976fe`.
- Runtime: 52.701 seconds.

At 2026-10-28 12:41 JST, the next incremental diagnostic covered 19
publisher-verified archives (s2, s10–s20, and s22–s26 plus s28–s29). It
hashed 19,000 actual MPG payloads with 19,000 unique hashes and zero exact
cross-split or within-split duplicate groups. The remaining 13 speakers still
require the complete split audit.

- Report: `C:/CodexArtifacts/pretrained-reuse-r2/grid-video-partial-media-audit-20261028-attempt-06/report.json`.
- Report SHA-256: `2284b4592431f51114313e9a7675e76f0a034756d21a257d45badb950cb33094`.
- Runtime: 63.124 seconds.

At 2026-10-28 12:45 JST, the next incremental diagnostic covered 20
publisher-verified archives (s2, s10–s20, and s22–s29). It hashed 20,000
actual MPG payloads; all hashes were unique, with zero exact cross-split and
within-split duplicate groups. Twelve speakers remain outside this partial
screen and the full audit is still required.

- Report: `C:/CodexArtifacts/pretrained-reuse-r2/grid-video-partial-media-audit-20261028-attempt-07/report.json`.
- Report SHA-256: `6b7307b75ed5f0964d72e883c454f4dc9ed86597e22acdcdddb7f9e302693a28`.
- Runtime: 64.494 seconds.

At 2026-10-28 12:57 JST, the next scan covered 23 publisher-verified archives
(s2, s10–s20, and s22–s29 plus s31–s32). It hashed 23,000 actual MPG payloads;
all were unique, with zero exact cross-split or within-split duplicate groups.
Nine speakers remain outside the partial scan.

- Report: `C:/CodexArtifacts/pretrained-reuse-r2/grid-video-partial-media-audit-20261028-attempt-10/report.json`.
- Report SHA-256: `36fa28393c20387374c69b906408e4eaf1296fe6b8e1352e9012f58f5ab44a01`.
- Runtime: 75.087 seconds.

At 2026-10-28 13:00 JST, the next diagnostic covered 24 publisher-verified
archives (s2, s10–s20, and s22–s29 plus s31–s33). It hashed 24,000 actual MPG
payloads; every hash was unique, with zero exact cross-split or within-split
duplicates. Eight speakers remain unscanned.

- Report: `C:/CodexArtifacts/pretrained-reuse-r2/grid-video-partial-media-audit-20261028-attempt-11/report.json`.
- Report SHA-256: `1b3e2858980cc8472a8949ab21a7630ccabb5f3d4921eefc2d86479db53171d1`.
- Runtime: 73.749 seconds.

Executed command:

```powershell
python -m scripts.audit_grid_media_duplicates `
  --summary C:\CodexArtifacts\pretrained-reuse-r2\grid-development-split-direct-source-20261028-attempt-02\summary.json `
  --expected-summary-sha256 547e84167267428712ba5c7859869cd1eb7d558ee3502d9908f89223a77a3410 `
  --bundle-index C:\CodexArtifacts\pretrained-reuse-r2\grid-train-direct-speaker-audit-20261028-attempt-03\direct-source-bundle-index.json `
  --train C:\CodexArtifacts\pretrained-reuse-r2\grid-development-split-direct-source-20261028-attempt-02\train.json `
  --validation C:\CodexArtifacts\pretrained-reuse-r2\grid-development-split-direct-source-20261028-attempt-02\validation.json `
  --output C:\CodexArtifacts\pretrained-reuse-r2\grid-audio-duplicate-audit-20261028-attempt-01\report.json `
  --audio-only
```

The exact acquisition command was:

```powershell
python scripts/acquire_grid_video_archives.py `
  --summary C:\CodexArtifacts\pretrained-reuse-r2\grid-development-split-direct-source-20261028-attempt-02\summary.json `
  --expected-summary-sha256 547e84167267428712ba5c7859869cd1eb7d558ee3502d9908f89223a77a3410 `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\grid-video-archives-duplicate-audit-20261028-attempt-01 `
  --existing-archive s30=C:\CodexArtifacts\pretrained-reuse-r2\grid-validation-video-20261028-attempt-01\s30.zip `
  --workers 4
```

At the initial run snapshot, C: had 27.19 GiB free; the planned additional
archive storage is about 11.90 GiB, leaving more than the runner's 2 GiB
reserve. No GPU, cloud compute, model fitting, sealed audit, or source/worktree
mutation was used.

## Audit command after acquisition succeeds

```powershell
python scripts/audit_grid_media_duplicates.py `
  --summary C:\CodexArtifacts\pretrained-reuse-r2\grid-development-split-direct-source-20261028-attempt-02\summary.json `
  --expected-summary-sha256 547e84167267428712ba5c7859869cd1eb7d558ee3502d9908f89223a77a3410 `
  --bundle-index C:\CodexArtifacts\pretrained-reuse-r2\grid-train-direct-speaker-audit-20261028-attempt-03\direct-source-bundle-index.json `
  --archive-result C:\CodexArtifacts\pretrained-reuse-r2\grid-video-archives-duplicate-audit-20261028-attempt-01\acquisition-result.json `
  --train C:\CodexArtifacts\pretrained-reuse-r2\grid-development-split-direct-source-20261028-attempt-02\train.json `
  --validation C:\CodexArtifacts\pretrained-reuse-r2\grid-development-split-direct-source-20261028-attempt-02\validation.json `
  --output C:\CodexArtifacts\pretrained-reuse-r2\grid-media-duplicate-audit-20261028-attempt-01\report.json
```

The scan must verify all archive and source hashes, row/speaker assignment,
archive member counts, and train/validation disjointness before comparing
payload hashes. Do not start GRID training until the full scan completes and
the source-media integrity result is reviewed. Even a clean exact-byte scan
does not establish person identity across GRID and other training corpora or
clear product distribution rights.
