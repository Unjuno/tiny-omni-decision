# GRID exact media duplicate audit (2026-10-28)

Status: **acquisition is in progress; no duplicate-scan result or new training
result exists yet.** This is a prerequisite integrity check for any future
frozen-feature Path A fit on the corrected direct-source split.

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
