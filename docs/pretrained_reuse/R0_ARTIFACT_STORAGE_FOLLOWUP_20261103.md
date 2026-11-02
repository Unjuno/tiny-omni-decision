# R0 artifact storage follow-up — 2026-11-03

## Current snapshot

- Worktree: `C:\CodexWorktrees\r2-perception-control-freezer-20261031`.
- Branch/source commit: `docs/pretrained-reuse-roadmap-20261010` at
  `b0c020bb3f269ad299ccd1945cbd4870a0251a73`.
- C: free space: **6.48 GiB** (PowerShell volume accounting reported
  946.18 GiB used).
- `C:\CodexArtifacts\pretrained-reuse-r2` remains approximately **683.42 GiB
  logical size** by the previous full inventory; logical totals do not equal
  physical allocation where NTFS hard links exist.
- No Python, FFmpeg, curl/wget, or uv process was present after the CPU smoke.
- No GPU was initialized or used. No media, corpus, checkpoint, or existing run
  was deleted or rewritten.

## Large paths and disposition

The following are the largest currently inventoried R2 paths. Their names and
logical sizes do not prove that their contents are disposable.

| Path | Logical size | Disposition |
| --- | ---: | --- |
| `holoassist-textfree-unused-train-recovery-02` | 112.16 GiB | Retain: source media/recovery provenance; reuse is plausible. |
| `holoassist-textfree-unused-train-attempt-01` | 72.80 GiB | Retain pending source/manifest reconciliation. |
| `clevr4-100k-videoprism-v0` | 67.27 GiB | Retain: source archive and feature inputs used in prior work. |
| `greatest-hits-video-cache-01` | 65.61 GiB | Retain pending task and source-status reconciliation. |
| `holoassist-video-pilot-160-207-seed17-pitchshifted-attempt-01` | 33.76 GiB | Retain: experiment reproduction data. |
| `droid-expanded-site-holdout-data-attempt-01` | 17.34 GiB | Retain pending canonical-shard and provenance reconciliation. |
| `droid-expansion-sample-20261028-attempt-01` | 14.65 GiB | Retain: aliases are already hard-linked to canonical DROID shards. |
| `perception-test-av-v1` | 14.50 GiB | Retain: fixed evaluation inputs and provenance. |
| `perception-av-event-features-02` | 9.50 GiB | Retain pending feature-manifest and consumer audit. |

The two CLEVR4 `train-spatial-features.npy` paths are one NTFS hard-linked file
(2,949,120,128 bytes); the DROID sample paths similarly alias canonical shards.
Removing an alias would not recover their underlying storage. The earlier
inventory also found 26 `.part`/temporary files (1.765 GiB), but did not prove
whether they can resume or be regenerated. They remain untouched.

### Live recheck later on 2026-11-03

The active feature extraction process remains live as PID `5596` (started
07:03:44 local); the run metadata says `RUNNING`. At the latest snapshot it had
committed 78 / 1,780 groups, transferred 603.7 MiB, used about 31.4 minutes,
left **6.19 GiB** free on C:, and used 0 MiB / 0% GPU. The committed feature
groups occupied about 2.3 MiB. This is consistent with the measured low-storage
streaming path; the active process and output were left untouched.

The 26 partial/temp files were re-inventoried by path and size. They comprise
one 0.30 MiB CLAP progress file, 15 HoloAssist video `.part` files (about
928 MiB), and 10 GRID archive `.part` files (about 837 MiB). No matching final
file exists beside any of these partials, so they may be the only local resume
state. No active command line referenced their paths, but that alone does not
prove they are not resumable. They remain protected.

Four old hidden CLEVRER media temp directories contain 293 files totaling
333.62 MiB. SHA-256 comparison found 32 exact-duplicate hash groups, with a
maximum of 40.93 MiB of duplicate payload beyond one copy per hash; this is an
upper-bound storage opportunity, not proof that any directory is disposable.
The media files remain untouched. Hard-link conversion or deletion was not
attempted.

## Working disposition rules

- **Keep:** source datasets, canonical media, selected checkpoints, fixed
  evaluation inputs, and any cache referenced by a report, manifest, or run.
- **Recomputable but not automatically disposable:** frozen feature caches.
  Require source/checkpoint/code pins and a measured regeneration path before
  considering replacement.
- **Hold for per-source review:** partial downloads, failed attempt metadata,
  and old exploratory media. Age or a name containing `unused`, `failed`, or
  `tmp` is insufficient evidence of safe removal.
- **No deletion performed:** the prior automated deletion review for the
  apparent CLEVR4 duplicate was rejected. This follow-up did not retry deletion
  through another route. No other item has yet passed the required receipt,
  canonical-copy, active-reader, and recovery checks.

## Lower-footprint experiment path measured

The frozen RoboMNIST AV-match attempt-03 split selects 1,780 whole-repetition
groups (780 train, 1,000 validation) and 1,560 / 2,000 binary Decision examples.
Its exact ZIP member compressed-size estimate for the selected audio/video
members is **14,631,647,490 bytes** (13.63 GiB), so persisting all source media
locally is not compatible with current free space.

A CPU-only smoke extracted fixed features from two training groups using pinned
VideoPrism-B and EfficientAT weights. It streamed one uncached group's two ZIP
members by HTTP Range, verified their archive CRCs and uncompressed sizes, and
did not save raw media. Results:

- elapsed: **48.13 s** for two groups, including model load and 8,509,914
  network bytes;
- peak process working set: **4,712,833,024 bytes**;
- features: VideoPrism `[8,768]` and EfficientAT `[8,384]` per group;
- serialized two-group feature file: **68,863 bytes**;
- zero trainable parameters; no quality metrics, fit, GPU, or sealed-audit use.

Linear size estimates for the selected 1,780 groups are 65,617,920 bytes
(62.55 MiB) of float32 feature tensors and about 61.2 MB for the compressed
container if the two-group compression ratio holds. This is a storage estimate,
not a performance estimate. Network throughput is not measured separately by
this smoke; older measurements range from 0.134 to 0.492 MiB/s, so retrieving
all selected media could take many hours. Any expansion should stream members,
write only pinned feature outputs, log resumable per-group hashes, and stop if
the drive approaches the established 2 GiB reserve.

## Conclusion

Capacity pressure is real, but this audit has not established a safe deletion
candidate. The next experiment can avoid retaining the 13.63 GiB source-media
set by streaming and retaining only a small frozen-feature artifact. Revisit
cleanup only after a receipt-level audit proves that a specific path is
redundant or deterministically recoverable and is not an input to an active or
reproducibility-critical run.
