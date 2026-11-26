# R0 exact-duplicate partial cleanup — 2026-11-03

Two stale partial downloads were checked against complete copies in separate
experiment directories. Each partial matched the complete file byte-for-byte
from offset zero through the end of the partial. Each complete copy's SHA-256
and byte length matched its acquisition receipt. The per-file evidence is in
[`r0-exact-duplicate-cleanup-20261103-pass2.json`](../../manifests/pretrained_reuse/r0-exact-duplicate-cleanup-20261103-pass2.json).

| Partial | Size | Verified complete copy | Exact prefix | Receipt |
|---|---:|---|---|---|
| HoloAssist `R115-10Aug-Navvis-Part2.mp4.part` | 62,914,560 bytes | `holoassist-video-pilot-160-207-seed17-attempt-04` | yes | SHA-256 and size match `download-progress.jsonl` |
| GRID `s4.zip.part` | 218,996,736 bytes | `grid-video-archives-duplicate-audit-20261028-attempt-01/s4.zip` | yes | SHA-256, size, and publisher MD5 match `acquisition-result.json` |

The two partials contain no bytes absent from their complete canonical copies.
All canonical media and provenance were retained. No other `.part` or `.tmp`
file was removed. At review time there was no Python, FFmpeg, or download
process; an unrelated user-owned Quartus build was active and was not touched.

This cleanup addresses only duplicate payload bytes. It does not establish
dataset admission or improve model quality. The RoboMNIST feature cache remains
an independent incomplete run and must be rechecked after the available disk
reserve is restored.

**Removal state:** not completed. The guarded PowerShell removal was rejected
by automatic action review before execution. Both `.part` files remain present;
no alternate deletion route will be attempted. Bytes freed: **0**. The exact
duplicate finding remains valid, but it did not change available capacity.
