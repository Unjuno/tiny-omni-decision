# R0 artifact cleanup receipt — 2026-11-03

## Result

Removed seven redundant partial video payloads from
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\media\train`.
Their combined size was **541,065,216 bytes (516 MiB)**. C: free space rose
from **6.17 GiB** to **6.67 GiB** after cleanup.

For every removed file, the cleanup script checked that:

1. a complete same-named training video existed under
   `holoassist-textfree-unused-train-recovery-02\media\train`;
2. the complete file's SHA-256 matched `media_sha256` in the frozen
   `temporal-windows-v2\train.jsonl` receipt;
3. every byte of the `.part` file equaled the corresponding prefix of that
   complete video; and
4. the cleanup receipt had recorded both paths, sizes, full/partial SHA-256
   values, expected digest, and the exact-prefix result before deletion.

The machine-readable per-file record is
[`r0-partial-duplicate-cleanup-20261103.json`](../../manifests/pretrained_reuse/r0-partial-duplicate-cleanup-20261103.json).
It records status `COMPLETE`, all seven removed paths, and a total of
541,065,216 bytes.

## Preserved files and live work

The complete canonical videos and the recovery-02 temporal-window corpus were
retained. Acquisition logs and manifests were retained. Nine other HoloAssist
partial videos had same-named complete files, but their bytes did **not** match
the complete files' prefixes; they remain untouched. The eight GRID archive
partials, the GRID validation archive partial, and `clap-full-clip-features-01`
`progress.tmp` also remain because no verified canonical replacement was
established. There were 19 `.part`/`.tmp`-suffix files remaining after cleanup.

The active RoboMNIST CPU feature extractor (PID 5596) was not interrupted or
modified. A post-cleanup check showed 303/1,780 groups, matching group/receipt
counts, and 6.67 GiB free. No GPU work, sealed audit, checkpoint, model, or
corpus was touched by this cleanup.

This cleanup does not claim that all remaining temporary files are removable.
Further cleanup requires the same per-file canonical-copy, hash, and recoverable
provenance checks.
