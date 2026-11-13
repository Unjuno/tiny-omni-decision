# R0 artifact cleanup recheck — 2026-11-03

## Snapshot

- Worktree: `C:\CodexWorktrees\r2-perception-control-freezer-20261031`.
- Branch: `docs/pretrained-reuse-roadmap-20261010`, source head before this
  record: `4882b173`.
- C: free space at inspection: **7.39 GiB**.
- No Python, FFmpeg, downloader, or curl/wget process was present in the
  read-only process query.
- The user's two untracked PhysicalAI split manifests were left untouched.

## Candidate check

The old staging directory
`C:\CodexArtifacts\pretrained-reuse-r2\.clevrer-temporal-mix-media-v1.tmp-5hepno0z`
contains 161 files totaling **193,513,037 bytes**. Its intended-name canonical
directory `clevrer-temporal-mix-media-v1` contains 1,418 files, but none of the
staging files has a matching relative path there. This is not proof of bytewise
duplication elsewhere or deterministic recovery, so the staging files were
retained. No deletion was attempted.

Three other old CLEVRER `.tmp` directories remain part of the prior inventory.
Their combined size with the checked staging directory is about 0.326 GiB.
No receipt-level proof establishes that their incomplete media can be recovered
from a verified canonical copy, so they remain retained as well. The separate
26 `.part`/temporary files totaling 1.765 GiB remain pending source-receipt and
resume-capability review; this follow-up did not delete or modify them.

The broader inventory remains as recorded in
[`R0_ARTIFACT_STORAGE_AUDIT_20261103.md`](R0_ARTIFACT_STORAGE_AUDIT_20261103.md):
the 683.42 GiB logical-size store is predominantly data, cached media/features,
and experiment outputs. Age and naming alone are insufficient grounds to
remove these artifacts. No model, media, corpus, checkpoint, run, or cache was
deleted or rewritten in this recheck.

## R2 source gate cross-check

The pinned PhysicalAI `panda-stack-platforms` metadata and pilot reports still
expose episode IDs but no scene/layout/environment key. Existing pilot features
and validation were already used and are not reused here; no reserved media,
labels, validation outcomes, or new features were accessed. Without an
independent scene key or a new admitted source, there is no defensible next
quality run from this candidate. Keep it `REVIEW_NOT_ADMITTED` and do not fit or
re-evaluate it.

## Disposition

No deletion candidate met all three requirements: an independently verified
canonical copy or deterministic regeneration path, no active reader/writer,
and preserved provenance/hash evidence. **No files were removed.** The next
cleanup action needs a per-source receipt review that proves recoverability;
the next R2 action needs a source with an admissible grouped split and a frozen
task that can genuinely require both inputs.

## Follow-up: apparent duplicates are already hard-linked

A read-only check on 2026-11-03 resolved two large logical-size duplicates:

- DROID shard pairs `00003`–`00019` in `droid-expanded-site-holdout-data-attempt-01` and `droid-expansion-sample-20261028-attempt-01` all have matching byte lengths, SHA-256 hashes, and NTFS file IDs. The 17 sample-path aliases total 15,732,444,458 logical bytes (about 14.65 GiB), but already refer to the same NTFS file records as their canonical paths.
- `train-spatial-features.npy` in `clevr4-spatial-count-readout-features-01` and `-02` has matching size (2,949,120,128 bytes), SHA-256 `495F066859CB3879D9371B1C797B24E5ACD47CB88B3E9D34C973ACFA2A1BB460`, and NTFS file ID. It is already one physical file with two names.

These aliases do not account for C: free-space pressure; unlinking either alias would not reclaim the underlying bytes. No hardlinks, files, or attributes were changed. This check covered only the named paths and does not estimate total allocated size across the artifact store. The raw `.npy`/TFRecord contents were hashed, not parsed; no sealed audit was opened.
