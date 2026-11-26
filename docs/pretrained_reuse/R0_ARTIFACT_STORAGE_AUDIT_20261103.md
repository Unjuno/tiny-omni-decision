# Artifact storage audit — 2026-11-03 JST

## Snapshot

- C: free space: **7.40 GiB** at inspection time. This is above the repository's 2 GiB minimum reserve, but does not support a new large media/feature acquisition without another preflight.
- `C:\CodexArtifacts\pretrained-reuse-r2`: **683.42 GiB**, 507,338 files by summed logical file lengths. This is a mixed store of source archives/media, derived features, model outputs, and diagnostics; directory size alone does not imply that content is disposable.
- `C:\CodexArtifacts\pretrained-reuse-r1`: **11.37 GiB**.
- The currently open OneDrive repository contains **16.59 GiB** under `data/` and **13.38 GiB** under `artifacts/`. These are legacy corpora and experiment checkpoints; they were not changed.

Largest R2 directories observed:

| Directory | Size | Initial disposition |
| --- | ---: | --- |
| `holoassist-textfree-unused-train-recovery-02` | 112.16 GiB | Retain: source media/cache and recovery provenance |
| `holoassist-textfree-unused-train-attempt-01` | 72.80 GiB | Retain pending source/manifest reconciliation |
| `clevr4-100k-videoprism-v0` | 67.27 GiB | Retain: source archive and derived inputs |
| `greatest-hits-video-cache-01` | 65.61 GiB | Retain pending source/task status review |
| `holoassist-video-pilot-160-207-seed17-pitchshifted-attempt-01` | 33.76 GiB | Retain: experiment reproduction data |
| `droid-expanded-site-holdout-data-attempt-01` | 17.34 GiB | Retain pending source/task status review |
| `droid-expansion-sample-20261028-attempt-01` | 14.65 GiB | Retain pending canonical-shard reconciliation |
| `perception-test-av-v1` | 14.50 GiB | Retain: fixed evaluation assets and provenance |

## Cleanup decision

No large item was deleted. The large directories are datasets, cached media/features, or experiment artifacts that can be needed to reproduce a screen or compare a later candidate. The audit found 26 `.part`/temporary files totaling **1.765 GiB**; they are stale and no Python, FFmpeg, or downloader process was active, but partial downloads may still be resumable, so their usefulness has not been disproved. Keep them until each source receipt says whether resume is supported and whether that source remains eligible.

Two CLEVR4 spatial feature files are each about **2.747 GiB** and have previously been identified as duplicates. A prior automated deletion review rejected removing one; this audit does not retry that action or alter either copy.

Small caches and status files are not a meaningful space recovery. Do not remove checkpoints, corpora, source media, feature caches, or partial downloads based only on age or directory names. A future cleanup candidate must have (1) a manifest/receipt proving another verified canonical copy or deterministic regeneration path, (2) no active reader/writer, and (3) a recorded hash and recoverability check before deletion. Keep selected checkpoints and the minimal source/feature set needed to reproduce published reports.

## Scope and limitations

This is a read-only inventory snapshot, not a byte-for-byte duplicate audit of all 683 GiB. Sizes are logical file lengths; sparse-file allocation and hard-link deduplication are not reflected. No dataset contents, sealed audit data, model weights, media, or checkpoints were opened or changed. The separate current OneDrive worktree has user modifications; it was left untouched. The R2 worktree still has the user's two untracked PhysicalAI split manifests; they were not staged or changed.
