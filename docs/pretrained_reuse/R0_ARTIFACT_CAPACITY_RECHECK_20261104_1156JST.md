# R0 artifact capacity and retention recheck — 2026-11-04 11:56 JST

## Snapshot

- Worktree: `C:\CodexWorktrees\r2-perception-control-freezer-20261031`.
- Branch: `docs/pretrained-reuse-roadmap-20261010`, at `d7ab603dc1bc55514c6b6192b18508251904d8d0` before this record.
- C: had **8,369,664,000 bytes (7.80 GiB)** free before the cache cleanup and **8,552,054,784 bytes (7.96 GiB)** after it, a net increase of **182,390,784 bytes (174 MiB)**.
- `C:\CodexArtifacts\pretrained-reuse-r2` contains **521,491 files / 684.97 GiB logical length** in the current scan. This is not allocated disk usage: hard links and sparse files make logical totals differ from physical allocation.
- No Python, Quartus, FFmpeg, curl, or wget process appeared in the process query. The RTX 3080 reported 11 MiB used / 0% utilization.
- The two untracked PhysicalAI split manifests in the R2 worktree were not opened, changed, or staged. The separate OneDrive worktree's modifications were also left untouched.

## Cleanup performed

The user requested organizing artifacts and removing items with no expected reuse. The local pip HTTP cache was the only clearly reproducible cache confirmed in this pass. `python -m pip cache purge` removed 120 cache files; the measured free-space increase was 174 MiB. Installed environments and repository lockfiles were not changed.

## Retention decision

No model, checkpoint, source media, corpus, frozen feature cache, experiment run, or partial download was removed. The previous R0 audits require all three of the following before deleting experiment artifacts: receipt-level proof of a verified canonical copy or deterministic regeneration, no active reader/writer, and preserved hash/provenance evidence. No newly reviewed artifact met all three conditions. Age, a `tmp`/`unused` name, or a matching hard link is not a sufficient cleanup reason; unlinking an already hard-linked alias would not recover its underlying storage.

The temporary `C:\CodexArtifacts\venvs\robomnist-av-match-resume-20261104` environment is 1.85 GiB and lacks Flax; a separate pinned environment successfully completed the RoboMNIST cache extraction. Its cleanup was previously rejected by automatic action review. This pass did not retry or use another deletion route. The environment remains an identified, potentially reclaimable item, but is not counted as freed capacity.

## Next safe capacity action

Keep the current evidence and reuse assets. Before any further artifact cleanup, reconcile an individual partial/cache against its source receipt, canonical hash, resume or regeneration procedure, and active-process status. If no item passes that check, treat current capacity as a constraint and avoid new media acquisition; do not infer that the logical-size inventory identifies removable bytes.
