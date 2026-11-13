# R0 artifact retention and capacity recheck — 2026-11-04 12:23 JST

## Scope and current state

- Worktree: `C:\CodexWorktrees\r2-perception-control-freezer-20261031`.
- Branch: `docs/pretrained-reuse-roadmap-20261010`, source head `c0f1b7aa6b11804bea1538ff755cbac1b27672e7`; local branch matched `origin` before this record.
- The two untracked PhysicalAI split manifests were not opened, modified, staged, or used.
- C: had 7.23 GiB free at the 12:23 JST sample, down from 7.96 GiB at the 11:56 report. This is a point-in-time capacity sample, not attribution to a specific process or artifact.
- RTX 3080 Laptop reported 15,832 / 16,384 MiB used and 66% utilization. Read-only process inspection found Orbitune's separate training worker and watchdog active; no Tiny Omni GPU job was started and no process was stopped.

## Read-only artifact inventory

- `C:\CodexArtifacts\pretrained-reuse-r2` scanned at 521,491 files / 684.97 GiB summed logical length. This is not allocated disk usage because hard links and sparse files affect physical allocation.
- The largest folders are source media, feature caches, or experiment outputs. The prior retention rule still applies: require a verified canonical copy or deterministic regeneration receipt, no active reader/writer, and preserved hashes/provenance before deleting any experiment artifact.
- `C:\Users\junny\AppData\Local\Temp` has 6.48 GiB summed logical file length. This includes two old `pip-unpack-*` directories (0.52 GiB total, last writes 2026-10-03 and 2026-10-07) containing Torch 2.10 / 2.6 and dependency wheels; no active Python/pip command referenced those paths during the inspection. They remain because exact environment-recovery value and canonical wheel hashes were not established.
- The Hugging Face cache is 2.80 GiB and remains as reusable model data. The pip HTTP cache is currently empty. OpenCode temporary model/checkpoint files and WSL crash dumps also remain; no deletion was attempted.
- No item passed the established recoverability gate in this read-only pass. Consequently, no model, checkpoint, corpus, media, feature cache, partial download, environment, or temp file was removed.

## Next cleanup gate

Review one concrete candidate at a time against its run/source receipt, exact hash, canonical copy or reproducible recovery command, and active file readers/writers. Only reclaim capacity after those checks prove that the candidate is both unnecessary and recoverable. Do not use age, directory names, logical-size totals, or hard-link aliases alone as deletion evidence.

## Work impact

The R2 source/task admission state is unchanged: the current candidate matrix still has no synchronized multimodal task that passes rights, independent-group, both-input-necessary, and quality gates. The active Orbitune workload also leaves too little free VRAM for a safe new GPU experiment. CPU-only code and source-gate work can continue; no additional dataset acquisition or speculative fitting is justified by this capacity check.
