# RoboMNIST complete directory index implementation plan

> **For agentic workers:** Use superpowers:executing-plans; implement inline.

**Goal:** Actually inspect all60 public ZIP directories under a fixed small budget.
**Architecture:** New runner reuses pinned source grouping and bounded range reader.
No payload read, new split or fit. Separate CPU replay checks saved inventories.
**Tech Stack:** Existing Python3.11/stdlib zipfile/urllib/JSON; no dependency upgrade.
**Spec:** `docs/pretrained_reuse/R2_ROBOMNIST_DIRECTORY_PROTOCOL_20261014.md`.

## Global constraints

All60 ordered directories;512KiB/archive,12MiB total,900s body;16MiB output and
1GiB reserve for metadata only. GPU8GiB guard unchanged. No payloads/models/fit,
GT/audit, legacy intervention/cloud/main merge. New output, failclosed, no restart.

## Review focus

Cross-archive class substitution must fail; every real record has all8 streams.
Transport/body caps include failed reads; an incomplete prefix never becomes PASS.
Receiver views never become separate sample identities; positionGT not an input.
Consumer replay must bind actual files/config/source hashes, not merely a PASS flag.
No sampled-directory count extrapolation, blind quality or OOD claim.

### Task1: Bounded acquisition and complete-directory accounting

Create `scripts/audit_robomnist_all_directories.py` and
`tests/test_robomnist_all_directories.py`.
Interfaces: `Budget.charge(count, elapsed)` rejects negative/per-read/total/body
violations; `inspect_infos(infos,archive_name)` returns exact group streams/counts.
Tests: positive/negative/overlimit charges; correct firstclass; crossclass/missing
stream rejected; macOS forks do not add records. RED→GREEN, Ruff/full CPU, commit
source/protocol before actual execution. Reuse group_members(expected_folder).

### Task2: Actual run, independent metadata replay and publication

New output `C:/CodexArtifacts/pretrained-reuse-r2/robomnist-all-directories-01`.
Record exact committed source/protocol hashes, immutableversion/config/command;
confirm PID/process and all60 completion. Never restart on timeout/poll loss.
Run independent CPU saved-directory/group identity/accounting checks; record
actual bytes/resources, failures/unknowns. Minimal README/ROADMAP note, commit/push
small results/code/tests, check exact-head Actions. Leave unmet overall goal active.
