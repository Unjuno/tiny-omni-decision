# Bare-finger source audit implementation plan

**Goal:** Verify complete source grouping and distinguish native contact streams
from texture-shared photos before any Decision fitting.

**Spec:** `docs/pretrained_reuse/R2_BAREFINGER_SOURCE_PROTOCOL_20261015.md`.
**Architecture:** Pure metadata accounting, immutable receipt checks, no upstream
code/model execution. Separate one-record native format inspection outside Git.
**Execution:** Inline, per the user's autonomous public-source authorization.

## Tasks

- [ ] Add failing tests for all500 Cartesian trials/200 photos, malformed names,
  missing/duplicate trials, unsafe/aliased members and participant/photo distinction.
- [ ] Implement minimal source accounting and a bounded new-path auditor with
  before-parse hashes and terminal source rechecks; pass the focused tests.
- [ ] Commit executable audit before running it on immutable local receipts.
- [ ] Execute whole-directory accounting and fixed native numeric/image inspection;
  record real schema/access limitations without quality or synchronization claims.
- [ ] Publish small results, run Ruff/CPU tests/manifests, commit/push/check exact CI.

## Constraints

All500 trials,10 participants,10 texture identities,200 images; no source pruning,
invented video, split assignment, training or sealed audit. Native pilot selection
is fixed by source identity before reading payload, with no performance selection.
Every repeat/view of one underlying source belongs to the same physical group.
CPU source audit60s/1GiB/report1MiB/disk reserve1GiB; preserve prior GPU8GiB guard.
