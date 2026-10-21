# R0 live-state audit — 2026-10-21 18:34 JST

Additive, read-only reconciliation on the research branch. Older R0 snapshots
remain historical and are not rewritten. GitHub open-PR metadata came from the
public API; worktree/process/device state came from the local Windows host.
No branch, PR, foreign worktree, process, model checkpoint, or dataset was
modified by this audit.

## Remote and pull-request state

- `main`: `09246d7b1821662a6b1eba47fc08c43f9dc3863c`.
- PR #15, the roadmap source branch, remains Draft/Open at
  `71c4db1d4a8f6aed6963c4d31f3d87f9b51ab51c` against `main`; it was not
  changed. Its R0–R8 exit conditions remain the research guide.
- This work branch is `codex/r2-clevrer-fresh-reasoning-20261021` at
  `154cedec547772d940d73afa7816efee307f221a`; it has no PR. Its audio-result
  documentation CI run 39576963637 passed.
- Public GitHub reports the following open PR heads and bases. This is a state
  snapshot, not a review or authorization to merge:

| PR | State | Head | Base |
|---:|---|---|---|
| #9 Teacher v1 | Draft | `8bf351c9` | `main` at `498f557e` |
| #10 postquant worker | Draft | `1fbe0ada` | `codex/roadmap-compression-video-specialization` |
| #11 postquant numerical core | Draft | `8e39f7a8` | `codex/postquant-worker-entrypoints` |
| #12 postquant teacher cache | Draft | `c79a50bd` | `codex/postquant-numerical-core` |
| #13 ternary student recovery | Open | `6a343af3` | `main` at `09246d7b` |
| #14 QAT handoff | Draft | `c3391a0c` | `codex/ternary-student-recovery` |
| #15 pretrained-reuse roadmap | Draft | `71c4db1d` | `main` at `09246d7b` |
| #16 PE-Core T16 audit/readout | Open | `afea6624` | PR #15 branch at `71c4db1d` |
| #17 MAVD/Urbansas source screen | Open | `79f83b55` | PR #15 branch at `71c4db1d` |
| #18 CaptainCook4D task screen | Draft | `13b6fa22` | PR #15 branch at `71c4db1d` |
| #19 AMI cross-modal speaker screen | Draft | `406f003a` | PR #15 branch at `71c4db1d` |
| #20 typed Decision JSON output | Draft | `7c4ac37f` | `codex/r0-r2-continuation-20261021` |

PR #20 adds a typed serializer and tests; it does not run inference, change
parameters, or establish quality. It had two successful CI runs and no review or
issue comments at this snapshot. Its API base SHA is `9037a5b4`, while the
current remote continuation branch is `801c6dc7` (10 commits ahead). Before
any merge, recheck its effective diff and tests against the current base. No PR
was modified or merged.

## Local preservation and compute

- 39 registered worktrees; seven were dirty at capture: six pre-existing
  worktrees with user/experiment changes and this worktree, which contains only
  the intended additive R0 documentation edits. The six pre-existing worktrees
  were left intact:
  - `codex/teacher-quality-clean-dev-v1` — primary worktree, modified ROADMAP,
    Teacher report, new pretrained-reuse notes and a resume script.
  - `fix/student-qat-training-handoff` — two untracked legacy QAT configs.
  - `codex/textfree-video-program-order` — modified video trainer and tests.
  - `codex/textfree-video-coverage-v2` — staged/unstaged video-source and test
    work plus untracked protocol/scripts/tests.
  - `codex/teacher-quality-physionpp` — untracked Teacher resume script.
  - `codex/ternary-student-recovery-updated` — modified recovery implementation,
    docs/config/tests and untracked local config/repack script.
- This audit worktree is based on commit `154cedec`; its current changes are
  limited to `ROADMAP.md`, the R0 task-battery draft, and the two additive live
  audit files.
- At 18:34 JST the RTX 3080 Laptop had 15,821/16,384 MiB used, 355 MiB free,
  and 48% utilization. PID 21224, child of PID 30708, is Orbitune's
  `compound_cfe_train.py train` process with its YIELD sentinel. It is not a
  Tiny Omni run and was left untouched. No new GPU work should start unless the
  cooperative HotUGPU gate grants it.
- C: had 17.87 GiB free and system RAM about 11.26 GiB free. Cloud budget used:
  $0.

## Current evidence that affects R0/R2

- The frozen Speech Commands v0.02 official-test comparison is now complete:
  4,074 keyword rows / 239 speakers; baseline Accuracy/NLL/Brier/ECE
  `0.9772/0.2684/0.0909/0.1692`; the same candidate with frozen train-only
  temperature `0.4270` scored `0.9786/0.0946/0.0372/0.0237`. The Accuracy
  difference interval includes zero. All classes exceed 95.7%, but this is only
  fixed-vocabulary keyword recognition. Full hashes, calibration comparison,
  repeated-content accounting and caveats are in
  `R2_TEXTFREE_AUDIO_OFFICIAL_TEST_PROTOCOL_20261021.md`.
- The already-completed LibriSpeech dev-clean hard-negative transfer remains
  near four-choice chance on 1,938 examples, for both random and
  phonetic/lexical/length-matched distractors. It is low-power OOD development
  evidence, not a second Speech Commands score or general ASR proof. The
  test/train weights were not changed for either audio evaluation.
- The existing source screens do not yet admit a sound both-required AV task:
  - AMI is review-only. Participant split metadata is useful, but speaker-face
    identity, synchronization, privacy/governance and validation labels remain
    unresolved; reviewed visibility is not proof of speaker correspondence.
  - MAVD/Urbansas is not admitted: source-session overlap and upstream-test
    provenance remain unresolved, and its labels do not prove exact
    audio-to-visible-object correspondence.
  - CaptainCook4D is a provisional text+video task, not an AV task. Only two
    validation participants support the cited split, so it cannot qualify a
    high-accuracy held-out task.
- The pinned VideoPrism-B remains the currently executable/allowed video
  checkpoint candidate; exact V-JEPA 2.1 rights remain REVIEW. The open PE-Core
  PR adds another candidate but is not merged or independently adopted here.
- The earlier fixed R2 audio/video/image/text probes and the 687.3 MB modular
  assembly remain task-specific research artifacts, not one qualified Omni
  model. Independent video, broad speech, natural-image and both-required AV
  gates remain unmet.

## R0 disposition and next gates

R0 remains open. The engineering-host and provisional parameter/package targets
are recorded, but no end-user device, abstention/calibration floors, latency,
energy, or user-approved money/time release gates are frozen. The task battery
is still a proposal rather than a locked product acceptance contract.

Next: reconcile the task-battery draft with the new fixed audio test, preserve
separate narrow-task claims, and require a source/task gate for broad speech and
both-required synchronized AV before any corresponding fit. Continue with the
frozen-feature Path A baseline; do not add shared-Lens or weight-merging work
without measured evidence. Do not open sealed audits, alter foreign worktrees,
promote a candidate, merge a PR, or interrupt the live GPU process.
