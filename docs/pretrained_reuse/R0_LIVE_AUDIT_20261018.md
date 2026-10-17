# R0 live-state audit — 2026-10-18

This is an additive current-state snapshot. It does not rewrite the original
2026-10-10 `R0_INVENTORY.json` or change any legacy worktree, artifact, run,
checkpoint, or dataset.

## Repository and pull requests

- Repository: `Unjuno/tiny-omni-decision`.
- Audited worktree: `C:/CodexWorktrees/pretrained-reuse-omni`.
- Branch: `docs/pretrained-reuse-roadmap-20261010`.
- HEAD and `origin/docs/pretrained-reuse-roadmap-20261010` both resolve to
  `ebb60433dacdec1d002b52364e09914273e574ca`; worktree was clean.
- Public GitHub REST state at audit time: PR #15 remains open and draft, based
  on `main` at `09246d7b1821662a6b1eba47fc08c43f9dc3863c`. PR #9–14 also remain
  open; #13 is non-draft, the others are draft. Their head/base refs and SHAs
  were fetched and are retained in the audit command output. `gh` authentication
  is unavailable, so private review comments and authenticated CI details were
  not inspected.
- `main` currently points to `09246d7b1821662a6b1eba47fc08c43f9dc3863c`.

## Worktree preservation

All 15 registered worktrees were enumerated read-only. The audited roadmap
worktree and most other worktrees are clean. Existing user changes remain in
place: three uncommitted entries in `codex/teacher-quality-clean-dev-v1`, two
untracked YAML files in the PR #14 review worktree, one untracked script in the
Physion++ worktree, and eleven modified/untracked entries in the updated
recovery worktree. None were edited by this audit. Exact paths and current
HEADs remain available from `git worktree list --porcelain`.

## Manifest reconciliation

The original R0 snapshot recorded `manifests/evaluation-candidates.yaml` before
the LibriSpeech hard-negative evaluation source was added. Current SHA-256 is
`2787c66d920d01dee53e3df4aa0f52eb9097f52e5fb8a35f2994b6eaba089f56`; the
change is the three-line catalog entry for
`candidates/librispeech-dev-clean.yaml` with adapter
`librispeech-hard-negatives`, introduced by
`341f153ef05986ade2e72ef6bac23a51c4ad3955` (`feat: evaluate Path A audio
transfer with hard negatives`). The referenced manifest is an evaluation
candidate; this audit did not load its audio payloads.

`manifests/candidates/open-jev-test.yaml` matches the original snapshot exactly:
SHA-256
`dc83003fee71451a3342f3d7bebca0032936566e050f2eccbeb268412676d9f47`.
An earlier working note had a one-character transcription error in this hash;
there is no current file change to reconcile.

## Pinned-candidate inventory and scope

`docs/pretrained_reuse/R1_CANDIDATE_INVENTORY.json` contains 23 pinned
candidate records. It includes real checkpoint digests, parameter counts,
license dispositions, and runtime evidence where actually tested. Several
candidates remain `REVIEW`, `UNKNOWN`, or quality-unmeasured; inventory presence
does not admit them for training or release. In particular, V-JEPA 2.1's exact
checkpoint rights remain `REVIEW`; VideoPrism-B has an allowed pinned weight
basis and tested runtime, but its CLEVRER result is development-only and has
already been used for selection. Text, Speech Commands, CLEVR-4 and selected
video probes are task-specific Path A evidence, not proof of general multimodal
quality.

## Live local compute state

At 2026-10-18 06:06 JST, the RTX 3080 Laptop reported 15,821/16,384 MiB used,
48% utilization, 60 C, and a single CUDA process PID 18224. Process ancestry
identified PID 18224 as `compound_cfe_train.py train` in the separate
Orbitune/MIDI project, with PID 21472 as its parent trainer and PID 6452 as an
`idle_train.py` supervisor. This is not a tiny-omni-decision run. No process was
stopped or changed. No new GPU workload was launched during this audit.

The user has said that Orbitune is configured to yield when a new training job
starts. This audit did not verify that external watchdog's behavior; no such
handoff was attempted. More importantly, the current conditional AV design has
no admitted genuinely both-required paired task in its frozen common corpus
(zero joint audio/video observations in TRAIN and validation), and the latest
source review explicitly withholds that benchmark admission. Therefore this
audit does not start another AV-gate quality run on unsupported labels. The
61,634-parameter gate itself is not the material compute bottleneck; dataset
validity and full encoder execution dominate.

### Supplemental cross-branch reconciliation (06:14 JST)

After the first publication, the existing `codex/r2-av-postnorm-stability`
branch was inspected before deciding whether to train. It already contains a
matched E0G1 conditional-gate fit: seed 17, 32,768 updates, selected step
24,576, fresh optimizer from the fitted E0G0-8192 start, and exact best/final
reload verification. The run used the separate exploratory Greatest Hits AV
corpus; this is distinct from the common corpus, which still has zero paired
audio/video observations. The source review admits the former only for scoped
native-field exploration, not as a both-required or precise-synchrony benchmark.

The gate outputs were below 0.01 for every valid option by step 512 and remained
near zero. At step 32,768, joint-view Accuracy was +5.07 pp versus the E0G0
control, but the paired scene-bootstrap interval for joint NLL included zero;
video-only Accuracy was -3.83 pp with worse NLL. Since the residual also has a
query path, the gain does not demonstrate gate-based AV evidence use. The full
report and learning curve are on the [separate experiment branch](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-av-postnorm-stability/docs/pretrained_reuse/R2_AV_GATED_POSTNORM_LONG_FIT_20261018.md).

The report's next action is inference-only ablation and fixed input
replacement, not another training variant. A fresh guarded scoring attempt
started on that branch, completed 109 of 44,375 forwards, then stopped when
Orbitune PID 10324 acquired the GPU. Its partial output is preserved but has no
complete sensitivity summary and is not used as evidence. Other attempts either
failed their GPU preflight or stopped before a complete result. At 06:14 JST,
`nvidia-smi` showed no CUDA process, while the Orbitune supervisor/trainer
processes remained present and could resume. No user process was stopped or
modified.

## Current decision and open R0/R2 gates

The next action is not to repeat an existing run or to use the sealed audit.
Keep existing Path A and AV artifacts read-only. Before any future AV-gate
quality fit, admit a rights-cleared, source-grouped paired task with verified
alignment and defensible audio-sufficient, video-sufficient, and both-needed
labels, plus fixed swap/missing/corruption controls. R0 still needs explicit
deployment-device and per-task acceptance criteria. The product target, joint
quality, independent generalization, calibration, package/runtime limits and
95–98% per-modality aspiration remain unmet or unverified.

No training, sealed-audit evaluation, artifact promotion, paid compute, or
mutation of legacy worktrees/data/checkpoints occurred in this audit.
