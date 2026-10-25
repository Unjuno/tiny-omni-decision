# R0 live-state refresh — 2026-10-26 02:47 JST

This additive snapshot was taken read-only from
`codex/r2-synthetic-av-gated-temporal-20261026` at
`dd1f7fe3487c6474ab8a0eb60b9884966407b544`, synchronized with its remote
tracking branch. It does not replace earlier R0 snapshots or alter any other
worktree, process, PR, corpus, model, or run. Machine-readable details are in
[`R0_LIVE_AUDIT_20261026.json`](R0_LIVE_AUDIT_20261026.json).

## Repository and preservation

- Git reports 60 registered worktrees; six have tracked or untracked local
  changes. The primary user worktree remains dirty on
  `codex/teacher-quality-clean-dev-v1`; it was not edited. The current R2
  worktree is clean. All six dirty worktrees and every separate branch were
  left untouched.
- GitHub reports 22 open PRs (#9–#30). PR #15 remains Draft/Open against
  `main`; the R0 refresh PR #28 is Draft/Open against PR #15's roadmap branch.
  PRs #9–#14 remain on separate Teacher/ternary lineages. No PR was changed.
- The current branch has no PR. Its latest CI `test` check completed
  successfully for `dd1f7fe`.

## Processes and local resources

- The only matching Python processes are Orbitune's idle-training supervisor
  (PID 4696) and child (PID 23856), started 2026-10-20. Their command line is
  `tools/idle_train.py` with the external Orbitune watchdog; neither points to
  this repository. They were left running.
- The RTX 3080 Laptop reported 53 MiB / 16,384 MiB used and 0% utilization at
  the snapshot. C: had 42,032,627,712 bytes free; Windows reported
  14,001,926,144 bytes free RAM. No Tiny Omni training or feature extraction
  process was active. Any future GPU work still must use HotUGPU and pass its
  live reservation checks.
- No cloud compute or spend was used.

## R1/R2 evidence and open gates

- A separate 2026-10-26 CPU recheck reloaded pinned VideoPrism-B, Ruri-v3-30m,
  and EfficientAT artifacts and observed finite outputs. This is runtime
  evidence only; it provides no new task-quality result. VideoPrism remains
  114,365,184 parameters; EfficientAT 1,961,239; Ruri 36,705,536.
- The Path A reference bundle is recorded at 170,499,716 unique parameters and
  687,254,772 bytes. It meets the provisional pre-quantization parameter range
  but exceeds the provisional 200 MB package target. It is task-scoped and is
  not a qualified unified multimodal runtime.
- Existing text-free probes remain narrow: audio is 96.56% accuracy with
  materially worse calibration than its text-conditioned control; image macro
  accuracy is 88.30%, with Count at 58.19%; video temporal readout is about
  42% with clear overfitting; the sensor result has only three validation
  participants and weak class-balanced/transition behavior. These results do
  not satisfy the user’s per-modality 95% target.
- The current synthetic AV architecture sequence found that the direct raw
  temporal LSTM solves its generated rule, while multiple learned gate variants
  remained at chance. An identity-initialized residual gate matched the LSTM
  but did not improve it. This remains synthetic-only evidence.
- The separate Perception Test source-candidate branch completed a fixed,
  oracle-assisted train-to-development readout at 56.93% set Accuracy on 606
  events / 80 development videos. Input controls show feature dependence, but
  development NLL rose after epoch 7 while train loss kept falling. The source
  is the official training partition, the split is non-blind, and 80 videos
  are below the provisional 200-group floor. Do not continue tuning this split.
- No reviewed natural synchronized AV or AV/sensor source currently clears all
  rights, independent grouping, label-semantics, and both-input-necessity
  gates. R2 M-2 remains open; R3/R4 and quantization/recovery remain gated on
  stronger Path A task evidence. No sealed audit was opened.

## Next action

Continue R2 with task-specific, source-grouped Path A evidence. Admit a new
natural AV or AV/sensor task only after its exact data rights, independent
groups, frozen labels, and unimodal-shortcut controls pass before model
selection. Do not add a shared Lens, weight merge, or another gate solely from
the synthetic result. Keep the current 95% per-task goal and size targets
unchanged; the final task battery, confidence/abstention policy, and release
criteria are still not fully frozen.
