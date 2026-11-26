# R0 live-state refresh — 2026-10-21 11:16 JST

This is an additive snapshot on the PR #15 research line. It does not replace
`R0_INVENTORY.json` or the Oct 18–20 live audits. It reflects current GitHub,
worktree, local compute and artifact evidence gathered read-only from
`codex/r0-r2-continuation-20261021` at `8b6d657c2377e07f4bf80672c22be72b0631ce18`.
The machine-readable snapshot is `R0_LIVE_AUDIT_20261021.json`.

## Repository and preservation

- `main` is `09246d7b1821662a6b1eba47fc08c43f9dc3863c`.
- PR #15 remains Draft/Open at `71c4db1d4a8f6aed6963c4d31f3d87f9b51ab51c`.
  The present research branch is a separate child commit; it does not modify PR #15.
- Open PRs #16–#19 hold a PE-Core T16 audit and three audiovisual source screens.
  PRs #9–#14 are also open on their distinct Teacher / ternary lineages. No PR was
  merged, retargeted, rebased, closed, or otherwise changed.
- 34 registered worktrees exist; six are dirty. The primary user worktree is
  dirty on `codex/teacher-quality-clean-dev-v1`. Its contents were not edited.
  Existing checkpoint/data/worktree state was preserved.
- The current branch has no PR. Its previous AV screen commit is pushed; its CI
  passed. This audit itself is a documentation update only.

## R1 and Path A evidence in the current state

The pinned candidate inventory contains 23 checkpoint candidates. VideoPrism-B
is the clearest currently allowed frozen video candidate: 114,365,184 parameters,
checkpoint SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`. V-JEPA 2.1
remains `REVIEW` for rights, and PE-Core T16 is under the separate open PR #16.
The Path A v4 historical modular bundle contains 170,499,716 parameters but
687,254,772 payload bytes, so it is within the provisional parameter range and
above the provisional 200 MB package target. Quantized package size and the
release-quality battery remain unverified.

Path A already has task-scoped text, image, audio and video probes, plus multiple
saved/reloaded frozen runtimes. They are not a unified, independently qualified
Omni Decision system. Existing R2 records include 88.30% image taxonomy macro
Accuracy on an internal training-image split, closed-set audio tasks, synthetic
text controls, and CLEVRER cached-video decisions. These scores cannot be added
together as general quality.

## Audiovisual evidence reconciliation

The new 512-example Perception Test source-visibility screen is one small,
video-disjoint binary task. Its frozen Audio/Video/Joint readouts scored 56.25%,
57.81%, and 64.84% balanced validation Accuracy, with 100% train Accuracy in
all arms. Keep it as a low-quality, high-overfit diagnostic; do not fit again or
promote it.

A separate, stronger existing Perception Test source-correspondence experiment
is already present on `codex/r2-perception-av-scale-v1` (head
`58633f10a6010f31dc5a17c59ce9a7c09b462db9`). It used 388 videos / 908 events,
matched 2,048-update fits, and an already scored 146-event development cohort.
Expanded fit coverage did not establish an Accuracy gain. On a separate fixed
checkpoint transfer subset of 200 previously unused TRAIN videos / 504 events,
Joint scored 39.48% Accuracy and 1.8466 NLL versus capacity-matched Video-only
35.32% and 2.0045; the Accuracy paired interval included zero, while the NLL
interval favored Joint. This is narrow, train-split development/transfer evidence,
not a blind test. Do not repeat this experiment or treat either AV task as the
product battery.

The completed PCH recording-LSTM comparison also did not show clear temporal
benefit: chronological Accuracy 63.49% / NLL 1.2019 versus the frozen window-LSTM
baseline 65.08% / 1.0117 on the exposed development split. It is not evidence
against all recurrent models, but it does not justify another LSTM fit on this
same PCH split.

## Compute and operating constraints

At the snapshot, the RTX 3080 Laptop had 15,373/16,384 MiB in use and 803 MiB
free; the active CUDA workload was Orbitune `compound_cfe_train`, protected by
its cooperative idle supervisor. It was left untouched. C: had 18.15 GB free;
RAM had about 8.77 GB free. No Tiny Omni GPU task was active. Do not start direct
GPU work while occupied; use only the reviewed HotGPU cooperative reservation
guard after Orbitune yields. No paid/cloud compute was used.

## R0 gates and next action

- Engineering reference: local Windows PC / RTX 3080 Laptop 16 GiB; this is not
  a final deployment-device selection.
- Provisional engineering targets: 100–200M pre-quantization parameters and a
  complete package no larger than 200 MB. The latter is not measured after
  quantization.
- 95–98% remains the aspiration, not a locked acceptance rule. A task battery,
  per-task quality/confidence floors, calibration/abstention policy, latency,
  energy and local time limits are still undefined.
- Paid/cloud budget remains zero; live workloads must not be interrupted.

The next useful work is to consolidate the already-existing Path A modality
probes against a task-specific locked battery, and admit only a fresh grouped
source that supplies actual both-input-required labels before another fusion
fit. The current Perception visibility and PCH temporal experiments are closed
as decision inputs. M-0/M-1 and M-2 remain incomplete; R0 is not closed.

No test split or sealed audit was opened in this refresh.
