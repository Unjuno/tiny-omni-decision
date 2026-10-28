# R0/R2 cross-branch evidence recheck — 2026-10-28

This read-only reconciliation prevents repeating source acquisition or
development evaluations already completed on concurrent branches. It does not
merge or modify those branches, their worktrees, PRs, datasets, or checkpoints.

## Repository state

- Roadmap branch: `docs/pretrained-reuse-roadmap-20261010`, head
  `48ba77810da1ae0f833e68d3dee3d3fa217d4639` (PR #15 remains open).
- PR #31 is open at `codex/r2-droid-site-universe-audit-20261028`, head
  `aefb1a1a3cd09959e73dfac79dbf319fce63c367`; PR #32 is open at
  `codex/r2-droid-temporal-transfer-20261028`, head
  `a8db60d262b9655cb00897f26cdec9ee6a177070`. They build on separate DROID
  evidence branches; neither was merged or modified here.
- The active OneDrive worktree on `codex/teacher-quality-clean-dev-v1` has
  pre-existing uncommitted work. It was left untouched; this recheck is in a
  separate worktree on the roadmap branch.
- No model weights or raw media were opened, and no inference or validation
  evaluation was rerun; this recheck read saved reports, aggregate receipts and
  public PR metadata only.

## Latest source and task evidence

| Candidate | Latest evidence | Disposition |
| --- | --- | --- |
| DROID video + robot state | `codex/r2-droid-expanded-sites-20261028` at `23f02483`; fixed site-held-out validation: Video + sensor accuracy 0.3481, NLL 1.9034, Brier 0.8508, ECE 0.2447. Sensor-only was 0.3702, 1.5128, 0.7219, 0.0904. Permuting video worsened the joint readout, showing some aligned signal, but not useful complementarity over sensor-only. A separate six-shard sample found zero new sites. | Do not repeat site acquisition or fit this task again on exposed validation. A new task would need a prospectively justified video-essential target. |
| ADL audio + vibration | `codex/r2-droid-temporal-transfer-20261028` at `a8db60d2`; 25/25 archives pass publisher MD5 and local SHA checks. The source has 12 training and 2 development-validation participants, 231 unassigned `no-activity` rows, only 24/25 activities in both splits, and no video. The first-three download sample averaged 0.2136 MiB/s; the remaining corpus was already acquired, so no repeat download is warranted. | Integrity is established, but the source is too small and imbalanced for the roadmap's independent-group quality gate. Do not treat it as product-quality evidence or as AV. |
| GOOSE image + point cloud | `codex/r2-droid-temporal-transfer-20261028` at `a8db60d2`; 15 train-only scenario groups yielded 561 2D `(class, instance)` keys, of which 283 matched 3D keys (50.45% weighted coverage). Agreement varied substantially by scenario. | Do not use the current labels as Decision supervision; no full validation download. |
| GRID audio + video | `codex/r2-audio-vibration-source-20261027` at `2721093f`; corrected 64-quartet/256-query development probe scored 34.38% Accuracy, 2.0492 NLL, 0.9109 Brier, 0.3369 ECE. Video correspondence perturbation reduced Accuracy to 17.97%; audio mismatch gave 24.61%. The corrected direct-source split has 24 train and 8 validation speakers, but cross-corpus identity against Lombard remains unknown and the result is not blind. | This is the strongest current AV input-dependence diagnostic, but the weak calibration/quality and source-identity uncertainty do not justify promotion or tuning the exposed validation. Confirm data-use scope and cross-corpus identity before considering a separately initialized train-only readout. |

These results are development evidence only. They do not establish the 95–98%
goal, do not close R2-M, and do not justify shared-Lens or weight-merging work.
No large download or GPU fit was started. The next R2 action requires a newly
admitted, adequately grouped task where both modalities are needed, with
option/metadata priors and matched unimodal/perturbation controls fixed before
fitting.
