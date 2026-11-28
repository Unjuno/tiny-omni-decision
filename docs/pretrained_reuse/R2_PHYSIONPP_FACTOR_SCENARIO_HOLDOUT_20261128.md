# Physion++ factor-only scenario-holdout diagnostic — 2026-11-28

## Question and protocol

The earlier factor-only baseline grouped folds by source asset, but the same
scenario families appeared in train and held-out folds. This fixed follow-up
tests whether the recorded scenario/filename factors transfer to an entirely
unseen scenario.

- Input: only the pinned 652-row train manifest; SHA-256
  `0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b`.
- Ten leave-one-scenario-out folds, each holding all rows from one scenario
  out. All ten scenarios contain both labels. Source asset groups are checked
  for cross-fold overlap.
- Fixed categorical naive Bayes with add-one smoothing (`alpha=1`), using the
  scenario and source-record filename factors. An unseen held-out scenario
  category contributes no learned likelihood; no tuning was performed.
- Comparator: the train-fold smoothed class prior.
- No media, model, GPU, validation, or sealed/test data were loaded.

## Pooled results

| Grouped OOF readout | Accuracy | Balanced accuracy | NLL | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|
| Train-fold prior | 0.5199 | 0.5000 | 0.6967 | 0.2518 | 0.0004 |
| Recorded factors only | 0.5000 | 0.5006 | 0.7154 | 0.2598 | 0.1027 |

Per-scenario factor-only Accuracy ranges from 0.35 on `friction_platform_pp`
to 0.84 on `friction_collision_pp`; the strong result is not consistent
across scenarios. Full per-scenario counts and metrics are in the JSON report.

## Interpretation

The 59.82% source-asset-grouped OOF result from the earlier diagnostic does not
transfer as an aggregate when the entire scenario is held out: pooled
Accuracy is 50.00% and NLL is worse than the fold prior. This points to
scenario-specific associations in the recorded factors, with some scenario
exceptions; it does not prove the video dataset is contaminated or that a
vision model learned a shortcut. The within-scenario signal and weak
first-frame appearance baseline remain distinct observations.

Continue with a blinded, full-resolution human review and rendering-cue checks.
Do not change architecture based on this small train-only diagnostic. All rows
are already observed development data and cannot support blind generalization
claims.

## Reproduction

- Script: `scripts/audit_physionpp_factor_scenario_holdout.py`
- Tests: `tests/test_audit_physionpp_factor_scenario_holdout.py`
- Output:
  `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-factor-scenario-holdout-20261128-attempt-01`
- `audit-report.json` SHA-256:
  `2e84ca826d35528410692ae6b8b4c3544a26041d9cd4059b51661ef2dc4f9e31`
- `oof-predictions.jsonl` SHA-256:
  `136f29a260d53af2ce42a6623d43bd7c5b6e10ec9c9cb57bdce44e39d4df93b9`
- Script SHA-256:
  `26e7b195b5d3caae7d8de01fe84a74ee4905d764856db59273f42956e9343674`
