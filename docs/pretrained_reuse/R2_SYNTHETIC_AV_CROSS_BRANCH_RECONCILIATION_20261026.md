# R2-M synthetic AV temporal run: cross-branch artifact reconciliation

Captured 2026-10-26 from the independent worktree for
`codex/r2-synthetic-av-gated-temporal-20261026`. This record reconciles the
already completed experiment with the active pretrained-reuse roadmap. It does
not rerun training/evaluation or modify the source branch, run, checkpoint, or
external artifacts.

## Source state and experiment scope

- Source worktree HEAD: `4a9f856ab010254fee6f83f438cc52ef77c9a311`
  (`feat: add text-only Open Images control`); the worktree was clean and its
  local branch matched `origin` at inspection.
- The residual-gate training receipt binds producer source commit
  `0f471a9c3109b09ab322e8c34764ebf999917623` and experiment
  `synthetic-av-residual-gated-temporal-v4`.
- Seed 17; frozen synthetic AV feature cache; 4,096 training rows / 512 groups
  and 1,024 validation rows / 128 groups. The same four signal prototypes are
  shared across train and validation by design, so this split measures the
  generated rule task rather than independent natural-world generalization.
- Training completed 16,384 updates (128 epochs, 32-example batches), with
  524,288 row presentations but only 4,096 unique training rows. The selected
  and terminal checkpoint are both step 16,384. Validation selected the
  checkpoint; it is not blind.
- Reported trainable readout size: 595,844 parameters. The stated sum with
  frozen feature towers is 116,922,267, but those offline feature towers and
  this synthetic readout are not a complete product runtime.

## Result and interpretation

The plain concatenated raw-input temporal LSTM and the identity-initialized
residual-gated LSTM both reached 100% Accuracy on the selected synthetic
validation cohort. The residual-gated checkpoint scored NLL 0.000645, Brier
0.000240, ECE 0.000572; the plain LSTM's selected checkpoint scored NLL
0.000585, Brier 0.000278, ECE 0.000493. The residual gate therefore did not
establish a quality gain over the simpler recurrent baseline. Its audio-only,
video-only, missing-both, and shuffled-audio controls fell to chance, while
reversing both modalities together preserved the sequence comparison. These
controls show dependence on aligned synthetic inputs for this task only.

Two previous conditional-gate variants on the same synthetic task remained at
chance for their full 16,384-update budgets. This evidence argues against more
architecture tuning on the exposed synthetic task; it neither validates a
natural AV gate nor proves an architecture-family limit. No natural-data task
or product candidate is selected by these results.

## Read-only artifact verification

External files under
`C:/CodexArtifacts/pretrained-reuse-r2/synthetic-av-residual-gated-temporal-v4-attempt-01/`
were hashed in place:

| Artifact | SHA-256 |
|---|---|
| Source config `configs/pretrained_reuse/synthetic_av_residual_gated_temporal_v4.json` | `ed06d6d6220f2759a1bb978c3327adff8645fc043fe4e2971989b699aa90cb49` |
| Captured `effective-config.json` | `3095199c620983188f2b76a6ee0d3e86ff20ce2aab9ce9f80df29da0810c15c5` |
| `run.json` | `895a2696710b161462288f13316c8814e3606151a4893a9611ff2f6792344afc` |
| `residual_gated_temporal/history.jsonl` | `fbfa2b27bffb451465e22615092cf484f8ffca713b37121ce27cab4017fa75a9` |
| `residual_gated_temporal/best.pt` | `1cc33b123cea02c5e57f8cd27f69fe78d1ab459d18e3a228a8f3c6a504a0f811` |
| `residual_gated_temporal/terminal.pt` | `e862cdcd8e7b05b8cca1f065acbd536438e43d0503be89f361925fb31d435025` |
| Separate `evaluation.json` | `735816afda33eec2f6678e1ed3d52ce517aff5cc8e5f7ed32072f2a81108fee6` |
| Separate `predictions.jsonl` (71,680 rows) | `66ff5c760e556a29b05ea366f7f4392bb48e6505a7aee61076949d2b0d2e2d83` |

The evaluation receipt's `prediction_sha256` equals the independently computed
SHA-256 of the saved predictions file. The residual-gate report on the source
branch instead lists `fd4af1ef5c481134367dac89236da6d448a0ba89281cdae8eaa0676374b4039a`.
That value matches the earlier bypass-conditioned run's 57,344-row predictions,
not this 71,680-row residual-gate evaluation. This is a report hash-copy error;
the saved residual evaluation JSON and prediction file agree with each other.
The source worktree/report was not changed here.

The source report records a local RTX 3080 Laptop run at 120.70 seconds, with
508,658,176 allocated and 562,036,736 reserved peak bytes, and no cloud cost.
Those are reported run measurements; this reconciliation did not repeat them.

## Next gate

Keep the synthetic gate result diagnostic-only and do not repeat fitting on its
exposed IDs. Continue R2-M source/task admission: a naturally recorded,
rights-cleared task, grouped independently by source, must establish that the
target depends on multiple streams before a real-data fusion fit. The current
VisTouch Git checkout lacks media; RoboMNIST has unresolved common-clock and
payload/session gates; prior STARSS23 fusion did not beat audio-only. These
limits do not close the broader R2-M task matrix or the project objective.

Primary source report on the independent branch:
[`R2_SYNTHETIC_AV_RESIDUAL_GATE_RESULTS_20261026.md`](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-synthetic-av-gated-temporal-20261026/docs/pretrained_reuse/R2_SYNTHETIC_AV_RESIDUAL_GATE_RESULTS_20261026.md).
