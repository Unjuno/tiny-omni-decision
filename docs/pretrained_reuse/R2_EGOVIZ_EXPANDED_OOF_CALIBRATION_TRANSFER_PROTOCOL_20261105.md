# EgoViz expanded-train calibration transfer — fixed protocol (2026-11-05)

## Question and limits

Does a single positive temperature fitted only from group-held-out logits on
the existing expanded 48-session training corpus improve the stored linear
step-616 scorer's probabilities on the already-selected 69-session
generalization set?

This is a **post-hoc fixed-scalar analysis of an exposed development set**.
The unscaled 69-session metrics have already been reported. The analysis is not
blind, does not create a new independent test, does not select a checkpoint,
does not alter weights, and must not be used to tune another candidate. It is a
limited check of whether a training-only calibration rule transfers to the
previously unused session groups.

The task remains four-choice video-to-IMU timestamp correspondence with
automatically generated upstream labels. `group_id` is a capture session;
participant independence across sessions is unknown. This is not a natural
world action task or product quality evidence. The sealed audit remains
unopened.

## Frozen inputs and rule

- Training corpus: the existing merged **773 train examples / 48 sessions**
  from `egoviz-train-coverage-merged-20261030-attempt-01`; reuse all input
  files by read-only reference and verify each SHA-256 against `merge-report.json`.
- Scorer target: the existing 81,920-parameter linear `best.pt` selected at
  update 616 by the old development selector. Verify its SHA-256 against the
  expanded training `results.json` and the 69-session evaluation's
  `linear_selected_step_616` model record.
- Calibration folds: seed 17, five deterministic group folds from the 48
  training sessions. For each fold, hold one group partition out for OOF
  scoring; use the next partition as inner validation for the existing
  `JointTemporalScorer` fit/early-stop routine; fit on the other three
  partitions. Each train example receives exactly one score from a model that
  neither fit nor checkpoint-selected on its session.
- Each fold uses the existing scorer, AdamW LR `0.003`, weight decay `0.01`,
  batch size `32`, maximum `100` epochs, patience `12`, `min_delta=1e-6`, and
  seed `17`. Fit one positive inverse temperature with the existing bounded
  convex NLL optimizer over all training OOF logits.
- Only after the scalar is fixed, read the stored 69-session prediction file.
  Apply the scalar once to `linear_selected_step_616` logits. Do not change the
  checkpoint, refit, inspect the sealed audit, or use the resulting metrics to
  choose a model or another temperature.
- Report Accuracy, NLL, Brier, the same 10-bin ECE used by the stored baseline,
  and mean confidence for raw and scaled
  probabilities, paired session-cluster bootstrap differences (2,000
  replicates, seed 17), all OOF logits and transformed per-example
  predictions. Positive temperature must preserve every option ranking.
- Save all fold states and an independent-process reload receipt. Outputs go
  to a new external directory; existing data and checkpoints are read-only.

The primary comparison is descriptive transfer of a training-only scalar.
Bootstrap intervals quantify variability over the 69 stored session groups
conditional on the fitted scalar; they do not remove the prior exposure of this
development set or account for participant overlap, checkpoint selection on
older development data, or the fact that the analysis follows the unscaled
result.

## Execution constraints

CPU only; no media acquisition, encoder inference, GPU, paid compute,
checkpoint-selection on the 69 groups, or sealed-audit access. The additional
fold checkpoints and JSON outputs are expected to be under 3 MB total.
