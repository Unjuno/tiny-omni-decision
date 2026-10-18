# E0G1 saved-output AV correction ablation quality — 2026-10-18

## Scope and provenance

This post-processing compares natural-input logits already saved by the
inference-only E0G1 replacement run. It did not start new model inference,
training, checkpoint selection, calibration, or sealed-audit access. The source
run had already scored the fixed development-validation inventory with the
selected E0G1 checkpoint. The separately named post-processing output is
`C:\CodexArtifacts\pretrained-reuse-r2\greatest-hits-e0g1-ablation-metrics-09`;
it does not overwrite any earlier attempt.

- Arm: E0G1, selected step 24,576; cross-modal exchange disabled, conditional
  gates enabled, candidate-output normalization enabled.
- Checkpoint SHA-256: `9df53a55e610f4e473dc5f56528edd86610a72683c8f6c40145e9d87d899d87d`.
- Inference source: attempt 15, source commit
  `e1a40d1087667c234270d10c242c636c96b3f80f`.
- Inference receipt SHA-256:
  `58f53e48cd83111d5d6faf8c64a54e5041a21c68742b2183a0d4a5e44c57b6e9`.
- Saved predictions SHA-256:
  `eea8d109d90032800b198cca87b3dd9dfd0206f02d42115153c0799027ad016e`.
- The inference receipt records 44,375 forwards, 17,750 exact self-replacement
  checks, zero optimizer/encoder updates, and no calibration or sealed-audit
  loading. The input digest is
  `00aa90bed282446e77ae834afc4d9b1d1c1adeedcbaf73e47a49f084c1838868`.
- Evaluation: 8,875 previously observed development-validation questions,
  grouped in 409 parent recordings. These results are post-selection
  development estimates, not blind generalization.

The saved logits provide three same-input conditions:

1. **Baseline**: original native candidate readout.
2. **Query-only**: baseline plus the added correction evaluated with zeroed AV
   residual features.
3. **Full residual**: baseline plus the trained query-conditioned AV residual.

The baseline itself consumes the audio and video features. Thus baseline vs
query-only is not a no-modality control. Full residual vs query-only isolates
the additional conditional AV correction in this implementation.

## Natural-input validation metrics

| Condition | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Baseline | 0.68146 | 0.99125 | 0.45472 | 0.10170 |
| Query-only | 0.67076 | 0.98929 | 0.45930 | 0.10564 |
| Full residual | 0.67076 | 0.98929 | 0.45930 | 0.10564 |

The full-residual and query-only Accuracy are identical across all 8,875
questions. Their NLL and Brier differ only at floating-point noise scale
(absolute differences below `2e-8`). The conditional AV correction therefore
made no measurable contribution to these natural-input decisions for this
checkpoint. This agrees with the learned gate values collapsing near zero in
the training report.

Relative to baseline, query-only loses 1.07 percentage points of
example-weighted Accuracy, improves NLL by 0.00196, worsens Brier by 0.00458,
and worsens ECE by 0.00394. The paired parent-recording bootstrap for
query-only minus baseline Accuracy is `[-0.01736, -0.00398]` (10,000 resamples,
seed 17, percentile 95% interval). For NLL the interval is `[-0.01980,
0.01629]`; for Brier it is `[-0.00178, 0.01086]`. These are unadjusted,
post-selection validation diagnostics.

Macro-over-task Accuracy is 0.68142 for baseline and 0.67054 for both corrected
conditions. Per-task Accuracy / NLL:

| Task | Baseline | Query-only | Full residual |
|---|---:|---:|---:|
| Action | 0.78373 / 0.50566 | 0.78306 / 0.50715 | 0.78306 / 0.50715 |
| Material | 0.57186 / 1.53333 | 0.57253 / 1.50565 | 0.57253 / 1.50565 |
| Reaction | 0.68866 / 0.93494 | 0.65601 / 0.95576 | 0.65601 / 0.95576 |

The largest task-level change is the 3.26-point Reaction Accuracy drop in the
corrected readout. Material NLL improves, but action and reaction NLL worsen.

## Interpretation and next step

This result does not support the claim that the learned AV residual improves
quality: enabling it yields effectively the query-only predictions. The
query-conditioned correction as a whole trades a small NLL gain for lower
Accuracy and worse Brier/ECE versus baseline, with a concentrated Reaction
regression. The validation data were used during model selection and should not
be reused to claim blind quality.

Do not spend more compute replaying these same logits. Any follow-up should
change one justified factor and use a fresh, task-defined evaluation set, with
input replacement labels validated before interpreting counterfactual quality.

## Reproducibility artifacts

- `report.json` and `analysis.py` are in
  `C:\CodexArtifacts\pretrained-reuse-r2\greatest-hits-e0g1-ablation-metrics-09`.
- Report SHA-256: `64febcf09b48b689f51da41dd67f8e4e374436c28f2abff415c9c7c03d322973`.
- Analysis script SHA-256: `f6d65568389badfa53fcb549b76983273d1ea5b38ae35ad6af777c30aaaf868a`.
- The script verifies the source receipt, saved predictions, selected
  checkpoint, training receipt, validation labels and exact question/options
  alignment before calculating metrics.
