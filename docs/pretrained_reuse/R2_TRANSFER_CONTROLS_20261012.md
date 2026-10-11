# Predeclared fixed-weight transfer controls

Use the selected encoder-coverage step2048 exactly once on seed43,128 previously
unused numeric states and eight questions per state. No training or checkpoint
selection. Seven fixed variants: unchanged generator control; labels renamed
A/B/C/D; identifier key id renamed candidate; question wording paraphrased;
all numeric values and thresholds transformed by5*x+10; cost renamed price;
and all five changes combined. Only one factor changes in each individual arm.
Row/key and option order stay fixed; correct option positions stay identical.
All conditions preserve state membership and questions; no difficult case removed.
Cardinality remains four items and four/five options: candidate-count effects
are not identified by this experiment. Scalar time questions are not video tasks.

Reject overlap with the previous train/validation numeric state identities.
Verify every transformed answer independently before rendering. Compare Accuracy,
NLL/Brier/ECE overall and per task; paired state bootstrap differences versus
control,10,000 resamples, seed20261012,95% percentile intervals. All comparisons
are descriptive development diagnostics, unadjusted for multiple comparisons.
No prompt search or retraining against these outcomes. Do not select the best
rendering for a product claim; use failures to design broader predeclared data
coverage. No sealed audit is opened. Local inference only, new artifact directory.

## Completed measurement

Execution PASS; sourcee8f6a3f. Pinned selected step2048, no training or selection.
All7,168 predictions saved with IDs/options/targets/probabilities. Original data
and weights retained. This is a controlled follow-up to an observed failure,
not an independent blind product evaluation.

| Variant | Accuracy | NLL | Brier | ECE15 | Difference vs control (pp) | Paired95% interval (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| control | 0.79785 | 0.47846 | 0.26417 | 0.02058 | 0 | reference |
| labels | 0.76465 | 0.56100 | 0.30758 | 0.02742 | -3.32 | -5.18, -1.56 |
| identifier_key | 0.79004 | 0.48472 | 0.26694 | 0.02877 | -0.78 | -1.76, 0.20 |
| wording | 0.54395 | 1.00107 | 0.54180 | 0.05511 | -25.39 | -28.12, -22.66 |
| magnitude | 0.25879 | 2.19912 | 0.95122 | 0.33995 | -53.91 | -57.42, -50.29 |
| field_name | 0.79980 | 0.47848 | 0.26222 | 0.02402 | 0.20 | -0.78, 1.07 |
| combined | 0.22949 | 2.00261 | 0.96515 | 0.30543 | -56.84 | -60.35, -53.22 |

## Interpretation

Under these exact perturbations, the largest loss comes from magnitude: positive
integers1..199 become15..1005 through5*x+10, with thresholds transformed the same
way. This preserves all extrema, order, counts and eligible-set answers. Question
wording is another substantial weakness. Labels have a smaller negative effect;
renaming the identifier key or cost field does not explain the overall collapse.
The combined arm changes all five factors, so it is not an estimate of any single
factor and interactions remain possible. Candidate-count effects remain untested.

This supports a specific data/generalization bottleneck; it does not prove that
all failures arise from data rather than representation, capacity or optimization.
The positive same-generator result is repeatable on new seed43 states, while
range extrapolation is poor and overconfident. Do not select the best wording or
rename fields to inflate an evaluation score. Broader numeric/lexical training
coverage and preregistered held-out range/template tests are more justified than
another unchanged epoch or a larger model selected from these results.

No95% or general four-modality claim. Count and conditional-selection remain in
every arm; nothing was dropped. Scalar-time-order remains a numeric text task.
All per-task metrics and exact files/hashes are in the adjacent JSON.

## Runtime and verification

Local FP32 GPU inference: 29.980s; peak PyTorch allocation 192,056,320bytes; max input 157tokens, no truncation. No cloud cost.
State-group bootstrap uses128states, all eight questions grouped,10,000 draws,
seed20261012,95% percentile intervals without multiplicity adjustment. Group
overlap with prior curriculum train/validation is zero. CPU tests329 passed,
1 skipped; Ruff passed. No sealed audit read or legacy artifacts overwritten.
