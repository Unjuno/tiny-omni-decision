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
