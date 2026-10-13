# Replay retention: saved-prediction diagnosis

No training, model loading, calibration fitting or audit evaluation was performed.
All 2,540 original-development IDs, targets and option orders match across steps.
Question type and underlying state identity were recovered from the pinned original
validation records; the runner's source-level `group` field is NOT a state cluster.
The diagnostic uses 732 actual states for paired resampling.

| Type | Count | Parent Accuracy | Replay Accuracy | Parent NLL | Replay NLL |
|---|---|---|---|---|---|
| choice | 1008 | .607143 | .600198 | .964864 | 1.270887 |
| noul / binary | 987 | .740628 | .733536 | .519355 | .732429 |
| score | 545 | .499083 | .473394 | 1.169201 | 1.563388 |

237 previously correct answers became wrong; 209 previously wrong became correct.
Mean confidence rose .642509 -> .784211; mean confidence among wrong answers rose
.539081 -> .703733. These wrong-answer subsets differ, so this is descriptive,
not a within-error causal estimate. All three types show worse NLL/Brier/ECE.

State-paired bootstrap: 10,000 repetitions, NumPy default_rng seed17, resampling
whole states and retaining sample-weighted metrics; percentile95% unadjusted CI.
Accuracy delta -0.011024, CI[-0.027854, 0.005894]; NLL delta +0.288822,
CI[0.239039, 0.339881]. Accuracy regression is not resolved by this interval;
calibration deterioration is clearer. This is reused development evidence with
post-hoc strata, not a blind generalization or causal claim.

Reproduce with `python -m scripts.diagnose_context_replay_retention`.
Adjacent JSON records source/prediction SHA-256, all scheduled metrics and intervals.
No checkpoint is selected retrospectively. Keep the coverage parent as reference.
Before another fit, distinguish loss/target sharpness from representation changes:
inspect the original training objective and stored soft teacher targets. Do not
fit a temperature on these development results or silently replace hard labels.
The result does not establish a model-family capacity ceiling.
