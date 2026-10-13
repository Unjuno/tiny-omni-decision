# Screen context distribution audit

Read-only saved-feature and checkpoint inspection; no fitting, normalization
change, model selection or new inference dataset. Source hashes in adjacent JSON.

| Normalized feature block | Train RMS | Validation RMS | Validation abs p99 | Validation fraction abs>5 |
|---|---|---|---|---|
| crop |1.000|1.052|2.812|.000092|
| full screen |1.000|1.740|5.938|.016639|

Final checkpoint class-centered logit RMS (train/validation): crop2.450/2.385,
full-screen3.068/3.743. Initial full-screen values .430/.641. These are additive
linear contributions before bias; not Accuracy, causal importance or information
content. Feature weighting is per admitted region, matching the actual training
normalization, not uniform per unique screen. Repeated screen features do not
create independent observations. Four held-out apps remain a major limitation.

The measured full-screen distribution shift is larger than the crop shift and
can contribute to unstable context scores. It does not prove the only cause,
or prove a normalization fix would restore quality. Do not tune clipping,
normalization or feature weights on these validation results. Preserve the failed
append experiment. Any new representation comparison needs a prior fixed protocol,
training-only normalization and stronger app/scene diversity; no new head/capacity
increase is justified merely by these diagnostics.

Reproduce: `python -m scripts.audit_screen_context_shift`. Ruff passed.
This closes the numerical-input check; screen semantic quality remains unmet.
