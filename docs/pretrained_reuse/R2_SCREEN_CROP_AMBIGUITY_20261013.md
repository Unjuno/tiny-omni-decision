# Screen crop diagnosis

Read existing records/features only:1017 training regions (1012 exact crops),
723 validation regions (723 exact crops). No exact crop hash has conflicting labels
within either split. The resulting empirical bound of1.0 is vacuous: absence of
exact collisions does NOT prove label consistency, context sufficiency or achievable
100% quality. Near-duplicates and semantic ambiguity remain unmeasured.

A fixed raw-feature cosine1-nearest-training-neighbor diagnostic gives34.58%
Accuracy/23.47% balanced Accuracy. No k/temperature/feature search. Original linear
head was33.47%/26.24%; training-prior Accuracy48.69%. Thus bypassing linear fitting
with local retrieval does not yield a useful accepted screen classifier. No
probability calibration or NLL is claimed for this hard-neighbor diagnostic.

All examples retained. Output JSON pins record/feature hashes and all selected
training indices/predictions. This is observed development, oracle localization,
and four validation apps, not general screen decisions or blind evidence. Do not
restart the rejected linear head or infer a universal vision-model ceiling.
Next representation work would need whole-screen/local-region context controls
and a task contract; this result alone does not justify more training capacity.
Reproduce: `python -m scripts.audit_screen_crop_ambiguity`. No GPU/model/data changes.
