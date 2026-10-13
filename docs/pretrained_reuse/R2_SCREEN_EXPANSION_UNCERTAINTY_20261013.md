# Expanded-screen app dependence

Saved selected predictions, unchanged723validation rows. Paired4-app bootstrap,
10000replicates,seed17,percentile95%,sample-weighted,unadjusted. Very few clusters
and reused checkpoint-selection data: intervals are descriptive, not confirmation.

Accuracy delta+.07884, interval[-.02299,.17566]; NLL delta-.16620,
interval[-.47622,.20708]. Both span zero.

| App | Old Accuracy | Expanded Accuracy | Old NLL | Expanded NLL |
|---|---|---|---|---|
| KDevelop |.4746|.4322|1.5490|1.5729|
| Matrix |.3147|.3077|2.2307|2.1329|
| QGIS |.2363|.4418|2.3971|1.8535|
| VLC Media Player |.4235|.4412|1.8747|2.1673|

The aggregate gain is not uniform across apps. Expanded training-frequency prior
NLL1.6955 is better than expanded model1.9368 (both Accuracy prior .4869 versus
model .4136). Therefore keep the failed-quality decision. Do not justify a broad
scaling run solely by the pooled7.9point increase. New app-independent evidence
and a suitable semantic representation are still required; no audit opened.

Reproduce: `python -m scripts.compare_screen_expansion`; source prediction hashes
and per-app complete metrics in adjacent JSON. No retraining or selection changes.
Full CPU suite730 passed/1 skipped; repository Ruff passed before this diagnostic,
and diagnostic Ruff passes. Existing weights/inputs remain unchanged.
