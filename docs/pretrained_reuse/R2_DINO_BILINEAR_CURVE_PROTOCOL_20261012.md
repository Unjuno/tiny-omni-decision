# DINO bilinear fixed learning curve

The39.59M DINO/bilinear candidate remains stronger and smaller than the135.13M
ViLT candidate. Its original8-epoch endpoint was selected. Measure24epochs
without changing features, data, architecture or optimizer.

Fresh seed17 initialization; same pinned CLS384+patch-mean384 features and
MiniLM candidate384 features,147712-parameter bilinear head,8,424 train images,
512 reused validation images,all four10-option taxonomies. AdamW1e-3,wd1e-4,
batch64 and identical Python shuffle sequence. No new data or encoder tuning.

Require the original8-epoch history SHA256 and exact first8 history records
before continuing past8. Separate execution smoke and fresh output directory.
Stop at24, preserve earliest minimum validation NLL best and final separately,
exact best reload, all predictions and rotated-image control. Do not default
to final checkpoint. Report actual encounters versus unique images/questions.

Compare selected24-budget checkpoint with original8-budget result using paired
512-image bootstrap10000 draws/seed17, with selection-bias caveat. Save complete
curve and per-taxonomy metrics. Existing default8-epoch behavior is unchanged.
No sealed audit, full-model training, compression or automatic further extension.
