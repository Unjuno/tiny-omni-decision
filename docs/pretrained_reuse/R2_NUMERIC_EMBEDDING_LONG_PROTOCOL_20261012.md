# Numeric embedding fixed2048-step learning curve

This is a NEW fresh optimizer run from the same encoder-coverage checkpoint,
not a resume of512. All original settings/data/seed17/343selected numeric rows,
AdamW2e-5/wd.01,8questions/update,eval128,selector unchanged. Only max budget2048
and output path differ. Freeze this plan before launch; no additional seed/LR search.

Require all saved0/128/256/384/512 metrics, accounting and gradient fields to match
prior512 run exactly, ignoring ONLY elapsed/evaluation seconds. Hash that history.
Fail on mismatch; never loosen this gate to continue. Save best/final separately.

Compare to the completed no-delta2048 run at512/1024/1536/1920/2048 and each
predeclared validation-selected checkpoint. Include per-source/task metrics,
consumed/unique data, gradient norms, VRAM/time and paired state uncertainty.
No claim of improved model if selector prefers initialization. No audit/transfer
retuning, calibration or product promotion. No automatic extension beyond2048.

Available space must be>=1GiB before launch; existing artifacts remain untouched.
The paired512 analysis is mixed, and earlier baseline improved after512; the
purpose is to distinguish budget insufficiency from an ineffective change,
not to obtain a positive result. Rolling training CE is not a fixed-set gap.
