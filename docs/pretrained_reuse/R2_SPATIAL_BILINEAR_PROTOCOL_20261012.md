# Spatial information × candidate interaction control

Current DINO/bilinear89.26% is limited mainly by exact counting. The prior larger
spatial MLP did not improve the smaller head; do not repeat that experiment.
Test whether the successful candidate-interaction form benefits from retained
spatial cells using a matched-capacity control.

Use the existing pinned `[N,17,384]` FP32 cache (CLS plus16 spatial cells),
unchanged extraction/preprocessing/data. Spatial condition flattens16 cells/4
plus CLS; global control replaces all cells by their image mean before the same
flattening. Both observation inputs6528, candidate384, bilinear hidden128,
884992 trainable parameters and identical initial weights. Total40,331,392.
This preserves prior feature scaling and does not claim equal effective degrees
of freedom: global inputs repeat information and have lower rank.

Fresh seed17,24epochs,AdamW1e-3/wd1e-4,batch64, same shuffled order; all8424/512
images and four10-option tasks. No encoder training, new data, exact-count
tolerance changes or sealed audit. Cached global means must retain prior parity.
Run separate small execution smoke first; no previous artifact overwritten.

Select earliest minimum validation NLL independently; save best/final and require
exact reload. Record histories, train/validation gap, per-taxonomy Acc/NLL/Brier/ECE,
rotated-image control, same init/order hashes, parameters/time/VRAM. Compare spatial
against global and stronger small bilinear23-selected reference using paired512-image
bootstrap10000/seed17. The earlier8-epoch MLP history is provenance only, not a
required prefix for a different readout. No automatic further budget extension.
