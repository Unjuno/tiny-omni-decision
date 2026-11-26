# Fixed physical-mirror augmentation comparison

Motivation: physical reflection of64 training images reduced counting75% to60.94%
with the frozen spatial readout. Test actual augmentation, not contextual-cell swaps.

Extract8424 physically mirrored training images using pinned frozen DINO, same
whole-image224x224 preprocessing and FP32 spatial cells. Require exact first64
agreement with the completed diagnostic and hash every original asset. Existing
validation cache is unchanged. Mirrored views are not new independent images.

Next controlled training: spatial bilinear hidden128,884992 trainable parameters,
seed17,AdamW1e-3/wd1e-4,batch64,24epochs,all33696 questions each epoch. Compare
original-only against augmentation with identical init/order/update counts. Use
one per-image/per-epoch seed17 Bernoulli0.5 mirror decision shared across all four
questions; independent RNG must not alter question shuffling. Record realized
view counts. Original-only must reproduce the saved24-epoch spatial history.

Earliest minimum unchanged validation NLL selects best; preserve final separately.
No extra validation images, calibration, inference averaging or audit. Report all
four image-task metrics, train/validation curves, paired512-image bootstrap and
cost. Both remain narrow image candidates, not general or four-modality success.
Run a separate smoke before full training; no automatic budget extension.
