# Shared nonlinear cell readout protocol

Existing bilinear spatial readout improves counting over pooled features, but
physical mirror augmentation does not improve NLL or counting. Test a different
readout inductive bias: shared GELU(384->128) per cell, sum/4, plus GELU CLS384->128,
concatenate->128 and dot with candidate384->128 /sqrt128.180736 trainable
parameters;39,627,136 total. This jointly changes nonlinearity, weight sharing,
and parameter count; do not attribute any difference to just one of those.

Frozen original DINO/MiniLM features only, no mirror augmentation. Same8424/512
images,33696/2048 questions, all options,seed17,24epochs,AdamW1e-3/wd1e-4,batch64,
unchanged earliest min validation NLL selection. No extra seeds or extension.
Candidate permutation equivariance, cell permutation invariance and gradient
finiteness tested on CPU, then separate smoke. This is not object detection:
contextual cells already contain position information; no explicit pair relations.

Compare selected checkpoint with existing original spatial bilinear90.04% and
small bilinear89.26%, same IDs via paired512-image bootstrap10000seed17. Record
all metrics/curves, train gaps, hashes, best/final reload, time/VRAM. No audit,
calibration, ensemble, target simplification or parameter freezing changes.
