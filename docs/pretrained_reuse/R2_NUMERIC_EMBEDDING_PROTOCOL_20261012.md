# Fixed512-step numeric embedding comparison

Fresh optimizer from identical encoder-coverage selected weights (same as prior
comparison-vocabulary run), seed17; same original/curriculum mix, all6 blocks+
classifier trainable. Only addition: zero-init343x384 selected embedding delta,
IDs derived from training numeric spans. Base embedding stays frozen. No new data.

512updates,8questions/update,AdamW2e-5/wd.01,clip1,dropout off, validation every128,
earliest minimum mean original/curriculum NLL including0. Enforce identical4096
question prefix against parent2048-run, identical validation IDs and initial metrics.
Compare against that parent's fixed512 and best within0..512, not its2048 best.
Selected/full exposure reported separately; inherited training not zero-cost.

Separate2-step smoke first; preserve best/final, frozen hash and exact reload.
Track all task metrics/curves, observed numeric-row gradients, parameter count,
time/VRAM. Total22,845,313; trainable10,778,881. No automatic2048 extension,
transfer evaluation, calibration or audit. If mixed/negative, retain real results.
Rolling train CE is not a fixed-set generalization gap.
