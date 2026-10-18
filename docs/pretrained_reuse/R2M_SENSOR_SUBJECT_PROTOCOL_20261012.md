# Frozen subject-disjoint sensor comparison

Use the previously hashed UCI HARTH v1.2 archive and seed17 subject split.
Read all14 training and3 validation subjects only. Never open calibration or
sealed-audit subject contents. Preserve all source files and prior artifacts.

Preprocessing: six acceleration channels in g, 128 samples per window, stride64,
break at gaps >30ms, no interpolation. Retain mixed-label windows; target is the
original center sample at index64. Record full label sequences, subject identity,
source intervals and mixed-window flags. Report all12 predefined activity codes,
including support0; no rare-class removal or label-based subsampling.

Compare the existing 6,700-parameter temporal convolution with an independent
static baseline (six-channel mean/std/min/max -> Linear24,64 -> GELU -> Linear64,12).
The static baseline is intentionally permutation invariant, with different
capacity; any difference is not an isolated parameter-matched causal estimate.
For the selected temporal model also evaluate a fixed seed17 within-window time
permutation, retaining timestamps and labels, as an order sensitivity diagnostic.
Time permutation can create unrealistic dynamics; report that limitation.

Both runs: fresh seed17, float32, AdamW lr0.001 weight_decay0.01, batch128,
12epochs, all training windows once/epoch, deterministic shuffled training order.
No class reweighting, augmentation, scheduler or early stopping. Evaluate epoch0
and each epoch. Select earliest minimum validation sample NLL; retain selected
and final separately. No hyperparameter search. Save per-example probabilities,
subject/target IDs, per-class recall and class-conditional NLL, Accuracy, balanced
Accuracy, NLL, Brier, ECE(10 equal-width bins), history, parameters, timings and
artifact hashes. Report subject-level variation and transition-window subsets.
Validation has only3 independent subjects; do not call overlapping windows
independent trials or treat this as sealed evaluation. Preserve absent-class
limitations. Re-evaluate selected weights after reload for exact consistency.

Run a separate small CPU smoke before full training. Use the local GPU only if
unoccupied; this modest model needs no cloud compute or dependency changes.
This activity benchmark establishes the sensor branch only; screen, multimodal
complementarity, arbitrary option semantics and general Omni quality stay open.
