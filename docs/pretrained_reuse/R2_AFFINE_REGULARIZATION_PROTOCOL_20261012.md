# Bounded anchored regularization comparison

Two new CPU fits, coefficients 0.01 and 0.1, fixed before evaluation. Compare
against the saved coefficient-zero 200-iteration fit; do not rerun or extend it.
All arms start from the same original affine initialization, never the previous
optimized endpoint. Keep the same 8,424 training images, 33,696 questions, 1,536
features, four ten-option tasks, float64, two threads, full-batch L-BFGS settings
and maximum 200 internal iterations. No feature scaling or task reweighting.

Objective = mean CE + coefficient * sum((W-W_initial)^2) / (2*40).
Bias is not penalized. This explicitly controls departure from the existing
readout rather than weight decay towards zero. The initial objective equals CE.
Use lr1, history50, strong-Wolfe, 20 calls of max_iter10/max_eval20,
tolerance_grad1e-7, tolerance_change1e-10. Stop early only for measured gradient
infinity norm <=1e-7 or numerical failure. No convergence/ceiling claim at budget.

Save all closure losses, objective/CE/penalty, gradient, training metrics and
initial/best-objective/final weights. Evaluate final endpoints only, once each,
after both fits finish, on the existing fixed 512-image development validation.
No within-run validation selection, extra seeds, coefficient refinement,
calibration, reporting-set access or sealed audit. Separate eight-image smoke.

Report every arm and original reference: task and aggregate Accuracy/NLL/Brier/
ECE; full-train/validation gaps; paired image-cluster intervals against the
coefficient-zero endpoint (10,000 resamples, seed17, linear percentiles).
This is development hyperparameter comparison, not blind generalization.
No correction for repeated validation reuse or multiple comparisons is claimed.

Any research recommendation uses lowest aggregate validation NLL only among
arms with no task more than two absolute Accuracy points below the original
reference; ties prefer the larger coefficient. For counting specifically,
require favorable Accuracy and NLL intervals versus coefficient zero before
claiming clear improvement. Report mixed/uncertain outcomes. No product promotion.
If neither coefficient establishes a useful counting gain, stop this small
regularization family rather than launch an adaptive grid or budget extension.
