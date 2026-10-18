# Training-only affine optimization diagnostic

Use only the original8,424 training image features and all33,696 labels across
color/texture/count/shape. No validation/reporting/audit data or metrics are
loaded. Preserve the existing selected cropped global checkpoint and all caches.

Translate its bilinear fixed40 candidate scores into an affine classifier,
using the separate spatial summary and cell mean (1,536 inputs). Initialize in
float64 from the original weights cast to float64. Require algebraic initial
logit equality atol/rtol1e-10 against the same original model in float64 on all
training questions. Check the projected40 candidate vectors' rank. If not full
row rank, record that a free affine fit may expand the original fixed-projection
function family; do not silently assume equivalence.

One CPU, two-thread run, deterministic starting weights; seed17 recorded, no
random search. Fit61,480 affine parameters using unregularized mean CE over
all4 equally weighted tasks. Full batch, L-BFGS lr1,history50,strong-Wolfe,
max_iter10/max_eval20 per call, at most20 calls (<=200 internal iterations).
tolerance_grad1e-7,tolerance_change1e-10. Stop early only if measured global
gradient infinity norm <=1e-7 or a numeric failure. Inner optimizer stops are
reported; budget completion alone is not convergence. No budget extension.

This changes optimizer, coefficient parameterization, precision and regularization
relative to the old AdamW run. It tests attainable training fit of the frozen
features, not a pure step-budget effect or generalization. No new nonlinear
features, data augmentation, preprocessing, encoder update or modality dropping.

Save initialized/best-training/final weights and logits, all closure losses and
iteration/evaluation counts, gradients, per-task training curves, final calibration,
rank/singular values, hashes, runtime and exact best reload. Best training loss
is a diagnostic record, not a product checkpoint-selection rule. Training accuracy
is never reported as validation/test quality. No validation evaluation or artifact
promotion follows automatically. Separate fresh small smoke before the full run.
