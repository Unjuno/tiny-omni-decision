# Fixed affine endpoint validation

The training-only fit is complete. Freeze its final 200-iteration checkpoint
`817d2d054b1b8998da2687fd2d0b64396fa5d7c7d23d0d66d73bde130fc0b015` before
evaluation. Compare its initialized affine checkpoint and the existing original
cropped global predictions on the same 512 development images / 2,048 questions.
No new fitting, regularization sweep, calibration, checkpoint selection, or
training extension. This follow-up evaluates the already frozen endpoint; it
does not turn the preceding training-only experiment into validation tuning.

CPU, two threads, float64; identical cached cropped features, all four taxonomies,
all ten options per task, original order. Require initial probabilities to match
saved original outputs within 1e-5, with no answer changes. Fail on input hash,
identity/order, image-content split or finite-output disagreement. Preserve all
source artifacts; create a separate output directory.

Record all predictions, per-task and aggregate Accuracy/NLL/Brier/ECE and mean
confidence, fixed full-training versus validation gaps, and second-minus-first
image-cluster paired bootstrap differences for Accuracy/NLL/Brier (10,000 draws,
seed 17, linear percentile 95% intervals). ECE is descriptive, no bootstrap claim.
Primary diagnostic: do counting Accuracy and NLL both improve? Call evidence
clear only if their paired intervals exclude zero in favorable directions.
Report all other task changes and any >2 percentage-point Accuracy loss.
Mixed evidence or zero-crossing intervals remains uncertain; no cherry-picking.

This is reused development validation with historical checkpoint selection and
multiple prior experiments. Intervals do not correct that adaptive reuse or
multiple comparisons and cannot establish blind generalization. The 1,595-image
reporting set and sealed audit remain unopened. No product promotion follows.
The next scientific step depends on the observed generalization gap, not simply
training accuracy; any further fit needs a separately fixed protocol.
