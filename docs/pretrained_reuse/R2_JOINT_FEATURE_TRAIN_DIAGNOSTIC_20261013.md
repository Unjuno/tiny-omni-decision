# Train-only joint feature / optimization diagnostic

Follow-up to the completed fixed50-update experiment. Its results and selected
checkpoints remain unchanged. No validation features, metrics or labels are used
by this diagnostic; no hyperparameter sweep or product promotion.

## Feature geometry

Subtract each question's mean across candidate features (common score shifts
cancel in softmax). Training normal features have candidate-contrast RMS0.024484,
and beginning/end contrast-change RMS0.002696. Black features:0.052516/0.007642.
The normal contrast matrix has numerical rank192 versus black12, at a relative
singular-value threshold1e-6. Nonzero rank/change does not prove useful semantics.

Final fixed-run CE gradients remain nonzero: normal L2 norm0.035521, black0.139196.
Thus the50-update run cannot certify the optimum of the linear training problem,
and its failure cannot establish that the features contain no usable information.
Geometry artifact: joint-readout-01/train-feature-geometry.json (train only).

## One fixed convex optimization diagnostic

Fit each192-vector from zero on the same198 train questions only, CPU float64,
LBFGS/strong-Wolfe, max200 iterations/250 evaluations, L2 coefficient1e-4,
gradient tolerance1e-7, change tolerance1e-9. Objective is mean candidate CE plus
0.5*1e-4*weight squared norm. No feature rescaling, no model changes, no test data.

| Input | Train Accuracy | CE | Regularized objective | Both correct | Changed-target both correct |
|---|---:|---:|---:|---:|---:|
| Normal |0.525253|0.985284|1.013490|0.313131|1/53|
| Black |0.449495|1.060426|1.064526|0.242424|0/53|

Normal39 iterations/41 closure calls,1.129s; black30/31,0.025s. Maximum absolute
final objective gradients1.09e-6/2.66e-7; the optimizer stopped before its iteration
cap, but the gradient tolerance is not met. Do not call this an exact optimum.
Weight norms23.75/9.06; changed initialization, optimizer and regularization make
this an optimization diagnostic, not a one-variable quality comparison with the
prior AdamW run. No validation performance is inferred from the training gain.

The train-only result shows that further optimization can change training fit.
It still fails nearly every changed-target pair. Retain the prior rejection of
the fixed linear-readout candidate; do not claim a global model-family ceiling.
Do not automatically evaluate this diagnostic weight on the observed validation
or extend a search around it. A next quality experiment needs a specific hypothesis
about temporal evidence retention and condition-dependent object identity.

Artifacts: C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-joint-train-convergence-01.
Config, full closure histories, final weights and measured results are saved.
Reproduction script: scripts/diagnose_joint_train_convergence.py; use module
execution, and preserve its refusal to overwrite existing outputs.
Original executed script remains in the all-in-one-source-initial-01 artifact
folder as train_convergence_diagnostic.py. Only training corpus/features are read.
Executed script SHA256:008f8a4b34a033d36ba8251dae06ed1440ce5545d30afec79afb41edd51c980c.
The repository copy adds formatting, bound closure arguments and a main guard;
it preserves the fixed diagnostic computation and refuses existing output paths.
The first runpy CLI invocation failed before execution; runpy.run_path succeeded.
