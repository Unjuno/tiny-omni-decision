# Training-only affine fit (2026-10-12)

## Scope and execution

One bounded CPU diagnostic completed: 8,424 original training images, all 33,696
color/texture/count/shape questions, equal task weight, unchanged cropped frozen
features and fixed 40 candidates. No validation or sealed-audit data were loaded.
The original selected bilinear checkpoint initializes the affine weights; this
is a training diagnostic warm start, not a new independent quality experiment.

Run: `python -m scripts.fit_shared_affine_training --output C:\CodexArtifacts\pretrained-reuse-r2\shared-affine-training-fit-01`.
A separate `--smoke` process used eight images and two inner iterations first.
The full run used float64, two CPU threads, unregularized full-batch L-BFGS,
200 inner iterations / 229 closure evaluations, no extra seeds or extension.
Elapsed 22.178s including identity check; optimization/history
after initialization took 8.557s.
No GPU compute or cloud cost. Peak process RAM was not measured.

## Training results only

| Task | Initial Accuracy | Final Accuracy | Initial NLL | Final NLL | Final Brier | Final ECE |
|---|---:|---:|---:|---:|---:|---:|
| overall | 91.4292% | 94.4623% | 0.198187 | 0.146408 | 0.080128 | 0.020295 |
| color | 100.0000% | 100.0000% | 0.001975 | 0.001651 | 0.000127 | 0.001614 |
| texture | 97.6971% | 99.6795% | 0.063313 | 0.020115 | 0.007838 | 0.012387 |
| count | 68.0199% | 78.1695% | 0.726062 | 0.563198 | 0.312496 | 0.066527 |
| shape | 100.0000% | 100.0000% | 0.001398 | 0.000668 | 0.000052 | 0.000653 |

| Inner iterations | Training CE | Count Accuracy | Gradient infinity norm |
|---:|---:|---:|---:|
| 0 | 0.198187 | 68.0199% | 0.16961416 |
| 50 | 0.168509 | 73.9672% | 0.00129392 |
| 100 | 0.159794 | 75.8547% | 0.00503729 |
| 150 | 0.152521 | 76.8637% | 0.00963262 |
| 200 | 0.146408 | 78.1695% | 0.00475465 |

## Interpretation and limits

The same frozen representation supports a lower training error. Existing weak
counting therefore cannot be attributed solely to missing encoder information.
Optimizer, parameterization, precision and regularization all changed; this is
not a pure learning-budget comparison. Projected candidates have full row rank
40, so unrestricted affine scores are representable in the original fixed-candidate
bilinear family. This statement does not extend to unseen semantic candidates.

Loss was still decreasing and the final gradient infinity norm is 0.004755,
above 1e-7. Budget completion is not convergence or a representation ceiling.
Overall training ECE worsened 0.005212 -> 0.020295; count ECE worsened
0.017059 -> 0.066527. Increased training accuracy is not calibrated quality.
No generalization gain, absence of overfitting, or Omni 95–98% success is claimed.
Keep the original runtime reference. Do not promote this closed-taxonomy artifact.
Any follow-up must predefine a separate regularized validation protocol; the
completed diagnostic does not authorize an unbounded training-budget extension.

## Integrity and artifacts

Initial all-question logit difference: 1.28e-13, within 1e-10.
61,480 trainable affine parameters; exact best reload; best and final predictions
are identical. Original input hashes remain unchanged. Initialized/best/final
weights and logits, every closure loss/gradient, and full curves are preserved.
Artifacts: `C:\CodexArtifacts\pretrained-reuse-r2\shared-affine-training-fit-01`. Config, source/input/artifact hashes and all metrics are
in the adjacent JSON report. Run source was e077068 plus the archived uncommitted
implementation; the runner SHA and executed-source archive identify it exactly.
The files contain no optimizer state and are not an exact-resume package.

## Verification

Ruff passed; 582 CPU tests passed, 1 skipped. Model/dataset/catalog validation
and the manifest audit passed. Target-commit GitHub Actions is checked after push.
