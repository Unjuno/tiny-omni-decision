# Anchored affine regularization comparison

## Decision

Both predeclared coefficients completed 200 inner L-BFGS iterations. Relative to
the fixed unregularized endpoint, both improve counting Accuracy and NLL with
favorable image-cluster intervals. The predefined lowest-NLL rule recommends
coefficient **0.01** for further research; no task loses more than two
Accuracy points versus the original reference. This is not product promotion.

All results use reused development validation. Repeated model development and
prior checkpoint selection limit interpretation; these intervals are not blind
evaluation, are not multiplicity-adjusted and do not prove broader generalization.

## Fixed conditions

Same original initialization, 8,424 training images / 33,696 questions, four
10-class tasks, cropped frozen 1,536-dimensional features, 61,480 affine
parameters, float64 CPU with two threads. Only the anchored weight penalty changes.
Objective: CE + lambda * sum((W-W_initial)^2)/(2*40); bias excluded.
L-BFGS lr1, history50, strong-Wolfe, max_iter10/max_eval20 per call, 20 calls.
No feature scaling, new seeds, task reweighting, extra budget or calibration.
Each final endpoint was evaluated once after both fits completed. Intermediate
validation metrics were never used. A separate 8-image / 2-iteration smoke passed.

## Validation metrics

| Arm | Task | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---|---:|---:|---:|---:|---:|
| original | overall | 90.7715% | 0.224677 | 0.124737 | 0.012183 | 0.906650 |
| original | color | 99.6094% | 0.006115 | 0.003003 | 0.004867 | 0.995680 |
| original | texture | 97.2656% | 0.102003 | 0.048089 | 0.015795 | 0.972318 |
| original | count | 66.4062% | 0.785480 | 0.445594 | 0.052186 | 0.661699 |
| original | shape | 99.8047% | 0.005112 | 0.002263 | 0.003388 | 0.996901 |
| zero | overall | 90.8691% | 0.208769 | 0.116869 | 0.017195 | 0.919199 |
| zero | color | 99.4141% | 0.008794 | 0.005655 | 0.005042 | 0.996176 |
| zero | texture | 98.2422% | 0.052940 | 0.028032 | 0.013049 | 0.978639 |
| zero | count | 65.8203% | 0.771299 | 0.433162 | 0.061061 | 0.703735 |
| zero | shape | 100.0000% | 0.002041 | 0.000628 | 0.001752 | 0.998248 |
| 0.01 | overall | 91.9434% | 0.196412 | 0.109488 | 0.013229 | 0.908513 |
| 0.01 | color | 99.6094% | 0.006784 | 0.003814 | 0.003660 | 0.995868 |
| 0.01 | texture | 97.8516% | 0.062440 | 0.032372 | 0.010670 | 0.977902 |
| 0.01 | count | 70.5078% | 0.712771 | 0.399981 | 0.054351 | 0.662175 |
| 0.01 | shape | 99.8047% | 0.003653 | 0.001785 | 0.002128 | 0.998108 |
| 0.1 | overall | 91.8457% | 0.198989 | 0.110959 | 0.017428 | 0.905953 |
| 0.1 | color | 99.6094% | 0.006672 | 0.003615 | 0.005098 | 0.995799 |
| 0.1 | texture | 97.4609% | 0.068261 | 0.035443 | 0.006308 | 0.977047 |
| 0.1 | count | 70.5078% | 0.716614 | 0.402705 | 0.066564 | 0.653253 |
| 0.1 | shape | 99.8047% | 0.004409 | 0.002072 | 0.002566 | 0.997713 |

## Paired differences versus coefficient zero

512 image groups; 10,000 resamples, seed17, linear percentile 95% intervals.
All task-level intervals are retained in the JSON report; primary counting and
aggregate differences are below. ECE is descriptive, without interval claims.

| Coefficient | Task | Metric | Difference | Lower | Upper |
|---|---|---|---:|---:|---:|
| 0.01 | overall | accuracy | 0.010742 | 0.002930 | 0.019043 |
| 0.01 | overall | nll | -0.012357 | -0.020128 | -0.004529 |
| 0.01 | overall | brier | -0.007381 | -0.012108 | -0.002737 |
| 0.01 | count | accuracy | 0.046875 | 0.017578 | 0.076172 |
| 0.01 | count | nll | -0.058528 | -0.088103 | -0.030129 |
| 0.01 | count | brier | -0.033181 | -0.050991 | -0.016091 |
| 0.1 | overall | accuracy | 0.009766 | 0.000977 | 0.018555 |
| 0.1 | overall | nll | -0.009780 | -0.019726 | 0.000195 |
| 0.1 | overall | brier | -0.005910 | -0.011772 | 0.000030 |
| 0.1 | count | accuracy | 0.046875 | 0.013672 | 0.082031 |
| 0.1 | count | nll | -0.054685 | -0.091811 | -0.018783 |
| 0.1 | count | brier | -0.030457 | -0.052969 | -0.008453 |

## Fixed full-train versus validation

| Coefficient | Task | Train Acc | Val Acc | Gap | Train NLL | Val NLL |
|---|---|---:|---:|---:|---:|---:|
| 0.01 | overall | 93.8331% | 91.9434% | 1.8897% | 0.161904 | 0.196412 |
| 0.01 | color | 100.0000% | 99.6094% | 0.3906% | 0.001653 | 0.006784 |
| 0.01 | texture | 99.4302% | 97.8516% | 1.5786% | 0.022012 | 0.062440 |
| 0.01 | count | 75.9022% | 70.5078% | 5.3944% | 0.623186 | 0.712771 |
| 0.01 | shape | 100.0000% | 99.8047% | 0.1953% | 0.000767 | 0.003653 |
| 0.1 | overall | 93.0110% | 91.8457% | 1.1653% | 0.172723 | 0.198989 |
| 0.1 | color | 100.0000% | 99.6094% | 0.3906% | 0.001788 | 0.006672 |
| 0.1 | texture | 99.0622% | 97.4609% | 1.6013% | 0.029024 | 0.068261 |
| 0.1 | count | 72.9820% | 70.5078% | 2.4741% | 0.659085 | 0.716614 |
| 0.1 | shape | 100.0000% | 99.8047% | 0.1953% | 0.000993 | 0.004409 |

## Training curves and stopping

| Coefficient | Iterations | CE | Penalty | Objective | Gradient max |
|---|---:|---:|---:|---:|---:|
| 0.01 | 0 | 0.1981869 | 0.0000000 | 0.1981869 | 0.16961416 |
| 0.01 | 50 | 0.1686985 | 0.0014887 | 0.1701872 | 0.00319843 |
| 0.01 | 100 | 0.1637965 | 0.0039120 | 0.1677085 | 0.00071011 |
| 0.01 | 150 | 0.1623857 | 0.0048012 | 0.1671869 | 0.00086162 |
| 0.01 | 200 | 0.1619044 | 0.0051486 | 0.1670530 | 0.00138097 |
| 0.1 | 0 | 0.1981869 | 0.0000000 | 0.1981869 | 0.16961416 |
| 0.1 | 50 | 0.1728953 | 0.0021895 | 0.1750848 | 0.00046566 |
| 0.1 | 100 | 0.1727427 | 0.0023054 | 0.1750482 | 0.00011929 |
| 0.1 | 150 | 0.1727373 | 0.0023103 | 0.1750476 | 0.00009873 |
| 0.1 | 200 | 0.1727226 | 0.0023212 | 0.1750438 | 0.00021253 |

Both stopped at the declared budget, not the gradient threshold. No convergence
or representation-ceiling claim. All closures and per-call training metrics are saved.

## Remaining limits

Regularization helps this fixed readout comparison, while counting remains only
70.51%. Aggregate ECE for 0.01 is 0.01323 versus original 0.01218; calibration
is not uniformly improved. Closed-taxonomy image gains do not demonstrate
general option semantics or text/audio/video/sensor fusion quality.
Freeze 0.01 as a research reference; do not launch a finer coefficient grid on
these same IDs. Before more optimization, investigate what visual evidence is
missing from failures using training data only, or move to a predefined broader
multimodal task. No reporting images or sealed audit were loaded.

## Artifacts and verification

Run source `59dd88945b3455ceb5d585f0e624a869e15cf7e1`. Command: `python -m scripts.train_shared_affine_regularized --output C:\CodexArtifacts\pretrained-reuse-r2\shared-affine-regularized-01`.
Outputs: `C:\CodexArtifacts\pretrained-reuse-r2\shared-affine-regularized-01`; separate `shared-affine-regularized-smoke-01`.
Initialized/best-objective/final weights, final training logits, full training
history, closure objectives and gradients, validation predictions and input
hashes are preserved in separate directories. Exact final reload passes.
No optimizer state is saved; these are not exact-resume packages.
Measured fit times: 8.846s and 9.217s (CPU; exclude imports/input hashing/evaluation).
No GPU training or cloud cost; peak RAM was not measured. Legacy inputs and
weights remain unchanged. Adjacent JSON contains full hashes and every result.
Ruff passed; 584 CPU tests passed, 1 skipped. Manifest validation and target-commit
CI are checked before handoff.
