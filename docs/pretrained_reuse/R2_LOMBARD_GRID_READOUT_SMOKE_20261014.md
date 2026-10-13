# Native slot comparison: real GPU smoke passed, full learning pending

Separate process`python -m scripts.train_lombard_grid_readouts --smoke` completed
all three arms on the four predeclared **training** reference sources. This is
an execution smoke, not a learned-quality result or selected Teacher. Full native
video extraction remains live; its incomplete array must not be used for training.

## Actual observations

Source commit`9a9a7493f92d5f64763d2fade14c501ff120fddd`. Each fresh seed17 head
has77,235 parameters and six complete native vocabularies(total51 choices,
including25 letters). Initial tensor hashes and sample-order hashes exactly match
across audio/video/joint. Two optimizer updates per arm consume8 assets,4 unique,
4 repetitions and48 slot decisions; no best checkpoint or validation selector in
smoke. All gradients/updates finite. Saved final weights reload bitwise on the same
GPU; native JSON probabilities are exactly stable to complete option permutations.

|Arm|Smoke seconds|Peak CUDA allocated|Peak CUDA reserved|
|---|---:|---:|---:|
|Audio|2.7397|69,730,304bytes|71,303,168bytes|
|Video|0.1304|69,730,304bytes|71,303,168bytes|
|Joint|0.1391|70,040,064bytes|71,303,168bytes|

First arm includes cold startup, so these timings are not fair comparative
throughput benchmarks. They measure the small head on cached features, **excluding
encoder extraction and GPU driver memory outside Torch's allocator**. Local
RTX3080Laptop16GB,Torch2.6.0+cu124,FP32,TF32 off. Cloud cost0.
No heldout performance was evaluated; no calibration temperature was fitted.
The four sources' normalization is smoke-only and is not the full corpus transform.

Artifact directory`C:/CodexArtifacts/pretrained-reuse-r2/lombard-grid-pooled-readouts-smoke-01/`:
separate arm configs/normalizers/sample orders/optimizer logs/final weights/results.
Adjacent JSON preserves source-result SHA, initialization/order/checkpoint hashes,
resource/accounting evidence and exact feature/source provenance. No weight/media
files are in Git. All earlier Teachers/runs/caches remain protected.

## Implemented gates and pending scientific execution

The complete feature loader binds the fixed local-SHA source/rights/corpus/media
and output identities before training. It rejects allocated-but-RUNNING arrays,
wrong order/duplicate sources, dtype/shape/nonfinite/input hash differences and
changed native frame/source/parameter/reload profiles. Real four-source preflight
passed. Full mode requires independent calibration398/validation889 and all4080
training sources, plus **both complete native feature banks**.

Full trainer implements fixed24epochs, identical initialization/sampling, earliest
minimum raw validation macro NLL, separate best/final, calibration-only17-temperature
grid, raw/calibrated predictions and per-epoch fixed-train/validation metrics.
These full-run paths still require actual execution and validation. Never describe
the completed two-update smoke as the full24-epoch study.

Independent pure-control helpers/tests are prepared while extraction runs:
label-independent cross-speaker-group bijections preserve p/l condition and donor
marginals; paired bootstrap resamples entire groups(10,000/seed17/linear-percentile95%
in the final protocol); saved JSON/logit/provenance checks reject corrupted options,
labels, ordering, probabilities and predictions. No real validation controls or
bootstrap comparison have run. Final report/control orchestration is now implemented
in`python -m scripts.evaluate_lombard_grid_readouts`, with CPU fixture tests for
hash checking, frozen-weight controls and partial-controller rejection. Its **real
full-run integration remains pending**, and it must not be invoked before all
three24-epoch runs and the complete-source gates pass.

The report audits all24 histories, six-slot macro/count denominators, best selection,
total versus selected consumption, every768-update optimizer log and actual seed17
sample orders. It reloads both best and final on the training CUDA backend, checks
saved logits/predictions, recomputes train-only normalizers/calibration-only
temperatures, then records fixed-model controls and10,000 whole-group resamples.
Missing-stream masks were not trained, so their scores are explicitly diagnostic
OOD. Donor coupling limits the control bootstrap's independence interpretation.
No claim of generic-video capability or pretrained-data blindness is introduced.

The production-sized artificial full-path test later passed all3x24 epochs,
early-best/final separation and complete replay/control/report checks in29.57s.
This remains **mechanical test evidence**, not native learning or quality;
[exact scope and accounting](R2_LOMBARD_GRID_FULL_PATH_TEST_20261014.md).

Execution ruling: proceed with these independent pure helpers while the full-run
task awaits the verified live extractor; integration remains unverified until all
three actual arms finish. Condition-preserving donor stratification tightens the
label-independent control before quality is observed; it may fail if a condition
has insufficient speaker-group diversity, rather than silently relaxing that gate.
This does not change data, budget, selected-model policy or the frozen protocol file.

Pooled features discard temporal order/fine spatial detail. These native command
slots cannot certify generic video reasoning, the full four-modality95–98% target,
or product readiness. Keep the full goal active and report the real three-arm
results after feature completion; no extra seed/epoch trials were started.
