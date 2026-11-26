# Train-only six-coefficient native AV correspondence comparison

Prior goal turn is progress: actual raw-media/fresh replay and source checks/CI
at327c50b. Full Omni objective remains <=200M prequantization, native four-modality
decisions, reuse/custom JSON and95–98%/OOD quality. This is a bounded Path A
readout experiment under user's autonomous local/public ALLOW authorization.

## One prospective intervention

The immutable original matcher sums six native posterior-agreement terms with
equal weights. Fit only six nonnegative coefficients, constrained to sum6,
on the exact frozen TRAIN random+hard catalogs. Original encoder/head/normalizer/
12 native temperatures/source/media/splits/catalogs/options remain unchanged.
Full51 semantic options and25 letters stay; no parent deletion or new negatives.
One additional query temperature is fitted on CALIBRATION only after coefficients
are frozen, same17geometric .25..4 grid/earliest minimum NLL. No separate per-policy
temperature/weight search. This tests reliability weighting, not new encoder
semantics/capacity, independent quality or generic Omni competence.

All4080 TRAIN parents/36speakers,16320 roles per policy/32640 total. Exact source,
media/speaker/intended+actual normalized-content grouping gates retained;398cal
and889development parents remain. Candidate weights see only TRAIN term arrays
and quartet-choice targets. IDs/targets never enter posterior-term score API.
Catalog seed17, no randomized model initialization/seed search. No sealed audit.

Coefficient optimization is deterministic FP64 projected gradient on simplex:
initial all1,512 fixed full-batch iterations, initial step1, backtracking half,
Armijo .0001, maximum24 trials per iteration. Stable log-domain NLL/analytic
gradient, no clipping. On exhausted descent trials preserve weights and log
no-op; still execute all512 iterations. Report attempted/accepted/changed steps,
gradient norms, step sizes and every weight/NLL history. No validation stopping,
checkpoint choice, hyperparameter adjustment or last-point Accuracy selection.
Save final512 coefficients/hash BEFORE reading calibration scores; save query
temperature/hash BEFORE development scores. Raw TRAIN-only checkpoints at0/1/32/
64/128/256/384/512; only final512 is evaluated on development.

512iterations correspond to16711680 iteration-level role presentations over4080
unique physical AV sources, not16.7M new assets. Actual optimizer objective/gradient
label uses also include initialization and every Armijo retry:32640 times
[1+sum(accepted?backtracks+1:24)], minimum16744320. Record both separately.
Count normalized unique query/candidate contexts across policies and repeated
contexts; no neural encoder/head forward/backward or neural optimizer is used.
Registered AV122790899 parent +6 coefficients =122790905;5free simplex degrees.
This counts both full heads and normalization/calibration, AV slice only.

## Measurement and safeguards

First separate-process smoke uses the first8 TRAIN role rows, checks finite
gradient/simplex/monotonicity/save-reload, no calibration/development metrics. Then
execute full fixed fit. Exact all1 scores/metrics must reproduce original parent
before candidate comparison. Log all TRAIN curves; raw/calibrated train/cal/dev
Accuracy/NLL/Brier/ECE/confidence, gap, both policies; original native heads remain
unchanged (no new per-modality quality claim). Compare each policy on identical
3556 development roles from889sources/ninespeakers.

Keep the same uniform missing-input prototype and audio/video cyclic swap controls
with original/corrected labels, exact candidate+ID reorder checks, and per-query
ID/target/probability/log-probability snapshots. Paired whole-speaker bootstrap
10000/seed17/linear percentile95 for candidate-minus-parent Accuracy/NLL/Brier/ECE,
not row bootstrap. Selected parent models and observed development limit inference.

Eligibility for a later runtime trial (not product adoption): both policy Accuracy
>=parent, both NLL improve>=1%relative, neither Brier worsens, neither ECE worsens
>.005absolute. Report interval uncertainty even when point gates pass.
Also report p/l and closest-negative six-slot Hamming-distance development strata;
for strata with>=100 roles, Accuracy must not drop>2pp for runtime eligibility.
These metadata enter reporting only, not coefficient/temperature selection.
Failed gates close this six-coefficient comparison; no new step/seed/weight/tolerance
search to rescue it. Research coefficients/history are retained regardless of
outcome; no Teacher artifact overwrite or production promotion.

CPU FP64/NumPy, no new GPU/model loads or cloud. Cap600s/2GiB parent/64MiB output/
1GiB disk; old GPU8GiB guard/live processes protected. Source/runtime/raw-logit/
catalog/media receipts and payload hashes before/after. New output only. Separate
Actual imported hash helper and psutil monitor source/binary files are included.
fresh full fit/replay checks history, final coefficients/T, all prediction arrays,
metrics/intervals/counter controls and terminal inputs exactly. All code committed
before execution; all attempts/failures preserved; CPU/NoTorch tests/Ruff/manifests
and exact pushed HEAD CI before publication. No metric-driven source edits.
