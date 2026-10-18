# Native unique-coverage long-budget comparison — fixed protocol

## Question

The 32,768-update original-crop control repeatedly trains on a fixed set of
6,656 unique TRAIN questions. This follow-up asks whether replacing one repeated
pool cycle with previously unused TRAIN questions helps at the same long update
budget. It is a single seed-17 development comparison; it does not test the
four-modality product goal, prove a capability ceiling, or justify promotion.

## Fixed design

- Reuse the completed repeat control at
  `C:\CodexArtifacts\pretrained-reuse-r2\native-long-budget-fit-01` rather than
  rerunning it. Its result SHA-256 is
  `645bedbe73ed508fad68c4f636a41df6c8630cd996641c44ce2a87d31550250b`;
  source commit `8756a09f582554c0321ed5ece71846d6b19469cd`, best step 8,192.
- Train one new `fresh` arm from the original seed-17 initialization with a
  fresh optimizer and the same 32,768-update schedule, batch grouping, loss,
  option shuffling, frozen encoders, original crop/features and full validation.
  No early stopping. Scheduled evaluations remain
  `0, 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768`.
- The only intervention is the frozen `fresh` occurrence pool. It retains the
  original 8,192 occurrences and replaces as many second-cycle occurrences as
  possible with unused questions matched by modality, source, task and option
  count. Existing plan logic preserves the same modality/source/task/cardinality
  quotas; no labels or validation scores choose new rows.
- Repeat control: 6,656 unique TRAIN questions. Fresh arm: 12,800 unique
  questions, adding 6,144 rows from Text, Image and Audio; it adds no Video
  questions. Both arms retain 512 CLEVRER question rows grouped into 256 video
  scenes. The pool contains
  16,384 occurrences and is repeated for 16 cycles, totaling 262,144
  microbatches / 32,768 updates. Unique and repeated counts are recorded
  separately.
- The complete 24,346-row validation set, IDs, order, options and preprocessing
  remain fixed. The fixed selection rule is hierarchical macro NLL with
  `min_delta=1e-5` and earliest tie; initial step 0 remains eligible. Train
  probes are descriptive only: the fresh arm's probe has 12,800 rows while the
  historical repeat run's probe has 6,656, so those probe aggregates are not a
  matched memorization-gap comparison. Endpoint evaluation of the same 12,800
  row TRAIN union is retained by the runner.
- The already-completed repeat receipt and validation-ID hash are pinned in the
  new config. The runner rejects a changed control, changed source inputs, or
  mismatched split/feature artifacts.

## Run and resource policy

Use the new config
`configs/pretrained_reuse/native_unique_coverage_long_v1.json`; do not change
`native_long_budget_v1.json`, prior corpora, checkpoints or output directories.
Use distinct `native-unique-coverage-long-fresh-{smoke,fit,replay}-01` output
directories under `C:\CodexArtifacts\pretrained-reuse-r2`. Run smoke, fit and
fresh-process replay sequentially. Fresh-process replay must exactly reproduce
best/final validation and train-probe arrays plus endpoint union and state-pair
arrays. No calibration, sealed-audit evaluation, extra seed, quantization,
encoder tuning, or model promotion.

Only local compute is permitted. The current execution path uses the cooperative
HotUGPU reservation wrapper: it waits for Orbitune to yield, reserves memory
before launching the runner, passes the reservation PID to the child gate, and
does not stop or signal Orbitune. The child requires the reserved parent and no
other compute PID, utilization below 5%, and 8–14 GiB free after reservation.
The runner's existing per-process budget remains two hours, 6 GiB CUDA
allocation, 10 GiB working set and 2 GiB free disk. The completed repeat control
took 2,550.9 seconds total (1,822.6 training + 682.0 evaluation), peaking at
209,224,704 allocated VRAM bytes. The fresh probe is larger; its measured smoke
projection must remain under the existing two-hour budget before full training.
No paid compute is used.

## Interpretation fixed in advance

The primary comparison is selected and fixed-endpoint validation against the
existing repeat control using identical validation IDs. Report all modalities,
task/source slices, macro metrics, min-task accuracy, Brier, NLL, ECE, paired
scene bootstrap for Video, train-pool coverage, resource measurements and
overfitting signals. Do not select a favorable intermediate step outside the
predeclared selector. Treat this as low-power repeatedly observed development
validation, not an independent blind test. Mixed or negative results are
retained; no automatic budget or seed extension follows.

## Run status

The pre-run snapshot above was historical. Smoke attempt 03 passed after a
cooperative Orbitune yield; fresh fit 01 completed 32,768 updates and replay 01
passed. Smoke attempts 01 and 02 were interrupted before the documented yield
completed and are retained as blocked. See the
[results report](R2_NATIVE_UNIQUE_COVERAGE_LONG_RESULTS_20261018.md) for
receipts, SHA-256 values, comparison metrics and caveats.
