# Full consumed-TRAIN reachability diagnostic

The fixed probe used by the completed fits contains1,456 questions, but only186
were consumed. The1,250 constructed state contrasts are also a selected diagnostic.
Neither is a complete fixed-set measurement of what those models actually trained on.

Evaluate **every distinct question actually consumed** by the four completed
fits: original attention/LSTM2048 and independent/matched-batch1024. Verify their
actual update histories have the same8,192-occurrence multiset before inference.
This yields6,656 distinct TRAIN questions, preserving every task, option and target.
Sort by immutable row ID, without label/error/quality selection. Report unique-row
and occurrence-weighted metrics separately. Repetition is not additional evidence.

Compare the four fixed final checkpoints and the original untrained readout on
these same inputs. The initial reference has consumed zero examples; "consumed"
describes the evaluation set's membership in the four fits. No training, new
checkpoint selection, calibration, prompt search, or audit access. Keep all old
artifacts/configs/corpora unchanged. These are conditional development diagnostics,
not new independent quality estimates.

Use the exact pinned native encoder banks, query vectors, FP32 model implementations
and existing Torch2.6.0+cu124 environment. No encoder recomputation or extra modality.
Native LSTM has reset state per input block, exactly as in its completed fit.
Verify model state digests before/after evaluation and source/input hashes before/
after each process. Store full score arrays and source/task/modality metrics,
positive event recall, resource use, precise commands/commit/config/checkpoint hashes.

Execute separate smoke32 TRAIN questions (first8 sorted IDs/modality), full probe,
and full fresh-process replay. Require exact equality of all five complete arrays.
Use new directories `native-consumed-{smoke,probe,replay}-01` under
`C:/CodexArtifacts/pretrained-reuse-r2`. Resource guard1800s/process,4GiB CUDA
allocated,10GiB working set,2GiB free disk. No competing GPU work or paid compute.

For TRAIN/validation comparison use the already saved fixed-endpoint validation
scores. Compute gaps on the exact common source/task IDs; keep TRAIN-only and
validation-only tasks explicit instead of averaging incompatible task inventories.
Report all missing task coverage. Even common-task populations/label prevalence
can differ, so these are descriptive gaps, not causal generalization estimates.

## Decision consequences

- If actual consumed TRAIN is strong and validation is weak, prioritize data/
  distribution generalization; do not call the result insufficient head capacity.
- If actual TRAIN is also weak, the completed fixed-budget optimization has not
  learned those tasks. This does not distinguish capacity, optimization and
  insufficient coverage. Use this evidence to choose one bounded next intervention
  with a matched control, before LSTM/xLSTM expansion.
- Interpret event all-no separately from Accuracy. No independent task/quality
  gate is reduced or replaced by an ability to memorize TRAIN.

The model inventory remains <=200M before quantization. Native four-modality,
optional conditions, JSON probabilities, temporal/joint/OOD, per-required-task
95–98%, calibration and deployment remain the actual unfinished goal.
