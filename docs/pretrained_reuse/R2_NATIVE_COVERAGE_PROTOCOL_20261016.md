# Native unique coverage versus repeat: prospective control

The complete consumed-TRAIN evaluation is weak in Text/Image/Video as well as
validation. It does not distinguish optimization, data coverage and capacity.
Before adding recurrence or parameters, compare two freshly initialized fits
at the same update/presentation budget. This is one seed17 development experiment,
not a product promotion or a claim about pretrained encoder ceilings.

## Fixed design

- Preserve the attention readout:889,345 trainable parameters;196,254,874 total
  inference parameters before quantization. Frozen native encoder feature banks,
  all supplied candidates, CE loss and existing full validation are unchanged.
- Each arm starts from the original native seed17 `initial.pt`, with fresh
  optimizer/RNG. No trained checkpoint is an initial state. Separate32-update
  TRAIN-only smokes are discarded; they do not affect the full fits' RNG.
- Both arms:2,048 updates, eight one-question microbatches/update, two from each
  modality, CE/8, AdamW lr.001/betas(.9,.999)/eps1e-8/wd.01/clip1, FP32 and TF32
  off. Warmup62 updates followed by cosine over the full2,048 horizon. Final
  zero-LR call still consumes examples/computes gradients. No performance stop.
- `repeat`: concatenate the original8,192 occurrence pool with itself.
- `fresh`: retain that original pool; for each additional occurrence choose an
  unused TRAIN row with identical modality/source/task/option-count, ordered by
  seed17 ID hash without looking at target/performance. Exhausted buckets repeat
  the original occurrence. No source regeneration or removal of hard examples.
- Both occurrence plans use the same independent pairing method. Their numeric
  occurrence sequences and per-position modality/source/task/option-count quotas
  are identical. Different rows naturally have different labels/candidate text;
  this does not claim label or semantic difficulty matching. Candidate shuffles
  use the same occurrence/row-ID rule. Existing shared occurrences match exactly.
- The causal contrast is the two NEW2,048-step fits. Earlier1,024-step batch and
  four-microbatch2,048-step fits are historical references: presentation budgets,
  batching and/or learning-rate histories differ. This is not a silent resume.

## CPU preflight (completed before either fit)

Frozen plan:`C:/CodexArtifacts/pretrained-reuse-r2/native-coverage-plan-02`.
Runtime reconstructs and compares every pool/plan, verifies all source/input
hashes before/after, and saves effective config and actual occurrence history.

| Property | Repeat | Fresh |
|---|---:|---:|
| Total presentations |16,384|16,384|
| Unique questions |6,656|12,800|
| Repeated presentations |9,728|3,584|
| Unique observations |5,107|8,417|
| Unique media assets |3,114|5,385|
| Unique source groups |3,382|5,461|
| Added Text/Image/Audio questions |0/0/0|2,048/2,048/2,048|
| Added Video questions |0|0|

Every modality has4,096 presentations. Frozen Video has only512 questions and
all were previously consumed; both arms repeat them eight times. This comparison
does NOT scale unique Video data. Source/task/option-count totals are identical.

Plan01 is retained as a failed receipt-integrity attempt: its receipt included
the still-open stdout log, which changed after hashing. No training occurred.
Plan02 excludes the open log from immutable artifact pins; all plan/sample files
are byte-identical to01. `prior-attempt.json` retains this provenance.

## Evaluation and selection

- Full fixed validation24,346, all71 tasks including47 text slices; same IDs,
  order, options and preprocessing. Evaluate0,256,...2,048. Select hierarchical
  task/source/four-modality macro NLL, min_delta1e-5, earliest tie; initial0 is
  eligible. Save every evaluation, best and final separately.
- Keep the existing TRAIN probe1,456 at each point. Record actual consumed overlap
  for each arm; it is not a conventional fixed memorization/generalization gap.
- At the fixed endpoint only, evaluate the same12,800-row TRAIN union for both:
  original6,656; added6,144; current-consumed subset; full union. The additional
  subset has THREE modalities, so its macro is not a four-modality metric.
  Neither TRAIN metrics nor pair diagnostics select the checkpoint.
- Retain all1,250 fixed TRAIN state pairs. Strata are now explicitly named
  `parent_consumed`/`parent_unconsumed`: fresh may consume some formerly unused
  rows. Save actual current exposure for BOTH sides of each pair.
- Fresh-process replay must exactly reproduce full best/final validation and
  TRAIN-probe arrays plus fixed endpoint pair/union arrays. No calibration or
  audit access. Save raw per-example logits/IDs/targets and all metrics.
- Report macro and per-modality/task Accuracy/NLL/Brier/ECE, minimum task accuracy,
  STARSS positive recall and unavailable classes. High all-no accuracy is failure.
  Keep run-wide consumption separate from selected-checkpoint consumption.

## Interpretation frozen before results

Primary comparison: fresh versus repeat at fixed2,048 endpoints. Call coverage
useful at this budget only if fresh lowers macro NLL, increases macro Accuracy,
and no modality loses more than2pp Accuracy or gains more than.05 NLL. These are
research interpretation tolerances, not quality floors or permission to sacrifice
a modality. Report every measured trade-off even when within these tolerances.
Mixed/negative results remain recorded; no automatic seed/budget sweep or adoption.
Separately show validation-selected checkpoints and TRAIN reachability; do not
pick a favorable step from those diagnostics.

Diagnose Video difference using paired scene bootstrap of stored validation
predictions:5,000 draws, seed17, percentile95% with linear quantiles, including
temporal-only separately. This is one seed and repeatedly observed development
validation, not an independent blind result. Changes may come from shared-readout
transfer; Video unique coverage is identical in both arms.

Keep <=200M, each-task95–98%, temporal/conditional/joint/OOD quality, calibration,
deployment and audit gates open until their own evidence passes. No LSTM/xLSTM
expansion, encoder unfreezing, quantization, new Teacher, or product promotion.

## Execution and resource contract

Existing local environment only. Guards:2hours/process,6GiB CUDA allocation,
10GiB process working set,2GiB free disk. Stop on numerical/integrity/resource
failure; report incomplete. Check GPU idle and committed clean source first.
Do not edit tracked source while a runner is active. Never overwrite outputs.
Optimizer/RNG are saved, but this runner implements no resume API.

For each policy `repeat` then `fresh`, execute sequentially with new output folders
`native-coverage-<policy>-<phase>-01` under the R2 artifact root:

```powershell
python scripts/train_native_coverage.py --phase smoke --policy repeat --output <new-smoke-dir>
python scripts/train_native_coverage.py --phase train --policy repeat --output <new-fit-dir> --reference <smoke-dir> --reference-sha256 <verified-SHA256>
python scripts/train_native_coverage.py --phase replay --policy repeat --output <new-replay-dir> --reference <fit-dir> --reference-sha256 <verified-SHA256>
```

Repeat for `fresh` only after the first process exits. Record exact arguments,
source commit, environment, UTC times, fit/evaluation time, memory and hashes.
Only completed fit and exact replay establish execution success.
