# Fixed replay source coverage comparison (2026-10-15)

## Question and scope

The instruction-diversity run improves some novel wording but fails retention.
Its replay has 2,048 original questions and zero explicit curriculum questions.
Test whether covering the retained curriculum tasks changes this trade-off at
the same question/update budget. This is a single seed17 comparison, not a
capacity test, a new blind audit, or a four-modality product qualification.
Broad autonomous local experiment authorization applies; no paid compute,
legacy edits, process intervention, new dependencies, or sealed audit access.

## Frozen intervention

Control: `binding-language-fit-01`, complete512, selected0.
Candidate: `binding-replay-coverage-fit-01`, new output and new optimizer from the
same coverage-parent delta, never continuing control final or selected weights.
Exactly one fit of512updates, plus separate two-update TRAIN-only smoke.

Unchanged: 2,048 binding TRAIN questions/512states, all168instruction strings,
candidate descriptions, targets, 40options, binding order, source-index schedule,
seed17, FP32/SDPA, dropoutoff, 22,713,601total/10,647,169trainable parameters,
all6BERT encoder blocks and classifier, constantLR2e-5, AdamWwd.01, gradientclip1,
4binding/4replay per update, 0.5bindingCE+0.5(replayCE+KL(parent||student)),
KLcoefficient/temperature1, original/curriculum/binding validation IDs/order,
fixedTRAINprobe, evaluation at0/128/256/384/512, selector and all retention limits.

Only semantic intervention: replace half of original replay with curriculum TRAIN.
For each original scheduled replay batch, retain its first two original indices;
replace its last two with curriculum. Curriculum consists of128complete states,
8questions each, selected by seed17 SHA ordering of group, then independent
seed17 SHA question ordering. Selection reads no scores or validation labels.
Total replay remains2,048unique questions:1,024original +1,024curriculum.
Curriculum mix:512extremum,256temporal_order,128count,128conditional_selection.
Question content/options/targets and underlying groups are unchanged.
No ratio sweep. Dropping1,024original questions is part of this budget-neutral
intervention; changes cannot be attributed to adding curriculum alone.
Candidate-pair count, state coverage and runtime may change despite equal question counts.

A new TRAIN-only parent logits cache is computed before updates, with the exact
mixed batch order/shape. This replaces the previous original-only cache, not its
teacher or loss definition. Cache rows bind complete row content and option order.
No gradient/optimizer step during caching; initial student-parent KL must be zero.

## Integrity and execution

Original artifacts remain read-only. Freeze config, code commit, input hashes,
copied binding TRAIN/sample order, new replay and accounting before GPU execution.
Reject non-TRAIN rows, duplicate IDs, malformed8question curriculum groups,
source/state or canonical-content overlap with the full existing validation.
Preserved binding corpus has independent TRAIN/transfer states; new curriculum
and original structured-state types differ from the binding state schema.
The trainer does not parse transfer payloads. Regression loads them after selection.
No historical sealed audit is read. Fail closed on hash mismatch and output reuse.

Resource guards unchanged:1,800s per process,8GiBallocatedVRAM,5GiBhost peak,
2GiBminimumdisk,512MiBoutput. Local RTX3080Laptop16GiB only. Smoke uses another
process and output, followed by fresh parent loading/seed initialization in fullfit.
NaN/Inf/frozenweightchange/resource failures mean INCOMPLETE, not quality failure.
No quality-based earlystop, extra seed, extended budget, or automatic continuation.
Keep final512 and best separately. This saves weights, not full optimizer/RNG resume.

## Selection and comparison

Existing retention rule unchanged, relative to step0 coverage parent:
each old taskAccuracy no more than1pp lower; source NLL<=+0.02,
Brier<=+0.01,ECE<=+0.02. Select lowest binding validationNLL among eligible steps,
earlier step wins ties. Step0 fallback is experiment-local and cannot replace
the previous accepted KL512 reference as a claim of a new improved model.
Verify baseline validation raw scores against control exactly, selected reload
in-process, then all12selected outputs exactly in a separate process.

Primary comparison: fixed512 candidate versus fixed512 instruction-diversity
control on full original/curriculum validation, all task metrics, retention
eligibility and binding NLL/Accuracy. Report all checkpoints and selected versus
parent separately. State-paired bootstrap10,000replicates,seed17,percentile95%
linear intervals. A fixed512 retention pass plus lower curriculumNLL with upper
paired CI<0 supports usefulness of replay coverage at this budget. Otherwise
report mixed/not supported, with no causal model-capacity ceiling claim.

Post-selection regression uses the seven already-observed
`binding-language-corpus-01` conditions without edits. Compare oldlanguage512,
new512, and newselected if different; report all40options, all fields, calibration,
condition pairs, reversal and no-question controls. Same raw output fresh replay.
These previously seen scores cannot select checkpoints or tune the replay ratio.
Prior strict transfer gates remain reported, but the >=3pp improvement gate is a
diagnostic inherited from language expansion, not this retention intervention's
primary hypothesis. Never promote a retention-failing final512 even if transfer improves.

## Required records and next action

Save config/environment/code and data hashes, times/memory, all update losses/KL/
gradient norms, fixedTRAIN and full validation curves, source/task/group accounting
at each checkpoint, final/best weights, rawscores/probabilities, and regression
per-example predictions. Distinguish run4,096questions from selected consumption.
CPU tests cover deterministic replacement, complete states, preserved content,
zero repeats and sourcecounts, TRAIN-only cache and fixed config. Run Ruff, full
CPUtests, manifest validation and exactcommitCI; independent review before result adoption.
Publish results even if negative. No additional fit is implied by this protocol.
Keep general conditional semantics, native perception/temporal/joint quality,
all-resident<=200M and each-required-task95–98% as outstanding product gates.
