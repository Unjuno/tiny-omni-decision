# Conditional field binding with retention: one bounded training comparison

Prospective follow-up to the frozen semantic-state diagnosis. One new seed17
candidate starts from the retained22,713,601parameter coverage-parent model.
The frozen parent is the control. No extra seeds, schedule/capacity search,
encoder extraction, quantization or product promotion. This measures the joint
effect of this fixed mixed continuation; no replay-only training control is run.

## Fixed data contract

The four CLEVR-4 attributes define10,000possible complete states. Existing8,424
TRAIN images have5,636distinct fact combinations. The preceding diagnosis
observed308distinct oracle/predicted combinations. Excluding both leaves4,340
unused combinations for semantic validation/transfer; these have no actual media.

- Binding TRAIN:512distinct existing TRAIN annotation states, excluding every
  previous oracle/predicted diagnostic state. Four questions each=2,048examples.
- Binding selector:256state combinations absent from all8,424TRAIN annotations
  and preceding diagnosis. Four questions each=1,024examples.
- Transfer development:256different unused states, reserved until selection.
  Same states crossed with familiar/novel question wording and familiar/novel
  candidate descriptions. Four fixed conditions,1,024examples each.
- Retention replay:2,048original Typed Decisions Synth TRAIN questions selected
  by stable SHA17 ID order; one pass, no repeated IDs in this run.
- Retention evaluation: **all**2,540original and2,048curriculum validation
  questions, with unchanged IDs/order/criteria/options from the parent config.

All grouping uses normalized four-fact state identity; candidate/wording changes
never put one state into different splits. Selector/transfer combinations are
excluded against the full upstream TRAIN state pool, not merely the512fit states.
The source taxonomy and question/description templates are fixed in code before
results. `color/texture/count/shape` remain state keys; transfer is narrowly new
combinations/wording/descriptions, not unseen taxonomies or proven natural language
generality. Each question retains the same40typed candidate IDs. `texture:star`
and `shape:star` stay distinct. The old40choice diagnostic stays unchanged.

Two fixed TRAIN templates per field and two description styles are selected by
state hash. Transfer uses exactly one new question per field and one new
description style, with all four factorial arms. No wording search. Native media
are not consumed in this fit; annotation-derived states are semantic supervision.
Rights/provenance reuse the existing ALLOW CLEVR-4 and Typed Decisions manifests.
No old or new sealed audit is opened. The transfer set is development evidence,
not a release audit; do not retry against it.

## Fixed learning and checkpoint rule

512AdamW updates, LR2e-5constant, weight_decay0.01, gradient_clip1.0,
seed17, FP32SDPA/TF32off, dropout disabled,4CPUthreads. Existing six encoder
blocks+classifier (10,647,169parameters) train; embeddings/pooler remain frozen.
No architecture or parameter-count increase. Optimizer reset, not exact resume.

Each update uses4binding and4replay questions, separate source microbatches to
avoid padding short40choice rows to long original-state lengths. Equal weight
per question: each source mean loss is weighted0.5. One pass over2,048rows/source,
4,096unique question IDs total. Original replay may have been seen in earlier
training; do not call it newly acquired data. Count state/group reuse separately.

Evaluate at0/128/256/384/512. Step0parent is always retained. Eligible checkpoints
must keep **every** old source/task Accuracy within1percentage point of step0,
and each old source's NLL/Brier/ECE increases <=.02/.01/.02. Among eligible points
select minimum new-binding validation NLL, earliest tie. Save best and final
separately; no performance early stop or extra steps. Keep failed eligibility and
all learning-curve points. A fixed256question binding TRAIN subset is evaluated
at each point for an actual train/validation gap (not rolling minibatch metrics).

New binding inputs may not truncate. Original replay/retention keeps the parent's
512token `only_second` truncation policy; record truncation counts. Save all
validation raw scores/targets/IDs, metrics by source/task, selection/gates,
consumption/order hashes, learning rate, gradient norm, time and peak allocation.
NLL uses stable FP64 log-softmax. Retention baseline is recomputed with this same
method rather than compared to old clipped FP32 metric scalars.

## Smoke, transfer and verification

Separate smoke: two updates using TRAIN only, no validation/transfer. Require
nonzero finite trainable gradients, frozen-weight hash preservation and actual
memory/time evidence. Full run cap1,800s/8GiB CUDA allocation/5GiBhost/512MiBoutputs,
2GiB disk reserve. Do not modify these limits mid-run. Preserve incomplete runs.

Only after best selection, freeze it and score the four transfer conditions with
the parent and selected candidate, plus the prior oracle/predicted-state
diagnostic as regression. No selector or hyperparameter change from transfer.
Reverse candidate order once on the familiar transfer condition; map back to IDs
and report output error and answer changes. Question-absent control retains all
four original targets and must not exceed25%/zero all-four correctness.

Report Accuracy/NLL/Brier/ECE/confidence and per-field results; all-four and six
changed-question-pair both-correct. For parent-to-selected Accuracy/NLL differences,
resample whole state groups10,000times, seed17, percentile95% intervals. Selector
results have selection bias; transfer is one authored wording family and remains
development evidence. A >=3pp improvement with supporting grouped uncertainty
is a research progression criterion, not the95–98% product threshold. Retention
failures, weak-field failures and transfer failures remain visible.

Verify exact selected validation reload, then a separate-process selected-model
replay of validation/transfer outputs (no refit). Hash original sources before/
after; new outputs only. All four native modalities, temporal/crossmodal benefit,
calibration, resident size and package quality remain separate open product gates.
