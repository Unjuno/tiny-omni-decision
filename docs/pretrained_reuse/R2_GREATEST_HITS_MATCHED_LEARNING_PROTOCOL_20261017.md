# Matched AV learning: prospective conditions

The completed eight-update probe establishes mechanics and approximate local cost,
not quality. This protocol fixes the next full comparison before seeing its results.
It is scoped to Greatest Hits native material/action/reaction development labels.
It cannot certify both-required reasoning, physical AV synchronization or actor/site
independence. Those product requirements remain open.

## Shared conditions

Use C0 Attention and E0G0/E0G1/E1G0/E1G1. Every arm starts from the same existing
`native-long-budget-fit-01/best.pt` model at step8192 (SHA256
`d7b666cd7ea60dcf236bd06e53b0107e0d5f74193824c1f4b44bbe4d5cdc1b56`).
Keep the added tensors from the exact passing transport fixtures, including the
zero output correction. All original baseline tensors are copied exactly.
This is a model warm start with a fresh optimizer, not a training resume. The
earlier eight-update fixtures were random readouts, so their losses are not part
of this learning curve. Preserve the selected control and all older artifacts.

All readout parameters remain optimizer-eligible, frozen encoders are represented
by immutable features, no parameter-count increase beyond197,097,949. The existing
common corpus uses its original feature preprocessing. Greatest Hits uses its
fixed full-extent/8-frame cache. Each dataset's geometry is identical across arms;
this comparison does not estimate a preprocessing effect.

Fixed seed17,32,768 optimizer calls,8 microbatches/call, mean option CE. Each update
has4 joint Greatest Hits questions plus1 Text/1 Image/1 Audio/1 Video question from
the unchanged common TRAIN corpus. AdamW lr0.001, betas0.9/0.999, eps1e-8,
decay0.01, clip1, FP32, warmup62, cosine ending at32,768 as in the existing native
long-budget schedule. Every supplied option is retained and a deterministic full
permutation is applied per presentation. No gate-balancing loss, data augmentation,
new seed, early performance stop or outcome-dependent budget extension.

Within each of the five streams (joint plus four modalities), cycle through
source/cohort/task buckets in sorted round-robin order. Within a bucket, sort by
SHA256(seed, stream, cycle, bucket, question ID). Skip exhausted buckets. Never
repeat a question in a stream before all questions in that stream have been used.
Reset only after whole-stream exhaustion, with a cycle-specific order. This differs
from the old small repeated pool; comparisons to old curves are references, not
single-factor training-budget effects. All five arms share the new exact plan.
Labels do not influence sampling. Report actual label/class support, bucket
exhaustion, task mix drift, unique questions/observations/assets and repeat counts.

## Evaluation and selection

Evaluate at0,512,1024,2048,4096,8192,12288,16384,24576,32768. Do not shorten the
registered curve after an unfavorable early result. Safety failures remain
incomplete experiments. Save each scheduled checkpoint and best/final separately.

Use all8,875 fixed joint validation questions and all24,346 fixed common validation
questions in their existing order. Record Accuracy/NLL/Brier/ECE, confidence,
per-task/source/modality metrics and raw logits with ID/options/targets. Joint
macro metrics are equal averages across the three native tasks. Include full
fixed consumed-TRAIN metrics at the same points, not rolling batch losses called
a generalization gap; report their differing task mixtures. Missing-stream views
use the same joint question IDs and are input ablations, not certified answerable
subsets. Report both joint benefit and each unimodal result.

Checkpoint eligibility: every existing modality's hierarchical macro Accuracy
must remain within2 percentage points of that arm's shared initial reference,
and existing hierarchical macro NLL must not exceed initial NLL by more than1e-5.
The2pp margin is the existing provisional research retention rule, not permission
to lower the product task95–98% objective. Within eligible checkpoints, select
lowest joint task-macro NLL, min_delta1e-5, earliest tie, step0 eligible. Record
ineligible and unselected results as well. No seed/wording/option-count search.

For comparison, retain fixed equal-step results and each arm's selected result.
Require improvement against C0 and factor-specific contrasts within the2x2;
C0 alone does not control extra parameter capacity. Audit gate distributions but
do not interpret them as global importance or calibrated reliability. Label-free
input replacement diagnoses sensitivity only; correctness claims require defensible
unchanged/recomputed target labels. Freeze any intervention inventory before
scoring it. Do not invent conflict truth or a new unknown option after observing
model outputs. No sealed audit/calibration/test access or product promotion.

## Execution gates and cost

First execute the entire sampler on real TRAIN metadata and save its hash and
coverage. Verify warm-start copying, equal initial outputs, retained200M bound,
schedule, selection and option alignment with tests. This does not admit training.
Full video extraction must finish and pass independent integrity/clock/identity
binding with the separately revalidated full audio cache. Rehearsal banks must
also retain their recorded hashes. Measure a separate full-run-path smoke and
full evaluation cost before admitting a local run; keep the above scientific
conditions fixed. Record actual/projected runtime and resource guards before
launch. No automatic budget reduction to fit a timer, no cloud or GPU contention.

Only then run the arms sequentially, preserve source/config through execution,
save reload evidence and all attempted outcomes. Full training, retention quality
and the final paired comparison are pending, not represented as scheduled jobs.
