# Matched condition-relevance comparison, before model evaluation

## Question and scope

Does explicit request-relevance supervision help a fixed shared decision scorer
avoid selecting true but unrequested facts, without degrading retained tasks?
Run exactly two fresh seed17 arms, CE and CE+relevance. This follows the measured
oracle/native bridge failure. The prior MiniLM contrast-loss comparison remains
a failure: it used paired condition contrast, not the binary relevance labels
and shared DeBERTa pooled features tested here. No seed/loss/template sweep.

User authorization to choose strategy and proceed autonomously covers this
bounded extension of the existing fit. Existing experiments, data, selectors
and model artifacts remain read-only. This is one research comparison, not
completion of the four-modality95–98% product objective.

## Common fixed inputs and compute

Use `condition_relevance_fit_v1.json` and the frozen corpus manifest
`7593e36769996d07e3654694013a989a8cdcd8a4b58442a29bde4209feb26964`.
Both arms start from DeBERTa xsmall revision
`262ae02f29173eec1c250f90804dc7edc677dcff`, no previous delta or optimizer.
Freeze49,191,168 embedding parameters; train21,639,554 encoder/pooler/classifier
parameters. Add the identical initialized384→1 training-only auxiliary head
(385 parameters) in both arms:21,639,939 optimization parameters. Inference
still uses70,830,722 core parameters and native entailment logit0, without the
auxiliary head. This does not add another runtime model to the195.66M inventory.

Public pairs remain question+candidate text and state text. No source, target,
field ID, supervision, option ID or task routing enters the forward input.
All supplied candidates remain; no target-field mask. Native backbone, pooler,
dropout and classifier operations match the original forward exactly in eval
mode; the auxiliary head reads the same pooled representation.

One epoch,1024 optimizer calls per arm,1023 with positive LR. Each update has
8 new relevance questions,8 reused numeric and2 legacy questions. All18,432
questions are consumed once:8192new/10240reused,2626unique states,0native assets.
The actual sampler dry-run gives ordered-ID SHA256
`2b9cee7d8fb9dab5790b96f32b9deef9bdf5b73fae1a81fae42aa108c2593cfd`.
Use the same deterministic sample order across arms. AdamW(.9,.999), eps1e-8,
weight_decay.01, peak2e-5,31 warmup updates, cosine to0 at1024, clip norm1.
FP32/eager, dropout disabled, deterministic algorithms, TF32 disabled.
One question and all its options per backward; divide objective by18.

Legacy rows add coefficient1 KL against the same fresh step0 option distribution.
Both arms compute a binary relevance loss for new binding rows: half mean BCE
over the10 relevant candidates plus half mean BCE over the30 irrelevant ones.
CE arm coefficient0; relevance arm coefficient0.1. Auxiliary labels do not assert
fact truth. Only the correct-choice CE teaches the intersection of relevance and
fact truth. Auxiliary loss does not apply to numeric or legacy rows. It backprops
through the shared pooled representation; the head is discarded for inference.
Coefficient0 still computes the same auxiliary path, matching forward work.

Use TRAIN-only, separate-process two-update smokes. Main arms reinitialize all
RNG/model/optimizer states. No smoke weights or quality selection are reused.
Both run sequentially on the local RTX3080Laptop16GB. Per-arm limits:4h wall,
14GiB CUDA,12GiB host,2GiB output and4GiB minimum free disk. NaN/Inf, hash/frozen
failure, OOM or resource limit is INCOMPLETE, never evidence against the hypothesis.
Token limit512; new binding/numeric must not truncate. Preserve question and
candidate; record every state-side truncation in retained prose rows.

## Evaluation and selection

Evaluate0/256/512/768/1024 on the identical10,732 validation rows in order:
2048new binding,3072old development,5612legacy. Preserve all31 old metric slices
plus16 new query/field/form slices. Evaluate the fixed TRAIN subset of first8
SHA256(seed,ID)-sorted rows per training slice. Record Accuracy/NLL/Brier/ECE,
macro/minimum, predictions/probabilities, consumption, LR/gradient norm/time/VRAM.
Train/validation distributions differ; the fixed-train gap is descriptive.

The existing retention selector applies unchanged, now to47 exact slices:
Accuracy within-1pp, NLL/Brier/ECE no more than+.02/+.01/+.02 versus same-arm
step0. Among eligible checkpoints, minimize equal-slice macro NLL; ties choose
earliest step. Include0. Save every scheduled delta, best and endpoint separately.
Do not relax calibration gates or calibrate on validation. Reserved calibration
is binding-only and remains unopened during this comparison.

The primary causal contrast is the fixed1024 endpoints of the two arms; report
the two selected checkpoints separately, even if both select0. Verify identical
base/aux initialization, data/order, validation IDs, LR timeline and code first.
For new16-slice minimum Accuracy, paired bootstrap whole128validation states,
all16 questions together,10,000 replicates, seed17, percentile95% with linear
quantiles. Recompute the minimum each replicate. Report each field/form/query
as well: minimum alone does not describe all gains or losses.

Auxiliary supervision is promising only if its endpoint improves minimum new
slice Accuracy by>=3pp over CE with positive lower paired interval, AND its
selected trained checkpoint passes all47 retention gates. Report mixed outcomes
without calling them adoption. Within-arm step0 comparisons are supplemental;
they do not isolate the auxiliary effect. Validation selection and authored
wording limit inference; this is not a blind quality estimate.

## Replay, transfer and next action

Reload selected weights including the training-only head and require exact
saved validation/fixed-TRAIN scores. Retain precise input/code/weight hashes.
Optimizer/RNG are not persisted: no claim of exact training resume. A failed
attempt is preserved, not silently restarted or resumed from an old budget.

No old/new transfer or sealed audit is opened during smoke, fitting or selection.
After both arms finish, freeze their selected/endpoint identities before any
transfer comparison. The corpus already fixes the same256states under
seen/new question × seen/new candidate wording. Evaluate that fixed factorial
once per frozen comparison, with no checkpoint, wording or calibration reselection.
A separate inference-only runner must enforce those pins before access.

Native comparison follows only after this bounded comparison has been reported.
Use the existing observed image bridge only as regression; independent assets
and temporal/joint tasks still need a prospective native contract. Synthetic
improvement alone does not authorize promotion or further synthetic tuning.
If there is no useful transfer/retention gain, close this relevance intervention
and reconsider representation or teacher signal, without a new coefficient sweep.
