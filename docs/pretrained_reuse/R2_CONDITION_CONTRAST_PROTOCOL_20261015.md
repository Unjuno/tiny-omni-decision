# Conditional contrast supervision: prospective matched comparison

Question: does supervising how a candidate's probability changes across questions
improve conditional transfer, without sacrificing existing decisions?
This is the next experiment in NEXT_STRATEGY_20261015.md. Broad autonomous local
research authorization applies. No audit, native quality claim, cloud or legacy edits.

## Fixed data and two arms

Exclude every structured binding state in the existing image TRAIN annotations,
semantic-state diagnostic, original binding partitions, independent transfer and
language-transfer states. Pick a new deterministic seed17 SHA ordering from the
remaining finite state space:512 TRAIN,128 selection,256 post-selection states.
No repeats or question-count reduction; all40 semantic candidates remain.
No images are attached: native asset count is zero, states are synthetic combinations.

TRAIN has4 questions/state (all4 attributes),2048 unique rows,8 prespecified syntax
families equally represented:4 direct,4 exclusion. Candidate descriptions have2
fixed TRAIN forms. Selection uses2 different syntax families and original descriptions,
1024 rows/128states. Post-selection includes known-form, new focus, new complement,
new focus/complement descriptions, reversed focus descriptions and no-question.
Each post condition has1024 rows/256states; these are development comparisons,
not an unopened product audit. Exact literals are frozen in condition_contrast.py.

Each update uses all4 queries of one TRAIN state plus4 replay rows. TRAIN groups
are seed17 hash-ordered and consumed once. Replay is byte-identical to frozen
binding-replay-coverage-corpus-01; its original per-update order/shape is retained
(2 original+2 curriculum). Both arms see precisely the same sequence and counts.
Using new states/grammars/grouped batches changes conditions relative to older fits;
only the new two-arm comparison isolates the added objective.

- CE control: `.5*bindingCE + .5*(replayCE + parentKL)`.
- Contrast candidate: same loss plus `.5 * .25 * contrastCE`.

For a4-query state, let L be the4x40 option logits and y_i each query's target.
Set P=log_softmax(L, dim=options). contrastCE is cross_entropy(P[:, y].T, [0,1,2,3]).
For each of the4 correct candidate meanings, the query it answers should receive
the highest conditional probability. The auxiliary uses training labels only;
inference still scores all40 candidates and sees no target metadata. It requires
one complete state, common aligned candidate IDs, unique targets and finite logits.
It is not a new inference head. Coefficient.25 is fixed, not tuned on validation.

## Model, budget and safeguards

Two fresh seed17 fits from the same text-coverage-training-01/best-delta.pt on the
pinned22,713,601parameter MiniLM CrossEncoder;10,647,169trainable parameters in
all6 encoder blocks+classifier. Embeddings/pooler frozen, dropoutoff, FP32SDPA,
AdamW LR2e-5 constant,wd.01,clip1. Exactly512updates/arm with evaluation0/128/256/384/512.
4,096consumed unique questions/arm (2,048new binding,1,024original,1,024curriculum).
Same cached parent logits are reused from the pinned coverage fit: TRAIN rows,
full content/order, shape and parent hash checked; initial KL must be zero.
Two-update TRAIN-only smoke runs in a separate process before full fits. Do not
reuse smoke weights or optimizer. No extensions, seed retries or coefficient sweep.

Each process:1800s,8GiB allocatedVRAM,5GiBhost,512MiBoutput,2GiBfree disk minimum.
NaN/Inf, no gradients, frozen changes, hash/resource failures stop INCOMPLETE.
Savebest/final separately, all update losses/gradients, fixedTRAIN and validation
curves, consumption, raw scores/probabilities, environment, commit and hashes.
No optimizer/RNG checkpoint: do not claim exact resume.

## Selection and interpretation

Keep original/curriculum gates relative to step0: every taskAccuracy drop<=1pp,
sourceNLL/Brier/ECE increase<=.02/.01/.02. Also require>=99% Accuracy on legacy
binding validation, preserving the earlier KL512 capability. Among eligible
trained checkpoints, lowest new binding validationNLL wins; earliest tie.
Step0 remains experiment-local fallback even when legacy binding is below99%;
fallback is not adoption or replacement of the retained KL512 reference.
Selected reload must exactly reproduce all saved selection/fixedTRAIN arrays,
and a fresh process replays the selected outputs for both arms.

After BOTH fits/selection complete, evaluate their fixed512 and selected weights
(deduplicate identical checkpoints), plus the retained KL512 reference on the same
predeclared post-selection conditions. Never select from these transfer scores.
Primary causal comparison is contrast512 minus CE512. Report all fields and
Accuracy/NLL/Brier/ECE, changed-query pairs, minimum field, order/no-question controls.
Use paired whole-state bootstrap10,000draws seed17, percentile95%linear for Accuracy/NLL.

Mechanism support requires BOTH focus+newdescription and complement+newdescription
Accuracy gain>=3pp with lower paired interval>0, lowerNLL, no higherBrier,
ECE increase<=.02 and final512 retention eligibility. Mixed evidence is UNCERTAIN;
no benefit is NOT SUPPORTED WITHIN THIS DESIGN/BUDGET, not a capacity ceiling.
Quality admission additionally requires every field>=95%, pairbothcorrect>=90%,
known-form>=99%, and independent selected output verification. Neither threshold
is full four-modality qualification. Report improvement separately from admission.

Close this fixed two-arm comparison after results. If transfer still fails, do not
rescue with the same validation by more templates, ratio/step tuning or seeds.
Reassess conditional language representation within the whole200M budget before
another semantic-learning intervention. Native grounding/temporal/joint quality,
optional conditions, reusable observations, JSON and post-quantization quality remain open.
