# R2 instruction diversity — fixed one-factor comparison

## Question and limits

The fixed KL512 candidate reached 100% on new state combinations with known
questions, but only49.22% on independent question+description changes and9.47%
on exclusion instructions+descriptions. This motivates a single instruction
diversity intervention, not an encoder/capacity increase. Oracle structured
states isolate conditional reading; this is not native Omni quality.

## Fixed training comparison

Control: `condition-binding-retention-fit-01`, completed seed17/512updates.
Candidate: `binding-language-fit-01`, separate process/output; initialize from
the same coverage parent (`text-coverage-training-01/best-delta.pt`), never the
control's final weights. Dense6encoderblocks+classifier:10,647,169trainable /
22,713,601total parameters; embeddings/pooler frozen; FP32 SDPA dropoutoff.
Same2,048 binding questions/512states,2,048original replay/1,815states, same
sample-order indices, one use per question,40binding options, correct labels,
candidate descriptions, seed17,512updates,4+4questions/update, AdamW LR2e-5
constant/wd.01/clip1. Loss `.5*bindingCE + .5*(replayCE + KL(parent||student))`,
coefficient/temperature1. Exact verified original-TRAIN parent cache is reused;
this reduces cache work, not training. Initial replay KL must be zero.

Only binding question text changes. Eight training syntax families assign
64states/256questions each, every state with all4fields, in frozen row order:
original question; report target; explicitly requested attribute; target not
distractor; exclude the other3 without naming target; target then withhold
others; reject one then switch target; select named attribute among all4.
Distractor order uses deterministic state/field hash. No extra state/asset or
question repetitions are introduced. Original metadata/targets/options remain
unchanged; new ID, underlying ID and family metadata record the intervention.
The exact literals are frozen in `scripts/binding_language.py` before execution.

Selection remains original known-binding validation NLL among checkpoints
passing every old-task retention gate (each taskAcc decline<=1pp; each original/
curriculum NLL/Brier/ECE increase<=.02/.01/.02). Parent step0 fallback; earliest
tie. Evaluate0/128/256/384/512, no quality early stop. Savebest/final separately;
raw-score selectedreload and freshprocess replay required. Fixed TRAIN probe
changes question text and is logged separately, not expected equal to the
control's TRAIN probe. All3step0 validation arrays must match the control exactly.

## Independent post-selection development evaluation

Freeze256newstates before training, excluding union of allupstream TRAIN,
semantic diagnostic, original binding train/validation/transfer, and previous
independent transfer states. SHA order seed17/label`language-heldout-v1`.
Four fields/state;1,024questions/condition,40typed options. Seven conditions:
known; focus; complement; focus+new descriptions; complement+new descriptions;
reverse focus+descriptions; noquestion+new descriptions.
Focus: `Relevant: {target}. Irrelevant: {others}. Select the matching value.`
Complement: `Among the four recorded properties, answer for the one outside
this list: {others}. What is its value?`
These syntax families and new descriptions are absent from training. The
concepts are trained; individual values are known; combinations are untouched.
No arbitrary-language claim: only two heldout English syntax families, generated
development tasks, not sealed/blind product audit. Trainer reads only newTRAIN,
oldreplay/validation and manifest; heldout rows are not loaded during selection.

Primary comparison: oldfixed512 versus newfixed512 on all identical rows. Also
report selected candidate if different; never select using these results.
Predeclared quality gates for BOTH focus+descriptions and complement+descriptions:
eachfieldAcc>=95%, pairbothcorrect>=90%, aggregate gain>=3pp over control with
positive lower pairedstate CI, lowerNLL, nohigherBrier, ECEincrease<=.02.
KnownconditionAcc>=99%. These are research gates, not four-modality product
completion. Report improvement and qualitygate failure separately. Original
retention remains mandatory for adopting the selected model.

Bootstrap10,000whole256-state samples,seed17,percentile95%linear; same sampled
groups for paired scores, all4questions together. Accuracy/NLL interval, all
fieldAccuracy/NLL/Brier/ECE, overall/pair/all4 metrics and raw predictions saved.
Order control compares native selected option IDs and aligned probabilities;
noquestion distributions must be equal for all4queries of a state with <=25%
Accuracy, zero correctpairs. Fullfreshprocess inference must reproduce all
deterministic arrays/predictions/metrics/effectiveconfig. No hyperparameter,
template, threshold or seed rescue on this set after results are observed.

## Execution and accounting

Local RTX3080Laptop only; no cloud. Separate two-update TRAIN-only smoke before
one fullrun; smoke updates are discarded. Allcode/config/input hashes frozen
and checked; existing artifacts read-only. Guards:1,800s/process,8GiBallocated
VRAM,5GiBhost,512MiBoutput,2GiBmin freedisk. NaN/Inf/frozen changes stop incomplete.
Measure walltime, evaluation/update time, memory, input lengths/truncation,
sample/state/family counts, gradient norms and exactrevision/environment.
No optimizerstate is saved; no exactresume claim. Oldsource truncation policy
unchanged, newbinding truncation prohibited. Learning curve reports selected
and full consumption separately. Failure results remain; no second fit is
authorized by this protocol. No sealed audit, media ingestion, quantization,
Teacher promotion or main merge.
