# Fixed DeBERTa semantic adaptation, before the first evaluation

One seed17 local comparison: pretrained step0 versus one adapted recipe. It
tests learning on the fixed coverage/retention contract, not an isolated effect
of architecture, pretraining, auxiliary labels or a comparison with old MiniLM
scores from a different interface. No product adoption follows from this run.

## Immutable inputs and initialization

Use the existing decision-supervision-corpus-01 manifest
`f2fac1bb77e762748e9ce49077009bb69a50d7c2869b93c27de6ffbeb7c55d17`.
16,384 new TRAIN questions (8,192 binding, 8,192 numeric), 2,048 existing replay.
Selection has3,072 new and5,612 retained questions, all31 exact schema slices.
No transfer/audit payload is opened by fit, smoke or selected-checkpoint replay.
No existing dataset, model, experiment producer or artifact is changed.

Pinned DeBERTa xsmall revision262ae02f29173eec1c250f90804dc7edc677dcff,
pretrained initialization, no inherited adapter/delta/optimizer state.
Actual70830722 parameters: freeze49191168 embedding parameters; train21639554
in `deberta.encoder.*`, `pooler.*`, `classifier.*`. Full FP32, eager attention,
dropout disabled (`eval` with gradients), deterministic algorithms, TF32 disabled.
No new model parameters. Each trainable delta is approximately86.56MB of tensor
payload; serialization overhead is measured. No multimodal runtime is implied.

## One fixed recipe

Public input pair for every task, including prose legacy states:
`Question: <question>\nCandidate answer: <supplied option>` and `State: <state>`.
No task ID, source, oracle, rule, group, private target or supervision enters the
model. Score native entailment logit0; normalize across ALL supplied options.
This rendering differs from the prior frozen factorization screen. Step0 and
all new checkpoints share it. Do not attribute old/new differences solely to fit.

Token limit512. Preserve question/candidate; truncate only the state side on
legacy rows. Record every affected ID and token lengths; keep every evaluation
row. New programmatic rows must fit without truncation. Legacy quality is
therefore quality of this explicit possibly-truncated interface, not full-context
competence. Fail on an overlong question/candidate; do not silently shorten it.

One epoch,1024 optimizer calls. Each update contains8 binding +8 numeric +2
legacy questions, shuffled deterministically within/between buckets. Each
question occurs exactly once. Backprop one question's full candidate CE at a
time, divide by18; replay adds coefficient1 KL from this same step0 model's
cached option distribution. No other auxiliary loss. Fact/relevance labels are
used for error attribution, not additional forward paths. They provide hard
true-but-irrelevant candidates through the unchanged full candidate set.

AdamW betas(.9,.999), eps1e-8, weight_decay.01, clip norm1, peak LR2e-5,
31-update linear warmup, cosine to0 at update1024. Thus1024 optimizer calls,
1023 positive-LR calls; final zero-LR call consumes18 questions but changes no
weights. Explicitly retain that distinction. No performance early stopping or
budget/ratio/seed/wording search. NaN/Inf, frozen updates, hash failures, OOM or
resource limits terminate as INCOMPLETE, never a quality failure.

Separate-process smoke uses TRAIN only and2 scheduled updates, plus a backward
probe on the longest TRAIN candidate pair's whole question. No smoke metric is
used for hyperparameter selection. Main run reinitializes model, optimizer and
all random states; smoke weights are never loaded. Frozen source/code hashes
are verified before and after execution. Effective config includes environment,
training scope, ordering, token policy, input hashes and all exact code pins.

## Evaluation, checkpoint selection and uncertainty

Evaluate step0,256,512,768,1024 with identical input IDs/order/preprocessing.
Also evaluate a fixed TRAIN subset: first8 IDs per training metric slice in
SHA256(seed,ID) order. Report its gap descriptively: train and selection wording
distributions differ. Rolling train loss is not a fixed-example generalization gap.

The existing31-slice selector remains authoritative: each slice must retain
Accuracy within1pp and NLL/Brier/ECE within+.02/+.01/+.02 of step0. Choose
minimum equal-slice macro NLL among eligible checkpoints, including step0,
earliest step on ties. Save every scheduled checkpoint, best and final separately.
Export probabilities, IDs, targets and predictions for every evaluation.

Research advancement additionally requires>=3pp gain in the minimum Accuracy
of all16 NEW slices and a positive lower95% interval for its paired improvement.
Bootstrap10,000 times with seed17, whole states stratified binding/numeric,
all questions per sampled state kept, minimum recomputed every replicate;
percentile95%, linear quantiles. Preserve all31 retention requirements.
Selection/validation reuse and fixed templates limit this inference; no blind
quality claim. Report the endpoint as well as the selected checkpoint, even
when selection stays0. Native95–98% and complete<=200M gates remain open.

Reload the selected delta against the same pinned pretrained model and require
exact saved scores/IDs on all selection and fixed TRAIN rows. Repeat selected
inference in a fresh process with the same code/input/weight pins. No retraining
for replay. No claim of exact training resume: optimizer/RNG state is not saved.

## Resource and next-action contract

Use the existing RTX3080Laptop16GB only; no installation/download/cloud charges.
Limits in `configs/pretrained_reuse/decision_semantic_fit_v1.json` include4GiB
minimum free disk,14GiB peak CUDA,12GiB host,2GiB new output and4h wall time.
Save elapsed/train/evaluation time, peak memory, consumption/unique states by
source, all attempted outcomes and artifact hashes. No native media is consumed.

Freeze source before smoke. Measure actual throughput before starting the one
main run; if it violates the resource envelope, report the limitation without
silently reducing the data/evaluation. Following the outcome, use the strategy's
oracle-state/native-input comparison to select a single next intervention.
