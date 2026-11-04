# Independent conditional transfer, fixed weights

This bounded extension of the existing binding evaluator is authorized by the
user's autonomous local research instruction. Freeze this protocol, templates,
code and corpus before model inference. No new fit, calibration, model selection,
prompt search or seed search. Preserve every previous artifact and sealed audit.

## Question and evidence

The retained parent-KL checkpoint at step 512 passes the previous development
retention gates and scores 100% on a narrow semantic task. Determine whether
that fixed candidate transfers its conditional routing to unused combinations
and two new instruction families, including changed candidate descriptions.
The parent coverage checkpoint is the only comparison model. Both are the same
22,713,601-parameter Cross-Encoder with different saved encoder/classifier deltas.

Use 256 states selected by SHA-256 ordering, fixed seed 17 and the label
`independent-transfer-v1`, from the 10,000 possible four-field states. Exclude
all upstream TRAIN combinations, old diagnostic oracle/predicted states, and all
previous binding train/validation/transfer state identities. Read old state
partitions only for exclusion, not old transfer questions/metrics. Fail if there
are insufficient unique states. Every state keeps four questions and 40 options,
including ambiguous value names shared across different fields. Do not shrink
choices, remove difficult cases or redraw the cohort after scores are seen.

All new rows are development transfer with generated annotation states, not media
or a product audit. Individual attribute values are known; their combinations
are excluded from prior corpora. Exact natural-language templates are fixed in
`scripts/binding_transfer.py`. New does not mean linguistically diverse: there is
one paraphrase and one exclusion instruction per field. English only.

## Eight conditions on identical targets

1. `known`: existing training-style question/option interface, new states.
2. `question`: new paraphrase, existing option descriptions.
3. `options`: existing question, new full-sentence descriptions.
4. `joint`: new paraphrase and new descriptions.
5. `distractor`: ignore the other three attributes and request the target field;
   existing option descriptions. All three distracting fields are mentioned.
6. `distractor_options`: same exclusion instruction and new descriptions.
7. `no_question`: empty condition with new descriptions. Same state and choices
   must produce the same distribution for all four labels; Accuracy <=25%,
   condition-pair and all-four Accuracy zero.
8. `reverse_joint`: reverse every candidate in `joint`, keep IDs/targets aligned.
   Compare probabilities by identity (max difference <=1e-5, same argmax ID).

No condition is selected as a better prompt. Report all eight even if unfavorable.
Primary transfer conditions are `joint` and `distractor_options`, fixed in advance.

## Decision rule

To advance the candidate on this family, BOTH primary conditions need each field
Accuracy >=95%, condition-pair Accuracy >=90%, aggregate improvement >=3 points
over the parent with a positive whole-state bootstrap lower bound, lower NLL,
Brier no higher, ECE increase <=0.02. `known` must retain >=99% Accuracy.
These research gates do not replace any product goal or imply global non-inferiority.
Order/no-question violations are execution/integrity failures, not poor transfer.

Record Accuracy/NLL/Brier/ECE, mean confidence, every field, all-four and changed-
question pairs. Compare matched model outputs with 10,000 whole-state bootstrap
draws, seed 17, percentile 95% linear quantiles. The two predeclared primary arms
are both required, but no broad multiplicity or multi-seed claim is made.
Positive results apply only to this generated family, not native/OOD Omni quality.

If known conditions work but new wording fails, prioritize conditional-language
transfer. If only changed descriptions fail, inspect candidate semantics. If both
transfer, next compare oracle versus perceptual states on independent real inputs.
Never tune these two checkpoints or rerun alternative templates on these results.

## Execution and safeguards

Separate TRAIN-state smoke, then one full inference comparison, then one fresh
process full replay. Smoke uses two old binding TRAIN states and never scores
the new transfer set. It checks actual model loading, frozen tensor hashes,
finite logits, time, tokens and resource feasibility; no training is performed.
The smoke's known-interface scores must exactly match the saved parent-step-0
and candidate-step-512 fixed-TRAIN score prefixes from the previous fit.

One model resident at a time, local offline base. Fixed FP32 SDPA, TF32 off,
dropout disabled, four CPU threads, deterministic Torch; four questions/batch,
all actual 40 options. No truncation (fail above 512 tokens). Both deltas must
have the exact expected encoder/classifier keys; all parameters frozen, no
optimizer, model hash unchanged before/after every arm. Save full scores,
probabilities, per-question IDs/targets/predictions and summaries, environment,
input/code/corpus hashes, time and peak VRAM/host use. Replay all native arrays
and deterministic metrics, including order diagnostics, exactly.

Per process ceilings: 1,200 seconds, 4 GiB CUDA allocation, 4 GiB host,
256 MiB outputs, 2 GiB free-disk reserve. Stop on resource/numerical/integrity
failure without altering cohorts or precision. Outputs use new directories.
No GPU overlap, paid compute, teacher changes, quantization or product promotion.
