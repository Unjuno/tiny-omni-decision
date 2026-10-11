# Text query and truncation diagnostic

This post-hoc audit uses existing validation predictions; no new training,
checkpoint selection, corpus modification or sealed evaluation occurred.

| Type | Count | Accuracy | Uniform expected Accuracy | NLL |
| --- | ---: | ---: | ---: | ---: |
| choice | 1008 | .4722 | .2276 | 1.2079 |
| noul | 987 | .5643 | .5000 | .6775 |
| score | 545 | .4055 | .2190 | 1.3406 |

Exact instruction overlap is 394/2540: seen instructions score .4797,
unseen exact wording .4963. Different option-count distributions and tasks
confound this comparison; unseen wording is not unseen semantic task identity.
No conclusion that memorization is absent follows.

Actual formatted contexts were tokenized with the pinned local MiniLM tokenizer.
1005/2540 (39.6%) exceed the runner's explicit 256-token truncation limit; the
maximum is 868. The runner appends the question after the state, making question
loss a plausible input bottleneck. This audit has not yet measured which query
spans are lost or their causal effect on predictions. Tokenization only was run;
no overlength tensor was sent through the model. Increasing length alone must
respect the encoder position limit, and would be a new controlled experiment.

Raw compact records are R2_TEXT_QUERY_STRATA_20261011.json and
R2_TEXT_TOKEN_LENGTHS_20261011.json. External audit artifacts remain under
C:/CodexArtifacts/pretrained-reuse-r2/text-query-stratification-01/.
Next: inspect retained state/question coverage before changing capacity or training.

## Measured question retention

The actual fast tokenizer's offset mappings with right truncation and
max_length=256 show: full question retained 1,535; partial question 79;
no question-body tokens retained 926 (36.46%). Per-ID retention details are
saved in C:/CodexArtifacts/pretrained-reuse-r2/text-query-retention-audit-01/.
The compact summary is R2_TEXT_QUERY_RETENTION_20261011.json.

Saved Accuracy/NLL: full .5081/1.0055, partial .5570/.9600,
absent .4644/1.0774. These groups differ in tasks and lengths; this is not a
causal ablation. Full-question performance also remains weak.

Bounded proposed correction: preserve the full question in a separately
budgeted tokenizer pair, truncate only state, fail closed if the question
itself exceeds the budget, and keep the original runner's default unchanged.
Compare one new seed-17 run using the same model, split, head, optimizer,
epoch budget and checkpoint rule. Record truncation and retained-state counts;
this tests question-preserving serialization, not pure length or capacity.
No such corrected training has been executed yet.

## Controlled question-preserving run executed

Source commit 89b14e9 adds opt-in `--context-policy question_pair`; the default
legacy tokenization remains unchanged. Question is sequence one, state is
sequence two, and only sequence two may be truncated. Over-budget questions
fail closed. Actual tokenizer checks retained every question token in all
23,319 train and 2,540 validation examples. Ruff and CPU suite passed
(311 passed, 1 skipped).

The separate `C:/CodexArtifacts/pretrained-reuse-r2/text-question-pair-v1` run
completed with seed 17, 8 epochs, batch-questions 64, feature-batch-size 64,
the same pinned MiniLM, source splits, optimizer and readout. The command is
the original `scripts.train_frozen_text_probe` invocation with this new output
path and `--context-policy question_pair`. Full run report and epoch history:
R2_TEXT_QUESTION_PAIR_20261011.json. Existing baseline artifacts were unchanged.

| Selected epoch 3 | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| Legacy | .4937 | 1.0303 | .5946 | .0268 |
| Question pair | .4882 | 1.0279 | .5937 | .0292 |

Question preservation did not yield a substantial aggregate improvement:
NLL/Brier improved slightly, Accuracy/ECE worsened. No product-quality claim
or further seed search follows. This changes order, segment representation and
retained state as well as question retention, so it is not a pure causal test
of dropped questions. The remaining limitation cannot be assigned solely to
truncation. Both runs use development validation for checkpoint selection.
