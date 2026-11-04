# Fixed parent-probability retention comparison

The preceding512-update mixed fit learned new semantic binding but every trained
checkpoint failed the frozen old-task retention gates. This is the second
prespecified comparison on this binding corpus. Test whether one fixed output
retention regularizer preserves old decisions while learning binding. A failure
closes coefficient/LR/step searches on this corpus; do not tune a third variant
against these observed validation results.

## One change and preserved control

Control is the completed `condition-binding-fit-01` at source37ede80, result SHA
`d58df7c1e6bb4cda0f63b8e92554699f40e53baef65d53037fad583434684cb4`.
The new fit starts from the same coverage-parent weights, not that control's
rejected final weights. Identical seed17, architecture,10,647,169trainable /
22,713,601total parameters, fixed TRAIN/replay rows and order,4+4questions/update,
512updates, AdamW LR2e-5constant/weight_decay.01/clip1, FP32SDPA/TF32off,
dropout disabled and4CPUthreads. Embeddings/pooler remain frozen. No new media,
source, architecture, parameter, seed, step budget or option-count change.

Original objective: `0.5 * binding_CE + 0.5 * replay_CE`.
New objective: `0.5 * binding_CE + 0.5 * (replay_CE + KL(parent || student))`.
KL temperature1, coefficient1 fixed before running. Each CE and KL is a mean
over questions, with each question normalized over its actual candidates.
Teacher probabilities are detached. No padding mass, reverse-KL or extra label
replacement. This adds regularization strength to replay; it does not isolate
probability information from loss-weight changes through a weight-matched CE
control. It is retention from the same parent, not a stronger general teacher,
whole-embedding distillation or extra runtime model.

Parent logits are computed before any update, only for the2,048original TRAIN
replay questions, using their actual four-question training batches. Bind cache
entries to complete question/state/candidate-order hashes, parent delta hash,
source/code pins and all512sampler positions. Every row occurs once. Frozen
parent inference has no dropout; reset Torch seed17 before updates. No teacher
gradient or weights update occurs during caching. Record cache time/forward count
separately from optimization consumption. Extra teacher inference is an explicit
cost difference. Never generate retention targets on validation or transfer.

Reuse immutable `condition-binding-corpus-01` result SHA
`957e2ce3ade65c5d2caa26a1cbb0e95a3adeee5cf5b312110f8c2a7e3cfed971`.
Do not rebuild or alter it. Binding selector1,024questions/256states, full old
original2,540 and curriculum2,048validation, fixed256binding TRAIN probe. All
IDs/order/preprocessing remain the same. At step0 require exact raw-score parity
with all four control baseline arrays before any optimization.

## Selection, boundaries and measurements

Evaluate0/128/256/384/512. Same selector/minNLL/earliest tie and step0fallback:
every old task Accuracy decrease <=1pp; each old source NLL/Brier/ECE increase
<=.02/.01/.02. No performance early stop. Save all history and probabilities,
selected/final weights separately, train/validation metrics, consumed IDs/states,
source/task counts, CE/KL/gradient norm/LR/time/memory. Replay corpus omits old
curriculum TRAIN as in the control; do not conceal that limitation.

Use a separate TRAIN-only2-update smoke. Verify finite nonzero trainable gradients,
frozen weights, correct cache alignment, zero initial KL within floating-point
tolerance, and old-code/source integrity. Full resources remain<=1,800s,
8GiB CUDA allocation,5GiB host,512MiB output with2GiB disk reserve. A safety
failure means incomplete, not hypothesis rejection. New distinct directories;
optimizer reset and no claim of exact training resume.

Fresh-process selected inference must exactly reproduce all four saved selected
validation/fixed-TRAIN score arrays and12reload payloads. No repeat training.
**No transfer condition is parsed or scored in this comparison.** Old transfer
files may be checksummed as part of the immutable corpus but are not selection
signals. They are observed development regressions, not a new blind audit. If a
new checkpoint passes retention and materially improves binding, independently
specify an unseen transfer family in a later protocol before testing it.

Report parent/control/new learning curves and all eligibility failures, plus
state-paired10,000draw bootstrap intervals (seed17, percentile95%) on binding
validation. These are observed development comparisons with selection bias,
single-seed and no multiple-comparison correction. Neither a narrow100% nor
retention success establishes native four-modality95–98% or product adoption.
No sealed audit, cloud spending, legacy artifact overwrite, main merge,
quantization or new product Teacher.
