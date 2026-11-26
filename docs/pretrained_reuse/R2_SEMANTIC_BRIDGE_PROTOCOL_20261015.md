# Fixed TRAIN-only semantic-state bridge diagnosis

Prospective continuation of STRATEGY_20261015.md. This is inference-only and
does not create a product candidate, tune a prompt or measure generalization.
Preserve every original corpus/checkpoint and all previous evaluation tasks.

## Question and fixed cohort

Can the retained frozen text Decision model select a typed fact from a correct
state when the question changes? How much does replacing that state with the
frozen image model's predictions change the result?

Select256 of8,424 original TRAIN images by ascending SHA256 of UTF-8
`17:<image-id>` (ID as tie-breaker), without balancing/quality filtering.
Smoke uses only the first4 of the same order in a separate process. All selected
images remain, including those the image model misclassifies. No validation,
calibration, reporting or sealed-audit examples are loaded. This is a diagnostic
on images used to train the image head, not an independent evaluation set.

For each image retain the same40 typed options in existing taxonomy order:
10color,10texture,10count,10shape. ID `texture:star` differs from `shape:star`;
description is `texture: star`, etc. Every image receives these four questions:

1. What is the color of the objects?
2. What is the texture of the objects?
3. How many objects are there?
4. What is the shape of the objects?

The four typed targets are distinct even when their untyped words coincide.
State is sorted compact JSON of four string facts, no ID/target/question/source.
The unchanged text interface is `Question: <q>\nCandidate answer: <description>`
paired with `State: <json>`. No prompt variants or truncation are allowed.

## Conditions and actual model calls

- Oracle: four original TRAIN annotation facts; explicitly privileged information.
- Predicted: four argmax labels from the retained0.01 affine image head, computed
  from immutable native cached features. Labels never enter this input function.
- No evidence: `{}`, the same four questions; compute4x40pairs once and broadcast.
- No question: oracle state with empty question; compute256x40pairs and broadcast
  each answer over four tasks. Question-free Top1 cannot exceed25% because each
  image has four distinct correct IDs.
- Donor state: cyclic next-image oracle scores, with the same question/options.
  This is exact input-equivalent score reuse, not new model inference; report
  both original and donor-corrected labels. A donor may share individual facts.

Full unique pair forward presentations:92,320 (81,920 main +10,240 no-question
+160 no-evidence). Logical rows include broadcast and donor controls; distinguish
them from actual inference. Smoke:1,600pairs. No fitting or temperature selection.

## Immutable sources and execution

All exact paths/hashes live in
`configs/pretrained_reuse/semantic_state_bridge_diagnostic.json`.
Text base: `cross-encoder/ms-marco-MiniLM-L6-v2` revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`; retained coverage-parent delta
`b063f124...`. Its old config name mentions L12; actual six loaded layers and
22,713,601parameters are required. All weights frozen, eval, FP32/SDPA, CUDA,
TF32off, seed17,4CPUthreads,64candidate pairs/batch, max512tokens, offline load.

Image head: Linear1536to40FP64CPU, `0.01/final.pt`, full4x10logits. Reconstruct
input as global768 plus mean of16spatial768cells (mean in original FP32 then
convert to FP64). Selected logits must agree with the existing full-training
logits within1e-10 absolute and have identical argmax. The actual spatial cache
config says `single_frame: true`; do not substitute the older8repeated-frame
image configuration. VideoPrism is not loaded again in this diagnosis.

Freeze code/config before smoke/full. Record commands, commits, source hashes,
source revisions, installed packages, device/backend/dtype, parameter counts,
weight digests before/after, token lengths, wall time and peakVRAM/host memory.
900s/4GiBVRAM/4GiBhost/128MiBoutputs and1GiBfree-disk guards. Existing installed
packages remain. Separate fresh-process replay must reproduce selected IDs,
native/text scores, targets and metrics; do not hide a failed replay by rounding.
Read source pins before/after; write only new output directories.

## Measurements and interpretation

Save all raw scores, stable log-softmax, probabilities and typed targets/IDs.
Report per-question and macro Accuracy/NLL/Brier/ECE15, confidence, all-four
correct, six changed-question pairs both-correct, field-routing accuracy and
within-correct-field10choice accuracy. Original native10choice accuracy is a
separate baseline; never compare40choice Accuracy directly with old10choice
benchmarks. Record perception/Oracle/predicted-state error intersections.

Oracle failure points to a downstream semantic/interface problem **for this
fixed probe**, not to encoder capacity. Oracle success with predicted failure
points to input errors or their propagation. Top1fact extraction loses uncertainty;
serialization and40choice transfer are new to this text model. Neither outcome
identifies a universal model ceiling or causally separates every representation
and domain-shift effect. No new prompt/model is selected from these outcomes.

Close this one diagnosis after the fixed run and replay. Next work must address
the observed gap with a prospective independent task/data contract; no repeated
wording or coefficient search. Four-modality95–98%/<=200M product gates stay open.
