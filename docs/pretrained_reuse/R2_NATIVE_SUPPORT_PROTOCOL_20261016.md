# Fixed-budget positive support comparison

Purpose: test the audit's concrete finding that the sampler omitted available
TRAIN positives, while keeping the lightweight Omni model and full objective.
This is one predefined sampling policy; it is not a search for an optimal ratio.
The user has authorized autonomous local experimental work. No paid compute,
live-process intervention, calibration or sealed-audit payload access is used.

## Fixed conditions

Control: completed `native-long-budget-fit-01`, receipt
`645bedbe73ed508fad68c4f636a41df6c8630cd996641c44ce2a87d31550250b`.
Reuse its saved predictions; do not retrain the control.
Candidate: fresh seed17 original readout weights, fresh AdamW, same frozen corpus,
encoders and native features. Readout889345, total196254874 prequantization
parameters. Same8 microbatches/update,32768 updates, LR0.001/warmup62/cosine32768,
weight decay0.01,clip1,CE,option permutation rule, and hierarchical macro-NLL
best selector (`min_delta=1e-5`, earliest ties, initial0 eligible).
No performance early stop or outcome-dependent extra seed/budget.
Evaluate at0,512,1024,2048,4096,8192,12288,16384,24576,32768.
Full validation24346 IDs/order/candidates/preprocessing remain unchanged.

## The single intervention

Replace negative STARSS questions with unused positive TRAIN questions **within
the same exact event/question/options/query-vector condition**. Keep every old
positive. For an event with `n` unique consumed questions, the positive floor is
`min(ceil(n/2), number of available TRAIN positives)`. Existing above-floor support
is retained, including speech70/85. Choose new positives deterministically using
room/recording coverage and seed17 hashes; remove negatives by a separate seed17
hash. The identity replacement is applied to every old occurrence of that row.

Keep total6656 unique questions and262144 presentations, per-event quotas, each
row's repetition multiplicity, all occurrence IDs and **every non-STARSS row at
its original update/microbatch position**. Binary candidate shuffling still uses
the old hash rule; replacement row IDs change their own permutation hashes.
No new media, label editing, modality reweighting, crop/frame change, encoder
update or readout change. This estimates the combined support/diversity/label-prior
policy effect; it cannot attribute gains solely to one of those properties.

Frozen dry-run:292 unique substitutions,9344 of262144 presentations; STARSS
positives110→402 among1024 unique questions. Non-STARSS and speech unchanged.
Clapping0→9,telephone0→27,water tap0→28,knock0→10 unique positives.
Bell remains0 because the entire admitted TRAIN corpus has none. Water tap has
no validation positives. These are explicit unmet gates, not discarded tasks.
Source/room/recording support is saved for every event in the frozen plan.

## Measurement and decisions

Always retain all four modalities' Accuracy/NLL/Brier/ECE, macro/minimum metrics,
all twelve STARSS classes' positive counts, recall, specificity/balanced Accuracy,
Lombard retention, unique coverage, loss, gradients, LR, VRAM and wall time.
TRAIN probe is all6656 candidate-consumed questions; its STARSS membership differs
from the control. Do not call those two TRAIN averages a matched-question contrast.
Also save a fixed6948-question union of the original pool plus292 new questions;
replay both parent best/final weights on that union, matching their original
6656-question logits exactly. Record candidate final on the same union.

Primary quality comparison: candidate selected by the **unchanged validation
macro-NLL rule** versus control selected8192. Also report equal-budget8192 and32768
and full curves, including failed points. Diagnostic target is improved non-speech
recall/balanced Accuracy across every class where both validation labels exist;
report undefined classes explicitly and keep bell included. Recall gains alone
do not establish an improved calibrated Omni model.

A candidate is worth retaining for further research only if its selected macro
NLL improves, its non-speech macro balanced Accuracy improves, and no modality or
Lombard Accuracy declines by more than2 percentage points. This2pp is an engineering
retention flag, not permission to weaken the95–98% product gates. Report exact
trade-offs even if the flag passes. No product promotion from this development
comparison. If it fails, preserve it and move to the independently justified
field-of-view comparison; do not search seeds/ratios against validation.

For STARSS overall Accuracy/NLL/Brier differences, use paired recording-cluster
bootstrap5000 draws,seed17,linear percentile95% intervals. State selection/development
bias and small class counts. Validation is observed development data, not blind.

## Execution and protection

New config `native_support_v1.json`, new runner `train_native_support.py`, separate
`native-support-{smoke,fit,replay}-01` outputs. CPU tests prove fixed quotas,
independent TRAIN membership, no duplicate-positive fabrication, untouched slots,
unchanged loss/optimizer/selector AST and config budget. Separate32-update smoke
precedes full fit. Full fresh-process best/final replay is mandatory.
Pin all consumed files and Python source bytes; freeze source until replay ends.
Keep best and final separately. No resume interface is implemented; saved optimizer
and RNG state are not a claim of exact resume support. Stop safely on nonfinite
numerics, resource/integrity failure; do not call an incomplete run a quality failure.
