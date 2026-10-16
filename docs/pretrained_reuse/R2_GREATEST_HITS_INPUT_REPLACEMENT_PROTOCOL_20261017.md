# Prospective AV input replacement diagnostic

Purpose: detect whether a fixed selected model's output responds to audio and
video content. This extends the matched learning protocol's requested input
sensitivity diagnosis. It does not modify training, checkpoint selection or data.

Use all 8,875 existing Greatest Hits development-validation questions. For each
question choose one donor with the identical task/question/options from another
validation recording date. No TRAIN, calibration, sealed or official-test donor.
Selection uses only seed17, IDs, task and date; no target labels or predictions.
Sort donor dates and IDs, select the date by SHA256([17,"date",question ID]) modulo
eligible dates, then select the donor by SHA256([17,"donor",question ID]) modulo
that date's task pool. No redraws or score-dependent removal. Missing eligible
donors fail the plan. Preserve every question and report donor reuse/date balance.

After model selection is frozen, use the same inventory for all five selected
arms, once per arm. Keep the original question/candidate vectors and every option.
Evaluate natural inputs, self audio replacement, self video replacement, donor
audio replacement and donor video replacement. Replace both speech/event-audio
blocks together. Preserve each selected block's saved values, positions and masks;
do not manufacture cross-recording synchronization. Self replacements must reproduce
natural logits exactly; they test transport rather than quality.

Save ID, donor ID, view, raw logits and candidate gate values. Report paired
prediction flips, probability total variation, Jensen-Shannon divergence and
confidence change for donor replacements. Report task/date/parent counts and
aggregation alongside overall values; paired questions share recordings and are
not independent trials. Gate changes describe the added residual only.

The donor combinations are deliberately mismatched observations with no verified
counterfactual truth. Do not compute their Accuracy/NLL/Brier/ECE against the
original target or call a changed output correct. Source target agreement may be
described as metadata but cannot validate a mixed observation. Natural-input
quality remains in the unchanged matched learning evaluation. Sensitivity alone
does not establish useful complementary reasoning or robustness to real corruption.

Do not select checkpoints, modify training, tune donor matching, extend training
or promote a product using this diagnostic. A later justified experiment requires
a separately frozen design. This inventory is fixed before inspecting matched-run
validation results. The current stage generates metadata and verifies transport
mechanics; trained-model scoring remains pending full cache and matched learning.
