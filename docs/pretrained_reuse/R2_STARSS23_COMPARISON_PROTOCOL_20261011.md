# STARSS23 development comparison preregistration

The fixed configuration is `configs/pretrained_reuse/path_a_starss23_events.yaml`.
It is registered before any validation readout scores. No readout has been fit
yet. All 78 recordings, all complete five-second intervals, all twelve event
queries and the frozen room splits are retained. Previously exposed smoke
rooms remain training-only. This is a narrow event-presence development task,
not general natural-language AV reasoning or a blind audit.

## Matched training conditions

Train audio-only, video-only and audio+video readouts with the existing shared
candidate MLP. Each receives 384 audio dimensions, 768 video dimensions, twelve
one-hot task/event IDs and two one-hot option IDs. Use the same 1166-dimensional
input and 64-unit hidden layer for all three conditions; zero the absent
modality after train-only standardization. This keeps head parameter counts
equal. Event IDs represent this fixed query vocabulary, not a new pretrained
language encoder. Both media encoders remain frozen.

For each condition use seed17, AdamW at 0.001 with weight decay0.001, 24 epochs,
128 queries per batch and a seeded permutation of all train queries each epoch.
No class balancing/resampling or new random seeds. Select the earliest epoch
with minimum uncalibrated validation macro-event NLL. Save all epoch summaries,
best weights and final weights separately. Validation supports selection and
therefore its final numbers are development estimates.

After selecting each checkpoint, fit one scalar temperature using only the
calibration room: seventeen geometric candidates from0.25 to4. Select minimum
calibration macro-event NLL with deterministic first-in-grid tie handling.
Validation never fits temperature or normalization. Report calibrated and raw
metrics and unsupported classes. The calibration room and two validation rooms
are far too few to establish product generalization.

## Controls and reporting

Compare the same windows against a Laplace-smoothed train-event prior. For the
fixed joint model also report zero-audio, zero-video and mismatched-video
inference, without retraining. The mismatched-video permutation preserves the
exact video-feature marginal and sends every window to a different recording;
it does not silently substitute a same-recording window. If a single recording
contains more than half a split's windows, such a bijection is impossible and
the control fails explicitly. No labels are used in this mapping.

Report each event's Accuracy, balanced Accuracy, NLL, two-option Brier, ECE15,
positive/negative counts, and sensitivity/specificity. A missing class side
means balanced Accuracy is null, not zero or one. Bell remains in all reports
despite no positive training annotations. Always show the prior and class
prevalence; pooled Accuracy alone is misleading. Return probability vectors
in `[no, yes]` order, choosing no at a0.5 tie, and test reversed options after
restoring their original order.

Use10,000 paired bootstrap replicates by room, seed17, percentile95% intervals.
Explicitly label the two-room validation intervals severely low-power; many
windows do not supply independent environments. Retain all individual window
predictions for paired comparisons. No product promotion or sealed evaluation
is authorized by this development result.

`scripts/starss23_readout_common.py` implements the metric conventions and
label-independent cross-recording permutation. CPU tests cover marginal
preservation, deterministic mapping, impossible controls, absent-class
handling, Brier normalization and invalid probabilities. These helpers passing
tests is not evidence that the planned training/evaluation has run.
