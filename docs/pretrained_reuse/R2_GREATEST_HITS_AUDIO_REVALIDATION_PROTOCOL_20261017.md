# Revalidate the saved complete audio cache after a count assertion error

The producer61530ad exited1 after all11,151 events/409 parents/32,582 questions.
Its result remains INCOMPLETE, SHA
`74a12e2d2779751eed6867b3fd0186e6ad14d9ea44a527a2d9b0ceee41a1c10d`.
Frozen before/after states agree. The failed compound condition has one false
term: matched old probe events2 versus a hardcoded expected3.

The missing reference is parent2015-03-30-00-26-27/source line1, timestamp1.778125,
with material/action/reaction allNone. It has no frozen task rows. A later event
of the same parent is in the21-review cohort. Therefore neither the full corpus
nor the smoke contains that original event. Preserve the corpus and labels.
Replace the future producer's hardcoded count with the selected-event/reference
identity intersection; test absent, included and duplicate selections first.

Do not rewrite the producer receipt or infer success from coverage alone. A new
CPU-only independent verification must hash all823 produced files and all original
source/model/input pins; compare the complete frozen plan and unique ordered
event binding; validate all FP32 bank shapes and finite values; match22,302 feature
digests to captured encoder outputs; freshly decode all409 native media and replay
96kHz crops/direct16/32kHz resampling for all11,151 recorded input digests; compare
all clocks; match42 known smoke feature slices and shared queries/positions; and
verify frozen encoder states against the successful smoke. Record the omitted
reference and its pinned original annotation explicitly.

The revalidation accepts only the exact pinned producer/error above. It is not a
general partial-run bypass. Output a new PASS_FULL_AUDIO_CACHE_REVALIDATION receipt
only if every check passes. Keep original artifacts/status/log unchanged. This
verifies stored cache integrity and all input replay, not fresh full encoder
recomputation, training resume or quality. Future consumers must explicitly pin
and validate the independent receipt; a raw INCOMPLETE producer remains rejected.

Run from the isolated transport checkout with CUDA hidden and existing CPU/PyAV
environment, one decoder thread, max900s/4GiB host/16GiB disk reserve. No GPU
learning, encoder recomputation, calibration or sealed audit. Original video PID
17848 remains untouched. Audio PID12824 has terminated and was reused by an
unrelated process; never intervene based on its number alone.
