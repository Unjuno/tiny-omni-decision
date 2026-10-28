# ADL audio/vibration source: preregistered split feasibility protocol

This protocol was written before opening or inspecting the second activity
archive. Its purpose is to check whether the existing data can support a
participant-disjoint **development** comparison. It is not a quality
benchmark, selection rule for a product, or a replacement for a sealed audit.

For every non-negative, real participant ID other than the producer's special
`999` no-participant ID, form UTF-8 bytes for `adl-participant-v1:17:<ID>`,
where `<ID>` is its base-10 integer representation. Compute SHA-256, interpret
the first eight digest bytes as an unsigned big-endian integer, and take modulo
100. Values 0–69 are assigned to development train, and 70–99 to development
validation. The same ID has the same split across **all** activities, trials,
and devices; adding new IDs cannot reassign existing ones. Do not search seeds
or move individual IDs after seeing data.

The same participant/trial numbers may recur across activities; their complete
observation identity must therefore include the activity and sensor. All five
devices for one participant, activity and trial share the participant split.
If an observed participant/trial repeats within one activity and sensor, fail
closed rather than silently merging. Shared environment, recording sessions,
content, capture dates, clocks and media hashes still require separate checks.

ID `999` denotes no participant in the source paper. Keep any such rows
**unassigned** until a defensible recording/session grouping is identified;
do not drop them or allocate identical sessions to both splits. Treat missing,
negative or invalid participant IDs as unassigned/errors, not new people.

Report the per-activity number of participant groups and unique trials in both
development splits and the number of unassigned rows. An activity absent in
either split **fails the coverage check**. The first two archives are only a
bounded source feasibility sample; both passing would not establish coverage
for the other 23 activities. Do not fit or score a model, choose thresholds, or
inspect any sealed audit on the basis of this incomplete source screen.

The entire source has only 14 people and one environment. Even perfect group
separation here cannot meet the roadmap's provisional 200-independent-group
quality screen or establish broad audio/video capability. Further source data,
physical timing checks, label checks, and frozen matched unimodal/joint controls
are necessary before a scoped audio/sensor fit.
