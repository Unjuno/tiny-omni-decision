# ADL cross-activity timestamp feasibility check

Compare the first sensor's metadata from the two existing fully
publisher-MD5-verified activity archives, light-off and take-from-stove.
Check both cross-activity directions under the already fixed participant split:
training timestamps from one archive against validation timestamps from the
other. Report exact timestamp coincidences and nearest absolute-gap quantiles
in stored native units. No waveforms, model, selection, or audit set are read.

Verify both complete archive hashes, the existing split-protocol hash, unique
participant/trial pairs within each activity and finite timestamps. Fail on an
invalid input or an existing output. Preserve unassigned participant 999 as a
count, not as train or validation. This is a necessary metadata diagnostic,
not proof of disjoint recording sessions, physical synchronization, or absent
media overlap: timestamp semantics, units and event duration remain unresolved.
