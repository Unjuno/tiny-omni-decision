# ADL two-activity, one-sensor split content identity check

Read the first sensor HDF5 in each of the two fully publisher-MD5-verified
activities (light-off and take-from-stove); no other activity is selected.
Check the pinned archive hashes and fixed participant split protocol. For each
train or development-validation row, hash the length and little-endian int16
bytes separately for audio and each of the three vibration axes; retain only
aggregate counts. Reject invalid data and unassigned rows rather than assigning
them to either split. Compare every train row in either activity against every
validation row in either activity by each channel hash, including constant
channels. Report matching row counts, unique hashes, and constant/empty counts.
Do not write raw waveforms or per-row identifiers. No model, model selection,
quality score, training, or sealed audit is involved.

Exact byte identity can reveal repeated channel content but cannot establish
that nonidentical clips do not overlap in time, share a session, derive from the
same physical event, or contain similar signals. No activity beyond the two
checked archives is covered, and only sensor 1 is checked. No training
admission or product-quality claim follows.
