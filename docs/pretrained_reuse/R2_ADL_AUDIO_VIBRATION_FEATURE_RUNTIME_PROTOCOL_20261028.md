# ADL paired audio/vibration frozen-feature runtime smoke

This is a one-row training-split runtime smoke, not a quality benchmark or R2
task admission. Use the already verified 23_take-from-stove.zip archive, its
first lexicographically sorted sensor-1 HDF5 member, and the first row assigned
to training by R2_ADL_PARTICIPANT_SPLIT_PROTOCOL_20261027.md (seed 17). Read
audio and all three vibration axes from that same row only. Do not read
validation waveforms, labels, sealed audit data, or other archive payloads.

Use the first two seconds of signed PCM16 audio at its declared 16 kHz rate,
normalized by 32768 and resampled to 32 kHz for the pinned frozen
EfficientAT-DyMN-0.4 CPU encoder. Use the first two seconds of each vibration
axis at the row's recorded sample rate and linearly resample the three axes to
512 samples for the pinned frozen MOMENT-1-small embedding pipeline. MOMENT
receives [batch=1, channels=3, samples=512]; use embedding mode and mean
reduction over channels/patches. Record that the linear resampling is only an
input-path smoke choice, not validated signal preprocessing.

Require exact source/checkpoint hashes, finite nonzero feature tensors, the
expected output shapes, unchanged frozen model state, CPU-only execution, and
no CUDA initialization. Save features and a detailed receipt to a fresh
external output directory. Do not train a classifier, emit predictions, or
compute quality metrics. Matching row identity and durations do not prove
physical clock synchronization or that either representation supports a
useful decision.
