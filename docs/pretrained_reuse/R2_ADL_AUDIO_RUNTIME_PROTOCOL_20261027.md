# ADL frozen audio runtime: fixed training-row smoke

Select the first row assigned to train by the existing participant hash in the
first sensor of the fully MD5- and SHA-verified `23_take-from-stove` archive.
Do not inspect validation waveforms. Require at least two seconds of audio;
take its first two seconds without padding, normalize signed PCM16 by 32768,
and resample the publisher-declared 16 kHz stream to 32 kHz. Use the existing
frozen EfficientAT DyMN-0.4 CPU runtime, exact pinned source and checkpoint.
Save only feature and an execution receipt in a fresh external directory.
Inspect the three vibration axes only for array lengths and the recorded rate;
duration agreement does not establish synchronization. Fail on an invalid input,
different checkpoint, changed model state, or unexpected feature shape.

This is one input-path check, not model fitting, a label prediction, a split
integrity audit, or evidence that the audio is useful. The publisher's audio
rate is a source declaration, not a value measured from this HDF5 array.
Do not change the selection or use this as a quality benchmark.
