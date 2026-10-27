# ADL: frozen audio input smoke

The preregistered first training row of the fully publisher-MD5-verified
`23_take-from-stove` archive was read on CPU. The first two seconds were
resampled from the [publisher-declared 16 kHz](https://www.mdpi.com/2306-5729/9/12/144)
to the pinned EfficientAT input rate of 32 kHz. This rate was not independently
measured from the headerless HDF5 array.

The frozen 1,961,239-parameter DyMN-0.4 produced a finite nonzero `[1, 384]`
feature. Model-state hashes before and after match; CUDA was not initialized.
The complete audio array had 42,605 samples (2.6628125 s at the declared rate).
The vibration axes each had 2,876 samples at the recorded 1080.0356 Hz
(approximately 2.6628751 s). Similar durations do not prove the sensors'
physical clock alignment or correct labels. This was one training-side input
test, with no learning, validation waveform, held-out audit or decision score.

Runtime receipt SHA-256:
`5dc348eebc17709f303bfa21dd3c3a948b6df8485ac3c0b705a89273659b5b0e`.
The exact source archive and model hashes, input selection, timings, feature
hash, source revision and model-state hashes are in the receipt at
`C:/CodexArtifacts/pretrained-reuse-r2/adl-frozen-audio-smoke-20261027-attempt-01/run.json`.
The runtime took 3.81 s on CPU. No raw audio was saved. This does not admit the dataset for training
or resolve the natural audio/video and four-modality quality gates.

The script now requires a fresh `--output` directory and checks the fixed
protocol hash before reading the archive. A separate CPU replay in
`C:/CodexArtifacts/pretrained-reuse-r2/adl-frozen-audio-smoke-20261027-attempt-02/`
produced the identical feature SHA-256 `1bfabdbe9f247ff41aab993df795b1a569a9f89b7d6cbeb918fe344fc20ac495`.
Its receipt SHA-256 is `683ef6b5c19767623ce9a1c406d7fdb3cf5a41343db709e5d592835a9e96a462`.
It used the same single training row, no fitting, and no validation waveform.
An existing output directory is rejected before any file is written.

```powershell
python -m scripts.smoke_adl_frozen_audio --output C:/CodexArtifacts/pretrained-reuse-r2/adl-frozen-audio-smoke-NEW
```
