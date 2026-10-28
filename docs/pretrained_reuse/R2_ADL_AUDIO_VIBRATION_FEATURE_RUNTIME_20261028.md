# ADL paired audio/vibration frozen-feature runtime smoke

Date: 2026-10-28. Status: **runtime smoke passed; no Decision quality gate
measured**.

## Scope and integrity

This smoke executes the already pinned audio and sensor encoders on the same
single development-training row. It does not train a readout, inspect labels,
score validation, or open a sealed audit. The complete ADL download was already
present and authenticated; no repeat download was made. The existing worktree
with uncommitted user changes was left untouched. A clean feature branch was
created from roadmap commit 81d57da616886bc03d1690e2a86d124581af55e1.

The fixed row is row 0, participant 4, trial 101, assigned to train by seed-17
participant hashing. Only
23_take-from-stove/23_take-from-stove_multisensor-1.hdf5 was read from the
archive. Its SHA-256 is
73aae1893b07c23c0814362f906e9795dea9a03c7a6c13ab03e2f07e5af5f645; the
parent archive SHA-256 is
04bf32f55c93f54c49525d2b0581cb51f14c1ec302afb56b70a995fa4aa8f706. The
split-protocol SHA-256 is
9c301d4a3acc37711d25943e7165533410bb3297630eef19c7f6431d2daaf5f8.
No validation waveform or sealed-audit data was loaded.

## Pinned encoders and transformation

- Audio: EfficientAT DyMN-0.4, source commit
  a425fdce92572e602a1d5634799bd9f1f2efa806, 1,961,239 parameters,
  checkpoint SHA-256
  49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e.
- Vibration: MOMENT-1-small embedding mode, source commit
  38f7310ad594100747ca2a8357e9c7ca7d323e0e, 35,337,408 loaded parameters,
  checkpoint SHA-256
  785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877,
  config SHA-256
  6456b8e6f017323e8ef372c338c993380e246cb6291ffb8ca21fe7dfc53d0a6f.

The audio input is the first two seconds of signed PCM16, normalized by 32768
and resampled from its declared 16 kHz rate to 32 kHz. The three vibration
axes come from the same event row, with recorded sample rate 1080.0356 Hz; the
first two seconds are linearly resampled to 512 samples per axis for MOMENT.
This linear resampling is only a runtime-smoke choice, not validated signal
preprocessing. Matching record identity and duration do not prove physical
clock synchronization.

## Measured result

The smoke ran on CPU only with CUDA hidden and uninitialized. EfficientAT
returned shape [1,384]; MOMENT returned per-channel patch embeddings
[1,3,64,512] and a mean-reduced [1,512] representation. All outputs were
finite and nonzero; each model's parameter-state hash was identical before and
after its forward pass. The audio and sensor feature files had identical
SHA-256 hashes across all three smoke attempts.

| Measurement | Result |
| --- | ---: |
| Audio extraction path, including local model setup | 1.75 s |
| Sensor extraction path, including local model setup | 3.53 s |
| Whole-process wall time (attempt 03; three-run mean) | 7.75 s; 8.58 s |
| Whole-process peak working set (maximum measured) | 950,910,976 bytes (0.89 GiB) |
| GPU / CUDA allocation | None / not initialized |
| Combined encoder parameters | 37,298,647 |
| Training examples / validation examples scored | 0 / 0 |
| Accuracy, NLL, Brier, ECE | Not measured |

The encoder parameter sum is not a claim about total model parameters,
simultaneous residency, package size, latency under fusion, or decision
quality. The execution was sequential; the recorded peak working set therefore
does not establish peak memory for a fused runtime.

## Reproduction artifacts and verification

Exact command, run from branch codex/r2-adl-feature-smoke-20261028 with
CUDA_VISIBLE_DEVICES empty:

    C:\Users\junny\AppData\Local\Programs\Python\Python311\python.exe scripts/smoke_adl_audio_vibration_features.py --output C:\CodexArtifacts\pretrained-reuse-r2\adl-audio-vibration-feature-smoke-20261028-attempt-03

The script pins protocol, source, checkpoint, config, archive, and selected
HDF5 member hashes; it refuses to overwrite an output directory. The final run
receipt is
C:\CodexArtifacts\pretrained-reuse-r2\adl-audio-vibration-feature-smoke-20261028-attempt-03\run.json
with SHA-256
fb1cac6e26b03caa3b06ee721eaeb92e9169c4eb39fddea19d512945965a9628.
The executed script SHA-256 is
89a63ac4c7fb500cc3dcdcedeeba4a687c518da255a0bf38085b1494cccdc16b.
The audio feature SHA-256 is
1bfabdbe9f247ff41aab993df795b1a569a9f89b7d6cbeb918fe344fc20ac495;
the sensor mean-feature SHA-256 is
883e6c55ad903d74f25609ffad2ddb4df1563ff03e2f60051a55f4e07994f5e5; and
the patch-feature SHA-256 is
6436f33f29508235fb1556afd3df52ced393cb7b511c4e34fb18a864a0b0766b.

The focused CPU tests pass (7 passed); the full CPU suite passes (2,576 passed,
11 skipped); Ruff and git diff --check pass. Environment: Windows 10 build
26300, Python 3.11.9, PyTorch/torchaudio 2.6.0+cu124, NumPy 2.4.6, h5py 3.14.0,
Transformers 5.6.2, huggingface-hub 1.29.0, psutil 7.2.2. No dependency was
installed or upgraded.

## Interpretation and next gate

This proves that the pinned local audio and time-series encoders can both load
and produce frozen features from one same-row ADL input. It does not prove that
audio and vibration are useful for a decision, that they are physically
synchronized, or that their fusion improves a task. The source's fixed split
has only 12 training participants and 2 validation participants, one
environment, one activity lacking split coverage, and no video. It remains
below the roadmap's independent-group quality screen and is not admitted for a
quality fit. Do not train on the exposed source to make a product-quality
claim. The next R2 action remains finding a rights-compatible, adequately
grouped task whose target demonstrably requires both proposed input streams,
with frozen option/metadata, unimodal, and perturbation controls.
