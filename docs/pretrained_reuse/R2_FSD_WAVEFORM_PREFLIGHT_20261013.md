# Waveform-overlap preflight and fixed comparison conditions

Implemented FFT-based sliding normalized waveform correlation, invariant to
constant offset, gain and polarity. Four CPU tests verify a planted shifted
match, parity against direct Pearson correlations, undefined silent signals,
and nonfinite/shape rejection. This is not a robust semantic fingerprint.

Executed a limited real-data smoke: eight fixed FSD training clips versus each
clip's embedding-nearest STARSS full recording, channel0, common2kHz torchaudio
resampling. Verified STARSS file hashes before decoding. Maximum absolute
correlations range .0144–.3033, below the diagnostic review threshold .99.
No cases were removed and no model/threshold was fitted. Edited, time-stretched,
mixed or shorter reused fragments may escape this comparison. Other recordings
were not searched. Therefore the all-source duplicate gate remains open.

Result: `C:/CodexArtifacts/pretrained-reuse-r2/fsd-waveform-nearest-smoke-01/`;
companion JSON contains exact matched recording IDs, hashes, offsets and scores.
Additional128-clip acquisition remains a separate live operation; do not restart
it because this smoke has finished.

## Comparison conditions to hold before any additional learning

- Preserve all original STARSS train/calibration/validation rows and options.
- Reuse the pinned frozen EfficientAT pooled384 features and original training
  standardization; no encoder updates or fitted FSD normalization.
- One explicit whole-clip positive query for each of128 FSD clips. Unknown
  events are omitted, never negative. No crop inherits an unlocalized label.
- Fresh seed17, same74,753-parameter scorer, optimizer,24epoch budget, earliest
  minimum validation NLL selector and independent calibration grid as baseline.
- Record actual extra examples and optimizer updates: adding data at fixed
  epochs also increases compute, so do not describe it as a compute-matched test.
- Preserve every per-event metric and the negative-prevalence prior. Do not
  select/reweight events based on the already-observed validation results.
- Acquisition, complete finite-feature validation and cross-source overlap
  review must finish first. No training is launched by this document.

The intended question is whether bounded additional independent positive
coverage helps rare events without degrading other events/calibration.128 clips
cannot establish general95–98% quality. This remains an observed-development
comparison, not a product promotion or blind audit.
