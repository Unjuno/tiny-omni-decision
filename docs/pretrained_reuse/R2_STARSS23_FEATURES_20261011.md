# Frozen features for synchronized STARSS23 windows

## Result and scope

Four previously fixed five-second dev-train windows now have real frozen audio
and video features. All four pairs match window IDs, order and input SHA-256.
Each feature is finite and passes exact save/reload; a second independent
forward of the same input produced maximum absolute difference zero.
No readout was trained and no Decision quality metric was computed.

| Path | Loaded parameters | Feature | First extraction | Later extractions | Total process work | Peak working set |
|---|---:|---|---:|---|---:|---:|
| EfficientAT DyMN-0.4 | 1,961,239 | 384 float32 | 0.081 s | 0.041 s each | 3.711 s | 554,610,688 bytes |
| VideoPrism-B | 114,365,184 | 768 float32 | 6.438 s | 1.221–1.267 s | 18.362 s | 4,133,212,160 bytes |

Both ran on CPU; GPU was idle before execution. Total work includes model
loading, window archive reading, initial extraction and repeated forwards.
First video timing includes JAX compilation. These four observations are not a
latency benchmark or a simultaneous-residency measurement. Model tensors were
not updated. The combined 116,326,423 parameters exclude any text encoder and
future Decision head and do not establish the full product budget.

## Fixed preprocessing and limitations

- Audio: use original channel index 0 only, torchaudio sinc resample 24 to
  32 kHz, upstream `AugmentMelSTFT` in eval mode, model float32 in eval mode,
  DyConv temperature 1.0 matching pretrained inference. Strictly load local
  checkpoint; no network download. No spatial audio claim is made.
- Video: use the eight timestamp-selected frames of that same five-second
  window, resize the entire equirectangular frame to 288x288 with TensorFlow
  bilinear interpolation, divide by 255, run the pinned model and mean-pool
  its tokens. This preserves full image coverage but distorts aspect ratio;
  it is explicitly a plumbing choice, not validated visual preprocessing.
- Store each observation feature once for reuse across its twelve questions.
  The semantic usefulness of this temporal pooling remains unmeasured.

## Reproduction and provenance

Source base: `f07b71e` with new runner `scripts/extract_starss23_features.py`.
Exact executed runner SHA-256:
`6eb79dd70c53732bc09e8b515af15ff91ca8080b8910307f6eb6eb7196d0787b`.
Model source commits and checkpoint hashes are verified before loading and
stored in the result JSONs. Existing isolated environments were reused without
installation or upgrades.

```powershell
$env:PYTHONPATH='C:\CodexArtifacts\pretrained-reuse-r1\python-deps'
python -m scripts.extract_starss23_features --modality audio --windows C:\CodexArtifacts\pretrained-reuse-r2\starss23-window-smoke-01\result.json --source-root C:\CodexArtifacts\pretrained-reuse-r1\efficientat-audit-v1\source --checkpoint C:\CodexArtifacts\pretrained-reuse-r1\efficientat-audit-v1\dymn04_as.pt --output-dir C:\CodexArtifacts\pretrained-reuse-r2\starss23-audio-features-02
```

For video, use the separate VideoPrism venv (without the audio PYTHONPATH):

```powershell
& C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe -m scripts.extract_starss23_features --modality video --windows C:\CodexArtifacts\pretrained-reuse-r2\starss23-window-smoke-01\result.json --source-root C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source --checkpoint C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz --output-dir C:\CodexArtifacts\pretrained-reuse-r2\starss23-video-features-01
```

Outputs refuse overwrite. The initial audio attempt `starss23-audio-features-01`
failed at import because torchaudio was absent from the default Python path.
Its FAIL result is preserved. Attempt 02 uses the existing audit dependency
directory; there was no model or data change. Audio environment: Python3.11.9,
PyTorch/torchaudio2.6.0+cu124; video: Python3.11.16, JAX0.6.2, TensorFlow2.18.1.

Small Git records:

- `R2_STARSS23_AUDIO_FEATURES_20261011.json`, SHA-256
  `5df0a8d18b755fd4e0ad7bac5ead54510c2eb8583488abe8b083d358a0f59ae6`.
- `R2_STARSS23_VIDEO_FEATURES_20261011.json`, SHA-256
  `57ba216420624cc617942bf7b31d1b4820d4c55f8ff331354673e050845b469d`.

Actual feature arrays remain in their named external output directories. Legacy
Teacher artifacts and sealed data were not accessed or modified. Next work is
adequate room-grouped task coverage and calibrated matched audio-only,
video-only and joint readouts; these four windows cannot support that claim.
