# FAIR-Play train-only audio pair waveform check — 2026-10-22

## Scope and inputs

This CPU-only check compared the embedded AAC audio in eight already acquired
FAIR-Play / PseudoBinaural split 1 **train** MP4 files with their corresponding
PCM WAV files. It used the pinned FAIR-Play revision
`2666d6caf2d126ae43a84841e94e0c9ae24b3e2d` and the existing
`decode-grouping-receipt-v2.json`. Before decoding, the script verifies each
asset's exact byte size and SHA-256 against that receipt. No labels,
validation/test IDs or media, model, features, fitting, or GPU were used.

Both audio streams were decoded with PyAV and resampled to float mono at
48 kHz. The audit computes zero-lag Pearson correlation and the largest
absolute overlap Pearson correlation within +/-500 ms using FFT correlation.
Positive reported lag means the paired WAV waveform is delayed relative to the
MP4's embedded audio. Raw waveforms are not written to disk.

Reproduce in PowerShell with a **new, empty output location**:

```powershell
py -3.11 scripts/audit_fairplay_audio_pair_alignment.py `
  --receipt C:/CodexArtifacts/pretrained-reuse-r2/fair-play-train-group-preflight-attempt-01/decode-grouping-receipt-v2.json `
  --media-root C:/CodexArtifacts/pretrained-reuse-r2/fair-play-train-group-preflight-attempt-01 `
  --media-root C:/CodexArtifacts/pretrained-reuse-r2/fair-play-train-media-preflight-attempt-01 `
  --output NEW_ATTEMPT_DIR/report.json
```

## Results

| Train asset | Zero-lag correlation | Peak correlation | Peak lag (ms) |
|---|---:|---:|---:|
| 000379 | 0.0697 | -0.7773 | -18.33 |
| 000380 | -0.1090 | -0.3262 | +33.75 |
| 000381 | 0.0328 | 0.6878 | +125.87 |
| 000383 | 0.0919 | 0.7621 | +245.25 |
| 000384 | 0.1879 | -0.7057 | -27.85 |
| 000385 | -0.0240 | 0.8775 | +190.12 |
| 000386 | 0.2828 | -0.6666 | +139.58 |
| 000387 | 0.1157 | 0.9480 | +238.90 |

The median absolute peak correlation is 0.7339. Peak lag estimates vary from
-27.85 ms to +245.25 ms (negative to positive); all decoded container start
timestamps are zero. WAV and AAC sample counts differ by at most 1,024 samples
(21.33 ms) after resampling. AAC lossy coding, channel downmixing, and clip
trimming can affect correlation and its peak. Therefore these values only
support that each WAV/MP4 pair contains appreciably related audio in this
small sample; they are **not** a calibrated physical synchronization
measurement.

## Integrity and limits

- Artifact: `C:/CodexArtifacts/pretrained-reuse-r2/fair-play-audio-alignment-attempt-01/report.json`
- Report SHA-256: `0c85a8f9914469c6adaf4c25fc20c58a23ad775ba498cf063703b383c1159418`
- Input receipt SHA-256: `8b77a14b0fedc406106fabaeb0e57151c6a847d9d6d86aeb9a23c1fe6223aa1b`
- The report includes each media SHA-256, decoder metadata, and the canonical
  receipt hash. The code refuses changed/missing media or output overwrite.
- Tests: `tests/test_fairplay_audio_pair_alignment.py` covers positive/negative
  lag convention, non-finite input rejection, and rejection of a non-train
  receipt before media access.

This does not compare audio against video frame timestamps, establish whether
the paired WAV and video came from the same original scene, certify split
independence, prove task quality, or show that both modalities are needed.
FAIR-Play remains a research candidate and is not admitted for quality-selected
training based on this check.
