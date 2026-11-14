# Path A Audio CPU numerical parity diagnosis — 2026-11-14

## Result

The Audio CPU parity failure is strongly localized to frozen Whisper feature
generation, before the candidate readout and JSON serialization. The exact
Speech Commands cache was generated with CUDA BF16 autocast; the canonical
runtime uses CUDA BF16 autocast only on CUDA and FP32 on CPU. Re-encoding the
same first two already-observed validation clips on CPU produced features that
differ from the cached CUDA features. The resulting decision argmaxes remained
unchanged in the preceding CPU dispatcher run, but the probability vectors
exceeded its frozen `1e-5` parity bound.

This directly confirms a device/precision feature-path difference and is
consistent with the observed probability mismatch. It does not establish a
task-quality change, nor does it establish that CPU FP32 is unsuitable for a
separately trained or calibrated CPU candidate.

## Pinned evidence

- Runtime parity attempt: `D:\CodexArtifacts\pretrained-reuse-r2\path-a-canonical-cpu-runtime-v0-attempt-01\latency-distribution.json`, SHA-256
  `a6c84fe72c11f411fafebf31b067541fc944786ac0c788dcce7e8b8aabde99b2`.
- Speech Commands feature cache:
  `C:\CodexArtifacts\pretrained-reuse-r2\audio-probe-v0-retry3\audio-features.pt`, SHA-256
  `ace323458e7f443daf63382d2d41d99151d2a5801c671d30b0df68d52338ae64`.
  The first two validation IDs are `speechcmd-validation-dc3aa2cf1963db4c3905`
  and `speechcmd-validation-18a4e01cb64c0771518e`.
- Whisper Tiny revision: `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`; weights SHA-256
  `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`.
  Validation manifest SHA-256: `18520e5d422fa4a129aa1c9ef3295acd4f00c97c0ca3493cb972c28aafa7d0bb`.
- Cache extraction code: `scripts/train_frozen_audio_probe.py`, SHA-256
  `79b6447ebeb205e931a997d50b04035c0f46ca0dc37e64e26900576c6072ccea`.
  Its encoder extraction wraps the forward in
  `torch.autocast("cuda", dtype=torch.bfloat16)`.
- Canonical inference code: `src/tiny_omni_decision/audio_runtime.py`, SHA-256
  `98d9df174be586f454881adbc1e18e7d20fcd342e105c64b5424597b35dcc6ac`.
  `_encode_audio_batch` uses CUDA BF16 autocast for CUDA and `nullcontext()` on
  CPU, so CPU inference runs FP32.
- Prior CUDA cache check:
  `C:\CodexArtifacts\pretrained-reuse-r2\audio-cache-parity-v0\report.json`,
  SHA-256 `6d02a98c91dfd0a393b27d38dbca9afaf2b20ca750218c01d21c94392e1accdf`.
  On the first validation clip, one-event CUDA re-extraction matched the frozen
  feature and fixed-readout logits exactly (`max_abs_difference: 0.0`). This
  ties that cache row to the CUDA BF16 path.

## Same-input CPU feature comparison

The local fixed Whisper Tiny revision and feature extractor were loaded from
the existing assembly. Each listed waveform was read from the pinned
validation manifest, transformed with the same WhisperFeatureExtractor at
16 kHz, padded/truncated to `n_samples`, passed through the Whisper encoder,
and pooled over `ceil(waveform_samples / 320)` valid tokens exactly as in the
runtime. The measured Python process used Python 3.11.9 / PyTorch 2.6.0+cu124 / Transformers 5.6.2 and four Torch CPU threads. Cached rows and the two WAVs were read only; labels were not used, and no values were written back.

| Validation ID | CPU mode | Max abs feature delta vs cache | Mean abs feature delta | Cosine vs cache | Encoder forward |
|---|---|---:|---:|---:|---:|
| `speechcmd-validation-dc3aa2cf1963db4c3905` | FP32 | 0.08695 | 0.006316 | 0.9999619 | 0.220 s |
| same | CPU BF16 autocast diagnostic | 0.11867 | 0.007081 | 0.9999611 | 1.937 s |
| `speechcmd-validation-18a4e01cb64c0771518e` | FP32 | 0.02074 | 0.004386 | 0.9999832 | 0.196 s |
| same | CPU BF16 autocast diagnostic | 0.01453 | 0.002576 | 0.9999943 | 1.998 s |

CPU BF16 moved closer for one sample and farther for the other, and was about
9–10 times slower in this two-item diagnostic. It is not a consistent parity
fix, so no autocast policy or tolerance was changed.

## Disposition

Keep the current CPU results as measured: Text/Image/Video passed on the two
sampled assets each; Audio typed JSON remained internally consistent and its
argmaxes did not change, but saved-probability parity failed on both Audio
assets. Do not change the Audio checkpoint, feature cache, dataset, training
run, or parity threshold to make this result pass.

If CPU deployment becomes a release requirement, first decide whether matching
the CUDA reference probabilities is required. A legitimate CPU-specific model
would need an explicitly separate frozen-feature training/validation track and
its own task-quality evaluation; this diagnostic does not authorize that work
or justify changing the current product candidate. No further repeated
validation scoring is needed to reach this diagnosis.


## Fixed-readout isolation follow-up

To isolate the encoder from CPU text embedding, readout math, and JSON, the
fixed CPU `SpeechCommandsMultiQueryRuntime` loaded the same readout and CPU
MiniLM once, then scored all four saved question templates twice per clip: once
with the immutable cached Whisper vector and once with a fresh CPU FP32 Whisper
vector. The same saved probability rows were the numerical reference; labels
were not used.

The cache-feature arm reproduced the saved probabilities to at most
`5.50e-8` across all eight queries. The CPU-reencoded-feature arm reproduced
the prior CPU-runtime deltas exactly: up to `6.1774e-4` and `7.4472e-5` on the
two keyword queries. All eight argmaxes were unchanged in both arms. Therefore,
for these two clips, CPU text embeddings/readout and typed JSON are not the
material parity source; the probability drift enters with the audio feature.
This two-clip localization does not establish behavior for all Speech
Commands inputs or other audio tasks.

Machine-readable result: `D:\CodexArtifacts\pretrained-reuse-r2\path-a-audio-cpu-feature-isolation-v0-attempt-01\feature-readout-isolation.json`, SHA-256
`80698559a4ae1f32c920fc0ed4c601c31d4842a6707d33dd828212ed933ba9ea`. The
process ran CPU-only in Python 3.11.9 / PyTorch 2.6.0+cu124 with four Torch
threads, used pinned cache, checkpoint, manifest and prediction hashes recorded
in the JSON, performed no training, and loaded no sealed audit. No inference
code or tolerance was changed.
