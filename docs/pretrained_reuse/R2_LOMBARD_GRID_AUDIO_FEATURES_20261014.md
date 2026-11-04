# Full native speech-token extraction completed

Uses [media-gated fixed layout](R2_LOMBARD_GRID_MEDIA_20261014.md), retaining
4,080train/398calibration/889development-validation recordings. Four corrupt
training videos and19 unresolved labels remain protected separately. No model
quality, calibration, checkpoint selection or learning was performed here.

Actual full extraction passed5,367 unique source/decoded-wave/hash checks,
full audio coverage, finite outputs, exact same-GPU repeated inference for every
record, and unchanged frozen parameter versions. All674,009physical384D tokens
are retained with source offsets/sample counts; mean pooling is also saved.
First`ceil(samples/320)`steps follow the native30s padded Whisper frontend,
without source transcript or gold word alignment. No oversized clip was truncated.

Independent cache reload verified contiguous offsets, physical lengths, all split
counts and bitwise means for every record. The four CUDA full-release smokes
match their full-cache token slices bitwise. CPU/CUDA tolerance remains1e-4;
bitwise parity is claimed on the same backend only.

- Frozen8,208,384-parameter Whisper Encoder,FP32,seed17,TF32 off.
- RTX3080Laptop16GB,160.33s processing excluding load/initial source hashing.
- Allocator peak100,234,752bytes; reserved113,246,208 (excludes driver/context).
- Token cache1,035,277,952bytes,SHA
  `c8bec22068bf089b71c4e4ec31174e1a0a68d53f453bdf81228d3d27488a2763`.
- Mean cache8,243,840bytes,SHA
  `e09aab88f31957c53c2e48f38a7e79971739efc916547cd5fd3dd6a1bca0a509`.
- Source commit`2c44c09339712b055b9ac520dd36a7a67a5a63cb`.
- Exact runnerSHA
  `ded8dadf78bf9f832b45f72ecb6b496e2b99bcd5e1f08a3201cb7a56b7dd21ef`.

Local`C:/CodexArtifacts/pretrained-reuse-r2/lombard-grid-whisper-features-01/`
holds caches, identities, frozen hashes, progress, final report and reload audit.
Runner:`.../lombard_grid_whisper_features.py`. These are research caches, not a
<=200MB deployment package; they are not committed. No environment upgrade,
cloud compute, original artifact modification or sealed evaluation.

Next measure the visual extraction cost on real training clips, then fix the
native audio/video/joint readout comparison protocol before looking at quality.
The audited VideoPrism runtime uses JAX/Flax CPU here; official PyTorch support
remains a TODO in the [official repository](https://github.com/google-deepmind/videoprism).
Do not assume CUDA support or a new port's parity. Ordered features permit a
controlled readout comparison without changing source media or re-extracting audio.
Native command accuracy and the complete95–98% Omni target remain unmeasured/unmet.
