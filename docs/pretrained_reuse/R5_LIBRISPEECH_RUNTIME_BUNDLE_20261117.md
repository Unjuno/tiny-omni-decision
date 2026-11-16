# LibriSpeech Path A runtime bundle smoke — 2026-11-17

## Purpose and scope

This checks whether the selected 100-epoch LibriSpeech frozen-feature readout runs end-to-end from an immutable local bundle and produces the repository's typed probability JSON after save/reload. It is a runtime integration check on one already-observed development-validation item, not a new quality evaluation, benchmark, blind estimate, or product promotion. No training, calibration, quantization, sealed-audit access, official-test access, GPU work, or cloud work occurred.

## Implementation

- `SpeechCommandsMultiQueryRuntime` now accepts an expected readout SHA-256. Its default remains the previous Speech Commands readout hash, preserving that runtime. It validates the checkpoint hash before loading tensors.
- `scripts/build_librispeech_runtime_bundle.py` builds into a new directory and refuses overwrite. It validates the immutable source bundle and the selected training report/checkpoint, checks split overlap counters, copies the pinned model snapshots and selected readout, writes a file manifest, and verifies the resulting bundle.
- Added a readout-hash validation test. The previous Speech Commands bundle and all existing data/model artifacts remain unchanged.

The final bundle is:

`C:\CodexArtifacts\pretrained-reuse-r2-cdrive-20261117\librispeech-transcript-choice-runtime-v1-attempt-03`

It was built from source commit `2e138969aed0aec693ea60e1169839efc1bad09c`, contains 27 manifested files, and verifies successfully. Manifest SHA-256: `1499b5ce24a566545b07e364acdee9079e0fe5c36cfdf14dd603cce17b9bead4`. The pinned encoder snapshots were copied from the immutable `runtime-bundle-v2`; its manifest SHA-256 is `574640d88d6ddbbdb2d12f2466a248ce13c8d9eb3ac542d099b2f08981b690f4`. The selected epoch-40 readout SHA-256 is `f7ce2d1b04f6dde98e44cc1343916d594e2c095ad1d70f335323d0dbb5bc1f3a`.

The bundle embeds the exact runtime source. `audio_runtime.py` SHA-256 is `9ebf5a76c0c4bf8dc50692c7a80cb13ca8499146c1f6c802e29b23339450d38c`; `decision_json.py` SHA-256 is `967a035c22f9427fd06ccd669c0c4ae92db709b795312e2453a6d0c5650f3eb0`. Package code is not inferred from a Git commit alone: these hashes are in the bundle's own file list.

## End-to-end bundled-runtime smoke

The input was a held-speaker validation FLAC (`103-1240-0004`, 12.515 s), converted to mono 16-bit 16 kHz WAV. The runtime received the fixed question and four hard-negative transcript options. This sample was already used in development validation, so this is only a save/reload and interface parity check.

The successful run is:

`C:\CodexArtifacts\pretrained-reuse-r2-cdrive-20261117\librispeech-runtime-smoke-v1-attempt-04`

The smoke explicitly confirmed that Python imported `audio_runtime.py` from the bundle directory, not from the worktree. WAV SHA-256: `61f8372eeba66f1c2aadd63d54826d88d963047ee620442f8765a0543bb81865`. Smoke JSON SHA-256: `e3ba922a23c1551bfa073eda966c4b3de894bb5b262f9943a9c03c83f4def7d1`.

- Total parameters: 55,494,785 before quantization.
- Bundle size excluding manifest: 227,115,067 bytes, about 27.1 MB above the provisional 200 MB package target. No compression or quantization was attempted.
- First model load: 0.395 s; inference: 0.278 s. Second load: 0.339 s; inference: 0.183 s. These are single-example CPU timings, not a benchmark.
- Probabilities matched across the two loads exactly (maximum delta 0).
- Probabilities matched the saved development prediction within `8.76e-12`; argmax stayed option 3.
- Typed JSON returned `decision_type: choice`, caller-ordered stable IDs, probabilities, selected ID, and confidence. Confidence was 1.0 on this one example; this is not a calibration claim.
- CUDA was not initialized and no VRAM was used.

### Preserved incomplete attempts

All attempts remain on disk. `runtime-bundle-v1-attempt-01` omitted `decision_json.py`; its first smoke harness also passed unsupported `decision_type: transcript_choice`, which the serializer rejected. `runtime-bundle-v1-attempt-02` included the serializer, but its successful smoke put the worktree ahead of the bundle in `PYTHONPATH`, so it did not prove isolated bundle imports. The builder was fixed to include the dependency; the final attempt above placed the bundle first and asserted the imported module path before inference.

## Limits and next gate

This proves a reloadable task-scoped audio+text Decision runtime, not general audio understanding or a unified text/image/audio/video/sensor model. The bundle exceeds the provisional package target and its parameter count is below the roadmap's approximate 100–200M design range; neither is being treated as a final capacity decision. Validation was used for checkpoint selection. Keep this bundle as a local research candidate; do not promote or quantize it. The next substantive R2 requirement remains a legally admitted, source-group-disjoint task where multiple synchronized inputs are shown to be necessary. Existing negative or review-only AV/sensor results do not close that gate.
