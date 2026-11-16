# Path A shared LibriSpeech runtime route — 2026-11-17

## Purpose and scope

Add the selected LibriSpeech transcript-choice readout to the existing Path A
Audio/Text/Image/Video dispatcher without loading another Whisper Tiny or
MiniLM encoder. The new `librispeech` task route is a separate task head. It
does not change the Speech Commands route or any trained weights. This is
runtime integration and one-example parity evidence only; no new quality
evaluation, training, calibration, quantization, sealed-audit access, or GPU
work was performed.

## Implementation

- Added an immutable modular-bundle extension path. It verifies the source
  assembly and creates a new output; it refuses to overwrite existing paths.
- Added an optional `librispeech` task entry to the assembly builder. It checks
  the task bundle identity and pinned Whisper/MiniLM revision and weight hashes,
  validates the readout hash and parameter total, then stores only the new
  readout plus runtime code. Whisper Tiny and MiniLM are referenced from the
  existing Audio component by hash.
- The Torch worker loads the 344,321-parameter LibriSpeech head only when the
  assembly has that task. It shares the existing frozen Whisper and MiniLM
  instances, audio cache, feature construction, and typed-choice serializer.
- The dispatcher accepts `librispeech` in raw and typed requests. Existing
  `audio` behavior retains its original default readout.
- Tests cover independent-readout scoring, typed multi-query routing, immutable
  bundle extension, hash/path verification, and preservation of the source
  bundle.

## Source integrity and outputs

The original v4 assembly and LibriSpeech bundle each contained files under
`__pycache__` that were absent from their manifests. Strict verification
correctly rejected both source directories. Nothing was removed or modified.
Two new manifest-only copies were created by checking every declared path,
byte count, and SHA-256, copying only declared files, and re-running the strict
verifiers. The unexpected bytecode remained untouched in the originals.

The final package is:

`C:\CodexArtifacts\pretrained-reuse-r2-cdrive-20261117\path-a-librispeech-integrated-assembly-v1-attempt-02`

- Source commit: `97b27eff790888d84573ee0b6a0e62da980a52ba`.
- Package-manifest SHA-256:
  `7238cf34fdbe5fe907364398bee45e3afe66676c11be6b5270dd98eb1243c8d7`.
- Components: Audio, Image, synthetic Text, CLEVRER Video feature scorer, and
  LibriSpeech supplied-transcript choice.
- Unique pre-quantization parameters: `170,844,037` (the previous assembly plus
  344,321 readout parameters). Summed per-component accounting is
  `375,139,333` because shared encoder parameters are referenced for multiple
  tasks.
- Package size: `688,661,995` bytes, versus `687,254,772` in the base assembly.
  The provisional 200 MB package target remains unmet.
- All five components and their declared dependencies passed
  `verify_modular_bundle`.

The successful source-level CPU smoke is:

`C:\CodexArtifacts\pretrained-reuse-r2-cdrive-20261117\path-a-librispeech-integrated-smoke-v1-attempt-03\integrated-smoke.json`

Its SHA-256 is
`F3F5691495463826F6AB25E8A575E176858C85F2320933B02907CA4EDF390877`.
It uses one held-speaker development-validation item already evaluated and
discussed in the previous runtime smoke. With the same four hard-negative
options and the fixed question, the dispatcher produced typed `choice` JSON;
prediction and target were both option 3. Maximum probability delta from the
saved epoch-40 prediction was `8.7522e-12`. The resident Torch worker reported
one shared MiniLM instance, loaded both audio readouts, and used CPU only.
End-to-end load and request time was 9.51 seconds on this run. This single
replay is not a speed benchmark or an additional accuracy estimate.

## Verification and limits

- Targeted CPU tests: 36 passed before final source-commit rebuild.
- Ruff and `git diff --check`: passed before commit.
- Full repository tests and remote CI are recorded after the final report/docs
  commit.
- New code commit: `97b27eff790888d84573ee0b6a0e62da980a52ba`.
- The Speech Commands scorer remains independently selected and unchanged.
  Existing task quality limitations remain: LibriSpeech only covers supplied
  transcript choice on English read audiobook speech; image and video are
  narrow development tasks; there is still no admitted natural synchronized
  audio-video task proving that both inputs are needed.
- No model was promoted. The 90–98% multi-modality objective is not established
  by this runtime integration.
