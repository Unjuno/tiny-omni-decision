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

## Existing-route regression replay

The new five-component assembly was also used for the existing frozen
Video/Image/Text/Audio dispatcher replay, entirely on CPU, with the
pre-existing fixed development inputs. No labels were used for selection and
no checkpoints were changed. Video, Image, and Text passed saved-probability
parity and typed-JSON checks. Audio produced typed JSON and preserved all four
argmaxes, but its maximum probability delta was `6.1774e-4`, above the
unchanged `1e-5` parity gate. The run therefore correctly reports
`failed_parity`; adding LibriSpeech did not resolve or mask the known CPU
FP32-versus-saved Audio numeric discrepancy.

| Existing route | Max probability delta (first / warm) | Argmax mismatch | Typed JSON | First / warm dispatch (s) |
| --- | ---: | ---: | --- | ---: |
| Video | 0 / 1.49e-7 | 0 / 0 | pass | 29.232 / 0.018 |
| Image | 1.28e-8 / 1.19e-7 | 0 / 0 | pass | 1.396 / 0.005 |
| Text | 5.66e-7 / 5.66e-7 | 0 / 0 | pass | 0.074 / 0.087 |
| Audio | 6.18e-4 / 6.18e-4 | 0 / 0 | pass | 1.077 / 0.002 |

The sequential replay took 35.286 seconds. Its report is
`C:\CodexArtifacts\pretrained-reuse-r2-cdrive-20261117\path-a-librispeech-integration-four-route-replay-20261117-attempt-01\dispatcher-benchmark.json`, SHA-256
`53c1189eb01e5da12d7c6a8c596df7412236633073b9fbe17e2d49b2783d98ae`. It
used assembly manifest
`7238cf34fdbe5fe907364398bee45e3afe66676c11be6b5270dd98eb1243c8d7` and
source commit `9d807df88c9e9161fe955f964bfabf978ff07be6`.

## Verification and limits

- Targeted CPU tests: 36 passed before final source-commit rebuild.
- Full CPU suite: 2,921 passed, 11 skipped.
- Ruff: passed; training catalog validation: 6 sources; evaluation catalog
  validation: 7 sources; `git diff --check`: passed.
- GitHub Actions was triggered for the pushed PR head; its final result is
  tracked on PR #111.
- Existing-route regression replay: Video/Image/Text parity passed; Audio
  probability parity failed the unchanged gate as documented above. All four
  routes returned typed JSON with no argmax mismatch.
- New code commit: `97b27eff790888d84573ee0b6a0e62da980a52ba`.
- The Speech Commands scorer remains independently selected and unchanged.
  Existing task quality limitations remain: LibriSpeech only covers supplied
  transcript choice on English read audiobook speech; image and video are
  narrow development tasks; there is still no admitted natural synchronized
  audio-video task proving that both inputs are needed.
- No model was promoted. The 90–98% multi-modality objective is not established
  by this runtime integration.
