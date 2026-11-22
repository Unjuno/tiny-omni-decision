# R7 spoken-palette raw-input JSON runtime smoke — 2026-11-22

## Result

Built a separate CPU runtime bundle for the fixed spoken-color-to-screen-action
pilot, then ran one raw-input mechanics smoke using the first frozen TRAIN
example only. The smoke decoded the original WAV and generated screen image,
loaded the frozen encoders and scorer, and returned finite, normalized JSON
probabilities in the supplied option order.

This is **not a quality evaluation**. The smoke deliberately did not read the
target label, select a checkpoint, evaluate a validation/evaluation split, or
open sealed audit data. It does not establish natural audio/video quality,
generalization, or product readiness.

## Provenance and artifacts

- Source branch: `codex/pretrained-reuse-next-20261115`.
- Base source commit: `4db542ad7ff1e862450d8f849c04dca9a9503ea9`.
- Exact implementation files are hash-recorded in the final smoke receipt and
  package manifest; the worktree changes are committed separately after tests.
- Audio source: Sheffield/Lombard-Grid, pinned CC BY 4.0 manifest
  `5876e57a4c71108fe3483fde6e18f3977121942c9208567488f042247626577d`.
- Task corpus manifest SHA-256:
  `b75a9e5ea629b8e1d8cb188ef4ee9c9a9b3e784b0f387ce9eed0a27072d72c7a`.
- First TRAIN JSONL SHA-256:
  `7c2988c5d688d89e605c8e9683cb5db1155faf127f6a2b3e61e1d72ad71aa170`.
- The smoke input was source `s11_l_bbab2n`, split `train`; its audio bytes
  SHA-256 is `8408220551860f06b8ea61187802066a0b784578f88e130bfe9fdbec97cedd87`
  and generated screenshot SHA-256 is
  `de0cee89cdf50832df82e16cf8bf83afafd3293d3e7265920deaf75bb1dc2f53`.
- Selected checkpoint SHA-256:
  `48b4a12401f9da88472b290536ca5e3fe3b708319af9d221caa5ce4b75e9c1b5`.
- Dynamic INT8 DINOv2 state SHA-256:
  `cd8083402b68ed41f9a3367258698ba7cc70c455b58f31828e29346e281381d8`.
- Calibration temperature: `0.4061261981781178`, selected earlier on the
  already-observed Calibration split. Its metrics are in-sample diagnostics,
  not independent evidence.

Final bundle:
`C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-json-runtime-attempt-04\`

- Runtime manifest SHA-256:
  `185ff291899c0bc280ccf087550e1bbf9463401f24c76a4013b18994baebd854`.
- Model/config/license payload at build time: `57,969,347` bytes
  (`55.28 MiB`). No WAV, screenshot, corpus rows, or dependency wheels are
  included in the bundle.
- Logical pre-quantization parameter count: `30,363,521` total: Whisper
  encoder `8,208,384`, DINOv2-small `22,056,576`, and Decision scorer `98,561`.
  DINOv2 Linear modules use dynamic qint8 weights; Whisper, DINO embeddings and
  norms, scorer, and normalizer remain float32.
- The bundle depends on an external Python/ML environment. It is not a
  dependency-inclusive installer or a target-device release package.

Raw-input smoke receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-json-runtime-attempt-04\runtime-smoke-train-first-clamped.json`

- Receipt SHA-256:
  `969f097380530635ab9b9510f9663da0e57c8b667c34ef575ee7b73b7aaab223`.
- Result: `PASS_RAW_INPUT_TRAIN_ONLY_SMOKE`; no model updates.
- CPU: Intel Core i7-12700H; 4 PyTorch threads.
- Environment: Python `3.11.9`, PyTorch `2.6.0+cu124`, Transformers `5.6.2`,
  safetensors `0.8.0`, NumPy `2.4.6`, Pillow `10.4.0`, PyAV `18.1.0`.
- Bundle load was `0.888 s`; one warmed raw-input decision call was `0.318 s`.
  This single sample is a mechanics measurement, not a latency benchmark.
- CPU FP32 Whisper pooled feature versus the frozen training feature had max
  absolute error `9.06e-6`, mean absolute error `1.06e-6`, and cosine similarity
  `1.0` after clipping floating-point roundoff. The original feature run used
  CUDA BF16; this is a one-sample parity diagnostic, not bitwise identity.
- DINO dynamic qint8 state was rebuilt from the pinned config, loaded with
  `weights_only=True`, and strict-loaded into the CPU quantized architecture.
  A separate construction check observed 72 dynamic-quantized Linear modules.

## Implementation and verification

Added:

- `src/tiny_omni_decision/spoken_palette_runtime.py`: strict bundle hashing,
  frozen CPU loading, PCM16/float32 WAV decoding, Whisper pooling, DINOv2 option
  features, shared scorer, calibration temperature, and typed JSON output.
- `scripts/pretrained_reuse/package_spoken_palette_runtime.py`: fail-closed
  source-hash checks, source notices, non-overwriting output, and payload
  accounting.
- `scripts/pretrained_reuse/smoke_spoken_palette_runtime.py`: one TRAIN-only
  example, source-member checks, no label use, and runtime/parity receipt.
- `tests/test_spoken_palette_runtime.py`: input decoding, malformed WAV,
  bundle-path/hash, option validation, and JSON order/probability tests.

The on-device WAV was `pcm_f32le` (IEEE float32), not PCM16. The runtime was
expanded to accept that actual source format as well as mono PCM16, both at
16 kHz and capped to the Whisper 30-second window. The smoke showed the complete
path from raw audio bytes and image to calibrated option-probability JSON.

An initial packaging/smoke sequence exposed a direct-script import issue, then
the float32 WAV format, the calibration-temperature field location, and the
default `requires_grad` state of a freshly constructed scorer. These were fixed
in code. Earlier non-final attempt directories `attempt-01` through `attempt-03`
were preserved; `attempt-04` is the first passing raw-input smoke artifact.

Validation on the final code state:

- `ruff check .`: passed.
- `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q`: `3300 passed, 11 skipped`.
- GPU was not used by packaging, smoke, or tests. The separate GPU process was
  left untouched.
- No training, quantization update, model selection, sealed audit, or
  evaluation-split access occurred in this task.

## Remaining R7 and product gates

This advances raw-input loading and typed-output mechanics, but R7 remains
open. The bundle is an audio + generated-image pilot, has no Video input, has no
target deployment-device benchmark, excludes runtime dependencies, and was not
quality-scored in this smoke. The selected calibration temperature was fit on
an exposed Calibration split. The earlier exposed development INT8 result had
lower Accuracy and worse NLL/Brier/ECE than its float32 DINO baseline; this
runtime smoke cannot reverse that result.

R2-M also remains open: no newly admitted, rights-cleared, independent task has
shown that natural synchronized inputs jointly improve a Decision. Nothing in
this artifact establishes the broad per-modality 95–98% target. Keep the pilot
as a narrow executable reference, not as a product Teacher or quality pass.
