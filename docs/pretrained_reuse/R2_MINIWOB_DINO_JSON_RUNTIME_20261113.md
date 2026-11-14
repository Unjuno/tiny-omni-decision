# MiniWoB DINO Path A JSON runtime (2026-11-13)

## Scope

Added a local inference API that connects the already selected frozen DINOv2
small encoder and MiniWoB option scorer to the repository's supplied-option
probability JSON contract. This is an inference/runtime handoff check for the
existing synthetic `click-color-v1:screen-only` result. It is not a new fit,
checkpoint selection, benchmark result, or evidence for natural images, video,
audio, text, sensor fusion, or general Omni quality.

The input is one 160x210 screenshot and at least two options carrying an opaque
ID, display text, and normalized candidate coordinates. DINOv2 sees only the
whole-screen-resized RGB tensor; the scorer sees CLS, mean patch, local 3x3
patch, and x/y features. Option text is returned but is not a model input. The
runtime returns the standard choice JSON with normalized probabilities.

## Integrity and verification

- Encoder and readout are loaded from local paths only. Their supplied SHA-256
  values are checked before loading; the encoder is required to have exactly
  22,056,576 frozen parameters.
- `tests/test_miniwob_dino_runtime.py` covers normalized JSON output,
  candidate-order equivariance, duplicate-ID rejection, and screen shape.
- Together with the existing readout tests, Ruff passed and 7 focused tests
  passed.
- A separate CPU process loaded the pinned local DINOv2 checkpoint and the
  existing selected readout, then ran the first frozen TRAIN screenshot
  through pixels-to-JSON. The selected option matched the saved TRAIN replay;
  maximum probability difference was `1.252e-6`, and output probability sum
  was `1.000000004`. This is a single-example execution/parity smoke, not a
  new quality estimate.
- No CUDA, training, validation scoring, sealed-audit access, checkpoint
  writes, external model downloads, or cloud compute occurred.

## Remaining limits

This runtime requires the caller to supply MiniWoB-specific candidate
coordinates and local artifact hashes. It does not parse arbitrary screen
semantics, generate action candidates, or establish quality outside the
already evaluated synthetic task. The DINO image component remains separate
from the pinned VideoPrism video path; R2 M-2 and broader task-quality gates
remain open.
