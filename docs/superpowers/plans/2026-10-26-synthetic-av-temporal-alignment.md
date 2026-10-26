# Synthetic AV Temporal Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify that generated audio event order and timestamps align with the decoded video frame sequence, and record the already-tested readout comparison precisely.

**Architecture:** Add one CPU test that decodes the generated WAV segments and checks their dominant tone against the manifest and frame times. Document the fixed eight-step inputs, the existing readout alternatives, and the limits of the exposed synthetic validation. Do not retrain or choose a checkpoint.

**Tech Stack:** Python, pytest, NumPy FFT, Python `wave`, PyAV video renderer/decoder.

**Spec:** `docs/superpowers/specs/2026-10-26-synthetic-av-temporal-alignment.md`

## Global Constraints

- Keep the task synthetic and development-only.
- Keep each AV sequence at 8 steps, with 1 frame and 1 audio segment per 0.5-second step.
- Do not alter existing corpora, feature caches, checkpoints, or reports.
- Do not access sealed audit data or run another training/selection experiment on the exposed validation.
- Keep all new generated media confined to pytest temporary directories.

## Review Focus

- A reordered video frame must differ from its source frame at that index; covered by `test_video_extraction_decoder_preserves_every_rendered_symbol_in_order`.
- A reordered WAV segment must decode to a different symbol sequence; covered by `test_rendered_audio_symbols_keep_order_and_align_with_video_frames`.
- Incorrect segment duration or frame cadence must fail temporal alignment; covered by the same audio alignment test.
- Missing optional AV/NumPy/Pillow dependencies must skip only the integration checks; existing test helpers use `pytest.importorskip`.
- Passing synthetic integrity tests must not be presented as natural-media quality evidence; covered by the design note's scope and limits.

---

### Task 1: Verify rendered audio/video temporal alignment

**Files:**
- Modify: `tests/test_synthetic_av_task.py`
- Create: `docs/superpowers/specs/2026-10-26-synthetic-av-temporal-alignment.md`
- Create: `docs/superpowers/plans/2026-10-26-synthetic-av-temporal-alignment.md`
- Modify: `ROADMAP.md`

**Interfaces:**
- Consumes the `render_assets()` manifest and files produced by `scripts/prepare_synthetic_av_readout.py`.
- Produces a regression test proving 8 ordered tone segments match the 8 video symbols and share 0.5-second timestamps.

- [x] Add the WAV sequence/timestamp integration test and run it against a temporary generated cohort.
- [x] Run the frame-order and AV readout contract tests together (25 passed).
- [x] Record the exact feature granularity, readout alternatives, and limitations in the spec and roadmap.
- [x] Run Ruff and `git diff --check`, then commit the test and documentation.

