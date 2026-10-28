# GRID Path A feature extraction smoke (2026-10-28)

Status: **PASS for the current extractor on real direct-source train and validation media. No fitting or quality claim.**

## Scope

The new `scripts/extract_grid_path_a_features.py` extracts 384-dimensional frozen Whisper utterance features and ordered `[16,768]` native VideoPrism temporal features. It validates the corrected split/cohort/media receipts, writes exact feature IDs and targets, checks finite feature arrays, records media/feature hashes, and supports explicit resume only when config, contiguous IDs and completed feature hashes agree. Video work is chunked at 150 assets so it remains below the pinned worker's ten-minute lifetime. The worker also verifies its frozen parameter hash and all input-file hashes at shutdown. The extractor uses CPU only.

## Executed smoke

- Command: `python -m scripts.extract_grid_path_a_features --output C:\CodexArtifacts\pretrained-reuse-r2\grid-path-a-features-smoke-20261028-attempt-03 --smoke`.
- It processed two fixed training and two fixed validation assets from the frozen cohort. Output shapes: audio `[4,384]`, video `[4,16,768]`, target matrix `[4,6]`.
- Result: `PASS_GRID_PATH_A_FROZEN_FEATURES`, 51.88 seconds, `gpu_used=false`, `sealed_audit_opened=false`.
- VideoPrism loaded 114,365,184 parameters; worker before/after state hashes match. Whisper loaded 8,208,384 frozen parameters.
- Result SHA-256: `6d0be08327b94b0e10651e0ca3ac31aed962f44c5825aee9be131a0e21047086`.
- Effective config SHA-256: `49b260dc5913833bf1d17ef49da67281eabf56d1ad829dbb1b252c918cd7e0dc`.
- Extractor source SHA-256: `de29613e31667f815ac04afdb8d0a3b5d0ffd3292dc896699654282e682ab1ee`.
- Output is external at `C:/CodexArtifacts/pretrained-reuse-r2/grid-path-a-features-smoke-20261028-attempt-03/`; raw media temporary files were removed after worker terminal verification.

An earlier smoke attempt stopped before any model ran because the new script had a mistyped VideoPrism hash; the actual file and the prior worker's pin were checked, the constant corrected, and a fresh output used. This failure and the successful fresh attempt are preserved separately.

## Still pending

This smoke validates code, loading, alignment, array persistence and worker integrity only. The full fixed cohort remains 768 train and 256 held-out utterances; extracting it is estimated at about 45–50 minutes with worker restarts. Training the audio-only, video-only and joint readouts, then evaluating matched/mismatched controls, has not started.
