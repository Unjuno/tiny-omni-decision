# VideoMAEv2 token correspondence — 2026-10-13

Two fixed train tracks from the existing VideoPrism diagnostic; no training or validation. Same eight frame identities before reveal, repeated twice each for the sixteen-frame model input. Initial annotated box is an oracle; later boxes score predictions only. Frozen final-block tokens, 8x14x14x384, cosine match to the fixed initial token over each later grid. Bidirectional clip encoding sees the entire clip; this is offline correspondence, not causal tracking or hidden-object Decision accuracy.

| Video | Token center error | Static grid error | Token point in box |
|---|---:|---:|---:|
| video_10103 | 0.081082 | 0.052713 | 2/7 |
| video_10186 | 0.238457 | 0.174308 | 2/7 |

Errors are Euclidean normalized-image distances. Static control retains the same initial quantized14x14 grid center. Both videos favor the static control in mean error. These two examples do not support population claims; checkpoint/preprocessing/grid differ from VideoPrism, so cross-model differences are not architecture-only effects. Preserve the negative result and do not scale this simple nearest-token route into a training pipeline. Spatial tokens may support other readouts; this experiment does not establish an encoder ceiling.

Final-block capture is verified by exact equality of `fc_norm(tokens.mean(1))` and the official feature output. Pinned source/checkpoint/media and reused annotation-plan hashes are checked. CPU four threads, FP32, no environment updates or GPU. Existing artifacts remain unchanged.

Runner: `scripts/diagnose_videomaev2_track_tokens.py`. Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-videomaev2-track-tokens-01/` (token arrays, predictions, input/source/script hashes, exact pooling verification). The process completed successfully; Ruff passed. This diagnostic closes the direct nearest-token hypothesis for these fixed examples, leaving learned correspondence and transferable temporal reasoning unproven.
