# Oracle-initialized token correspondence — 2026-10-13

Two fixed train videos, no fitting. Exactly eight nonmasked annotated frames per video, evenly indexed over annotations before the existing reveal cut. Sampling differs from the original dense experiment and is a separate diagnostic. Initial box center selects one16x16 token; each later frame selects maximal cosine similarity to that fixed token. No search over feature layers, prompts or thresholds. Future-frame boxes are scoring-only.

The original dense4fps frames match only one occluder annotation per selected video. Thus the previously reported aggregate9,115 exact boxes across99 videos did not provide enough temporal correspondences for these two particular tracks. No interpolation was used.

| Video | Token center error | Static exact error | Static grid error | Token in box | Static grid in box |
|---|---:|---:|---:|---:|---:|
| video_10103 | 0.065815 | 0.036967 | 0.048567 | 5/7 | 5/7 |
| video_10186 | 0.171179 | 0.179315 | 0.174148 | 2/7 | 3/7 |

Errors are Euclidean distance in normalized image coordinates. Exact static baseline uses the initial box center; grid static baseline uses the same initial token center as the token tracker. Different box sizes make point-in-box an incomplete metric. Two videos do not support a population confidence claim.

Execution passed, correspondence quality did not establish reliable tracking. The mostly stationary video favors the static control; the moving video has large errors and only2/7 token predictions in the annotated box. This closes this simple nearest-token route without scaling it into a full training run. It does not prove frozen tokens lack all spatial information or establish an encoder-family ceiling. Bidirectional video encoding sees the whole eight-frame clip; this is not online causal tracking. No hidden-object Decision labels are predicted.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-track-tokens-01/`. Separate config, token arrays, predictions and summary/hash ledger retained. Runner: `scripts/diagnose_occlusion_track_tokens.py`; CPU JAX, pinned VideoPrism weights/source, preserved legacy artifacts. No paid compute, validation or sealed audit.
