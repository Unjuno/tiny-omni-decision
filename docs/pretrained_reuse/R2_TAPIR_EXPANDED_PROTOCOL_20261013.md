# TAPIR expanded train-only correspondence protocol — 2026-10-13

Extend the fixed dense diagnostic to the first eight eligible clips in the frozen99-video train manifest. Eligibility: exactly one occluder track, at least eight unmasked annotations before reveal cut (or full clip for upstream cut=-1), first annotation at frame0. No quality selection; no exclusions occurred before reaching eight. Initial box center is an oracle. Future annotations are scoring-only. Select eight evenly indexed eligible annotation entries; run all original frames through the last selected frame. Same TAPIR checkpoint, constructor, RGB256 normalization, four CPU threads and exact seven evaluation positions per video. No training, validation, calibration or confidence filtering.

Plan: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-tapir-expanded-plan-01/config.json`, SHA-256 `977cfcd414db9a34af1f88a9089e2a5e6f1eb65d7e89dc32c57ac7d0199e5b32`.

| Video | Input frames | Execution |
|---|---:|---|
| video_10103 | 451 | reuse completed dense result |
| video_10186 | 631 | reuse completed dense result |
| video_10212 | 1051 | new |
| video_10387 | 481 | new |
| video_1087 | 571 | new |
| video_10886 | 631 | new |
| video_10894 | 541 | new |
| video_10903 | 541 | new |

Report all videos' normalized center error, static-center error, point-in-box count and inference time. Preserve failures; do not replace difficult clips. Eight train clips only screen transfer feasibility and cannot meet a Decision-quality gate. Do not interpret tracks as hidden-object answers or as automatic object discovery. Current execution is running; no result claimed for the six new clips.

Runner: `C:/CodexArtifacts/pretrained-reuse-r1/tapir-audit-01/track_expanded01.py`; new output `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-tapir-expanded-tracks-01/`. Existing2-video results, model weights and all legacy artifacts remain unchanged. No GPU or paid compute. The selected prior results are copied into the new summary, not recomputed.
