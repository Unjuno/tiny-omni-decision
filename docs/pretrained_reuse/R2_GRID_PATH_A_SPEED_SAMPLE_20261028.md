# GRID Path A VideoPrism CPU timing sample (2026-10-28)

Status: **three fixed train-split video encodes passed; this is a runtime estimate, not a quality result.**

## Scope and provenance

- Branch/commit before the sample: `codex/r2-droid-temporal-transfer-20261028` / `4509bcb04ba7e676c4e59fd3c9897e6d4230a504`.
- Source split: corrected direct-source GRID train manifest, SHA-256 `3a4304e7158ee89380a6d4827ef23c5f1117494e10b1493bcb9a97a3fe3be816`.
- Video archive set: completed direct-source archive acquisition, verified by the full media duplicate audit.
- Encoder: VideoPrism-B, 114,365,184 parameters, source revision `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`, checkpoint SHA-256 `61fa62eedbe8f4d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- Backend: the existing pinned native JAX/TF CPU worker; no GPU, training, model update, or sealed-audit access.
- Fresh external output: `C:/CodexArtifacts/pretrained-reuse-r2/grid-path-a-speed-sample-20261028-attempt-03/`.
- Result SHA-256: `436414afd9dcfddfa3b9a99e61eca920f2f3cb32a736b020de2d0d7d51039405`.
- The failed preflight-only attempts `attempt-01` and `attempt-02` are retained; neither initialized the model or produced features.

## Measured samples

| Train asset | End-to-end worker request seconds | Decode/preprocess seconds | Encoder seconds |
|---|---:|---:|---:|
| `s2_bbaf1n` | 9.1895 | 0.3412 | 8.8355 |
| `s3_bbaf1s` | 2.4946 | 0.3319 | 2.1480 |
| `s5_bbae8n` | 2.4682 | 0.2741 | 2.1788 |

All three source videos decoded to 75 timestamp-ordered frames over three seconds and yielded finite `[16,768]` native VideoPrism features. The arithmetic mean is 4.7174 seconds/video. The first request includes JAX's one-time compile/warm-up; the two subsequent requests average 2.4814 seconds/video.

## Planning estimate and limitations

The corrected split contains 31,991 unique assets. Extrapolating the steady-state sample gives roughly 22.0 hours for video encoding alone (`9.19 + 31,990 * 2.4814` seconds), before audio extraction, validation, write overhead, or interruptions. This is an early estimate from three clips and is not a guaranteed ETA. The existing native worker has a ten-minute safety lifetime, so it cannot process the full split in one process. A resumable/chunked extraction path must be implemented and tested before full extraction. The three measured clips are retained in the external attempt; their feature outputs are not treated as training evidence.
