# GRID Path A frozen-feature smoke (2026-10-28)

Status: **PASS for one training and one speaker-disjoint validation media pair; no task-quality claim.** This verifies the pinned feature-extraction path on real corrected-split media. It does not close the R2 functional or quality gates.

## Provenance

- Source branch/commit: `codex/r2-audio-vibration-source-20261027`, `2721093fc72984bbf46de5501d6b42175e426d1f`.
- Corrected external split: `C:/CodexArtifacts/pretrained-reuse-r2/grid-development-split-direct-source-20261028-attempt-02/`.
- Train SHA-256: `3a4304e7158ee89380a6d4827ef23c5f1117494e10b1493bcb9a97a3fe3be816`.
- Validation SHA-256: `f05cf77258aa2ea2145eb0002fc828b9f2cdf7b5f7d0f91adea72b3204a70812`.
- Split summary SHA-256: `547e84167267428712ba5c7859869cd1eb7d558ee3502d9908f89223a77a3410`.
- Bundle index SHA-256: `fb54bb49a0f9425cd6206508e99a1d2fef25220444e3853af11aac75431861ed`.
- Split counts: 23,993 train and 7,998 validation. The source builder reports zero speaker, asset, code, normalized-target, and known-observed-code overlap. Physical cross-corpus speaker independence remains unknown.

Deterministic examples: train `s10/lwwc5a`; validation `s30/bbbx2a`. Audio was read from each speaker's individual tar in the pinned direct-source bundle index, and video from the corresponding publisher ZIP range. Both media payloads passed integrity checks and decoded to 75 ordered frames over 3 seconds. This does not use the separate aggregate `audio_25k.zip` whose speaker-folder mapping audit found filename mismatches; the smoke verifies the direct-source bundle path only.

## Results

- Frozen Whisper encoder: 8,208,384 parameters; output shape `(384,)` after pooling `(1500,384)` tokens. Features were finite; encoder parameters remained frozen. The two audio forward passes took 0.144s and 0.120s on CPU.
- Frozen VideoPrism-B: 114,365,184 parameters; pinned source revision `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`; output shape `(16,768)`. Worker parameter hash was unchanged. CPU forward time was 7.95s for first-call initialization and 2.14s for the next video; decode/preprocess was about 0.30s each.
- Runtime: Python 3.11.9, Torch 2.6.0+cu124, Transformers 5.6.2, PyAV 18.1.0, NumPy 2.4.6. CPU only; no training, GPU, checkpoint selection, or sealed audit.
- External receipt: `C:/CodexArtifacts/pretrained-reuse-r2/grid-path-a-frozen-feature-smoke-20261028-attempt-03/result.json` (SHA-256 `cd9165e4e891f9628525226517dc10ce5be6f28c96e187553ac95a3f67f98dcf`). Runner snapshot: same directory's `run_smoke.py` (SHA-256 `a1a4deb740e158bbec95388848d1d58b252d8a10d815eb93ce7d2f87209205fd`).

This is a technical smoke, not a decision evaluation. No features were used to fit a model. Next, construct and freeze a group-safe audio-video correspondence training cohort from train speakers and a new fixed validation cohort from held-out speakers, then train only a small fusion/readout head over these frozen features. Keep this GRID result internal research until product rights are clarified.
