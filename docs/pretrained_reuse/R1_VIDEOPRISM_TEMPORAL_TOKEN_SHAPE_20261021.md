# VideoPrism temporal-token shape probe — 2026-10-21

## Result

A CPU-only forward of the pinned VideoPrism-B checkpoint on one previously
selected 8-frame Perception Test development-train clip returned a finite
token tensor of shape `[1, 2048, 768]` and an empty auxiliary dictionary. The
official encoder implementation specifies outputs as `[B, T*N, D]`; the base
model documentation says its 16-frame output reshapes to `[B,16,16,16,768]`.
Thus the observed 8-frame sequence has 256 spatial tokens per frame after the
temporal encoder. The upstream code rearranges each spatial position through
its temporal Transformer and restores time-major `T*N` order before output.

The existing `extract_perception_temporal_match_videoprism.py` averages axis 1
and saves only a `[768]` vector. That removes explicit time and spatial-token
alignment. Therefore, the earlier mean-pooled small-readout results do not
test an order-preserving temporal Decision head. The probe does **not** show
that the encoder understands the task or that an LSTM/GRU/readout will help.

## Reproducibility and limits

- Project source at probe: `b844a825ade5ee6b28802c9eb50af9797a23e59a`.
- VideoPrism source: `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`.
- Model: `videoprism_public_v1_base`, 114,365,184 parameters.
- Checkpoint: `C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz`,
  SHA-256 `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- Input: `video_2738`, 8 synchronized frames, maximum target/PTS error
  14.73 ms; video SHA-256
  `505d4684f8cf321ae01c35588c3fb2384c90fdeeeed6be3842447aeefc352fbd`.
- Feature manifest SHA-256:
  `2cb6a492aa671d544f9e102a44619a0c89bf4819e7eabda74168561b87081d4e`.
- Runtime: Python 3.11 environment, JAX 0.6.2 on CPU, TensorFlow 2.18.1;
  model load, decode, and forward took 3.922 s. No GPU was used.
- Output: `C:\CodexArtifacts\pretrained-reuse-r2\videoprism-temporal-token-shape-smoke-20261021-03\result.json`,
  SHA-256 `60afcf556fe341b49705865392a15d47c6d6ff02c84f1f42cb36a43aa9e47a1d`.
  The exact one-off probe script is beside it, SHA-256
  `22655b7d8c0193f7eeee34f2ba9202c743c902b28248cde0e2c4f3a636209ccf`.
- The first two isolated attempt directories were retained unchanged. Attempt
  01 stopped before model load because the source manifest has no
  `media_video_sha256` field; attempt 02 stopped before model load because it
  referenced the wrong project worktree path. Attempt 03 used the hash from the
  already saved feature result and the actual current project worktree.

This is a shape/runtime diagnostic on one already observed train input, not a
blind evaluation. No labels or predictions were used. No existing feature,
checkpoint, run, split, or dataset file was modified. No training or
checkpoint selection occurred.

## Next evidence needed

The result supports testing a small temporal readout only after obtaining or
freezing a fresh grouped development split. First compare per-frame spatial
pooling `[T,D]` with the current global mean and a parameter-matched temporal
control; retain best-unimodal and modality-shuffle controls. Keep the current
validation cohort closed to repeated tuning. Full token extraction would use
about 6.29 MB per 8-frame clip in float32 before metadata, or about 3.22 GB for
512 clips; this cost and its CPU impact need a measured preflight before any
bulk run. Orbitune still owned the GPU at the last live check, so no bulk
feature extraction or GPU work was started.
