# GRID corrected direct-source pilot: matched-input technical smoke

Status: **PASS for the single-quartet input/runtime checks; no quality claim.**
This is not the 64-quartet validation evaluation and was not used for model
selection.

## Fixed inputs and run

The smoke used the first row of the corrected, pre-inference selection:

- selection SHA-256: `e897ff02a435a463211f0a157495058adcb5c390bf384437e3e56dc8066da281`
- speaker: `s30`
- quartet ID: `7b0ef094781e8b04f0d3c40bf4a1371142285565d5a77803a9080594859708e4`
- four balanced audio queries and four matching candidate videos
- fixed Whisper encoder + Lombard audio head, VideoPrism-B + Lombard video
  head, and the already frozen native correspondence calibration

The direct s30 audio archive SHA-256 is
`9fafc520f218bc0e1b1b14e0b396c5470f4b0cb908d3712c3ca7b52de5d34792`; its
matching direct alignment archive SHA-256 is
`7bee9636e68e34661dd8bef06d815e8ac7d31aa09a3b6ed7664fb0de3e337b94`. The
s30 video ZIP SHA-256 is
`69e7c2ef14b1bd4f2d0012a8a13fe8e9375c4f11a03c4de9f03f4071f48f12fd` and its
Zenodo publisher MD5 had previously matched. Each selected ZIP member CRC was
verified by `zipfile` while reading. The inference controller and worker check
the pinned model, source-code, processor, calibration and dependency hashes.

The reproducible command is:

```powershell
python -m scripts.smoke_grid_direct_speaker_quartet `
  --selection C:\CodexArtifacts\pretrained-reuse-r2\grid-av-direct-official-pilot-20261028-attempt-03\selected.jsonl `
  --audio-tar C:\CodexArtifacts\pretrained-reuse-r2\grid-official-speaker-audio-20261028-attempt-01\s30.tar.part `
  --alignment-tar C:\CodexArtifacts\pretrained-reuse-r2\grid-official-speaker-audio-20261028-attempt-01\s30-align.tar `
  --video-zip C:\CodexArtifacts\pretrained-reuse-r2\grid-validation-video-20261028-attempt-01\s30.zip `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\grid-av-direct-technical-smoke-20261028-attempt-01
```

## Result

The run processed four audio/video query roles in 50.41 seconds on CPU; the
VideoPrism worker took 45.56 seconds. No GPU, training, calibration change,
sealed audit or checkpoint selection was used. All JSON probability vectors
were finite and normalized. Reordering candidates together with their opaque
IDs preserved the output exactly, and each explicit missing-modality policy
returned the configured uniform distribution.

For these four roles, the scorer chose the **same video candidate every time**;
that candidate was the target for only one of the four balanced roles (1/4 in
this quartet). This is a single technical-smoke observation, not an estimate
of task accuracy. The per-query probabilities and mismatched-input controls
are preserved in the external receipt:

`C:/CodexArtifacts/pretrained-reuse-r2/grid-av-direct-technical-smoke-20261028-attempt-01/summary.json`

Receipt SHA-256:
`436b7cf5a214ec549c17d2c6efd52a0e6f6ad3ef24d041393ddd3b1dbabd0283`.
Worker result SHA-256:
`30caf3b500a1916e2692336390c00286472864b3c371f6d4b4cb7a2be281eb88`.

## Subsequent full evaluation

The frozen 64-quartet / 256-role development sample subsequently completed
with matched audio/video and uniform missing-input, option-order,
mismatched-audio and mismatched-video controls. The actual result, full metrics,
speaker-grouped intervals, source hashes and resource measurements are in
[`R2_GRID_DIRECT_FIXED_AV_EVALUATION_20261028.md`](R2_GRID_DIRECT_FIXED_AV_EVALUATION_20261028.md).
The original 48-minute CPU projection was too high: batched VideoPrism worker
execution completed the evaluation in 894.12 seconds. Do not tune this sample
or promote a product model.
