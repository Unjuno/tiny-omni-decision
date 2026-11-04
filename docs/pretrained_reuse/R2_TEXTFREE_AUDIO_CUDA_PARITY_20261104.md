# Path A text-free audio CUDA feature parity — 2026-11-04

## Result

The raw-WAV Whisper Tiny runtime reproduced the frozen CUDA-generated TRAIN
features exactly for four fixed Speech Commands training clips from four
speaker groups. The readout probabilities also matched exactly. This is an
implementation/runtime parity check, not a quality evaluation.

| Measure | Result |
|---|---:|
| Feature records / distinct speakers | 4 / 4 |
| Maximum absolute feature delta | 0.0 |
| Maximum absolute option-probability delta | 0.0 |
| Top class unchanged | 4 / 4 |
| Runtime time in audit script | 0.846 s |
| Peak PyTorch CUDA allocated memory | 60,880,896 bytes |

The previously recorded CPU BF16 pass on these same four samples had feature
drift of 0.00998–0.02127 and a maximum probability delta of 0.000967. The exact
CUDA match indicates the earlier discrepancy was backend-specific for this
small set; four examples do not establish general numerical parity across all
audio inputs.

## Frozen inputs and provenance

- Source: Speech Commands v0.02, `google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da`; TRAIN manifest SHA-256 `02a43981e153b0f63bd55192f614fe504114f05f212ca0f4da37e2d91480a9da`.
- Audio encoder: `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`; encoder state SHA-256 `d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8`.
- Readout SHA-256: `46808a882ab75f8f19a138e4a91d13ceb42cfdc82534c4608c42ee615774d3cb`.
- Frozen feature cache SHA-256: `ace323458e7f443daf63382d2d41d99151d2a5801c671d30b0df68d52338ae64`.
- Code commit: `fef96cb4a6eb002e0b012ace327f2dc7f4268d5d`.
- Environment: Python 3.11.9; PyTorch 2.6.0+cu124; NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB. The audit used the existing BF16 CUDA path. No cloud or paid compute was used.

The four IDs and media hashes are preserved in the external run record. The
runner parsed the frozen `train.jsonl` manifest to verify source/split and
select distinct speaker groups, then read only those four media files. It did
not use target values for selection, inference, or metrics; the manifest parser
does deserialize the full TRAIN rows. No validation manifest or sealed audit
was opened. No weights, media, or feature cache were modified; no training or
calibration ran.

## Reproduction

Run from the repository root at the code commit above, with `PYTHONPATH=src`:

```powershell
python -m scripts.verify_textfree_audio_patha_feature_parity `
  --sample-dir C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-sample-seed17-128-64-retry2 `
  --feature-cache C:\CodexArtifacts\pretrained-reuse-r2\audio-probe-v0-retry3\audio-features.pt `
  --encoder-dir C:\CodexArtifacts\pretrained-reuse-r1\whisper-encoder-runtime-01 `
  --readout C:\CodexArtifacts\pretrained-reuse-r2\textfree-audio-decision-03\best-readout.pt `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\textfree-audio-patha-cuda-parity-20261104-attempt-02 `
  --examples 4 --device cuda
```

The external JSON report is
`C:\CodexArtifacts\pretrained-reuse-r2\textfree-audio-patha-cuda-parity-20261104-attempt-02\parity-report.json`,
SHA-256 `5723576c2c345435fb76e75048a8278821d10534821cd2a78fadf3221bdc1c74`.
It records `PASS_CUDA_FEATURE_PARITY`, exact feature and probability parity,
and the per-example hashes. The package utility now accepts `--device cpu|cuda`,
fails closed if CUDA is unavailable, and places cached reference features on
the model device before readout.

## Scope limit

This validates numerical consistency on four TRAIN clips only. It provides no
accuracy, calibration, OOD, task-generality, or multimodal-fusion evidence.
The route remains a closed ten-keyword classifier and does not support
arbitrary audio questions or arbitrary candidate text.
