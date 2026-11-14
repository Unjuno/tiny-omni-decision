# Fresh-speaker Speech Commands transfer — 2026-11-10

## Result and scope

One locked, task-specific transfer evaluation completed on 1,000 clips from
562 previously unobserved speaker groups in the official Speech Commands v0.02
TRAIN split. The frozen Whisper-Tiny encoder and calibrated 10-way keyword
readout reached 93.8% accuracy, NLL 0.229046, Brier 0.086609, ECE-15 0.023717,
and mean confidence 0.945438. A 10,000-replicate speaker bootstrap (seed 17)
gave a two-sided 95% accuracy interval of 91.76%–95.62%.

This is evidence for transfer on limited-vocabulary English keyword
classification only. The point accuracy is below the provisional 95% target;
the interval is not evidence that general audio decision quality is at least
95%. This cohort is now observed and must not be used for training,
temperature-fitting, checkpoint selection, or repeated model selection.

## Frozen model and source

- Source: `google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da`,
  v0.02, official TRAIN records only; source card declares CC BY 4.0.
- Encoder: `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`,
  8,208,384 parameters, checkpoint SHA-256
  `d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8`.
- Frozen readout: 3,850 parameters; 8,212,234 resident parameters total;
  SHA-256
  `171f6a9303b2fd9ad63efe2bee6af119549fb4f26ea663e9c1297f51a678e1c3`;
  fixed temperature `0.42702477413437295`.
- The evaluation cohort was excluded by speaker, source path, source row, and
  exact WAV hash from the available historical exposure inventory; selected
  cohort PCM had 1,000 unique contents. Historical normalized-PCM overlap is
  UNKNOWN because the prior raw WAVs are unavailable.
- Neither official validation/test shards nor sealed audit data were read.
  No model training, calibration, or selection occurred on this cohort.

## Runtime and provenance

- RTX 3080 Laptop GPU, PyTorch 2.6.0+cu124, Transformers 5.6.2, CUDA 12.4.
- Model load: 0.248 s; inference: 12.008 s (83.27 examples/s); peak CUDA
  allocation: 60,880,896 bytes. No training or cloud compute was used.
- Implementation commit: `14a9809e0e8a06c32708def0cd342df241c078c0`.
- Effective config SHA-256:
  `12d1e61985ff4f362fbf4ac43286865685a534941adfa7e7eeac495ba6e09b99`.
- Evaluator SHA-256:
  `64885d0dd7d448913b8dcc02d221cce666b33df736427b996feef22a7061c4f5`.
- Predictions SHA-256:
  `06a0eea3386875717453ecc6535fc546d31026e1ef08c28482651d17c223f559`.
- Metrics SHA-256:
  `7ed4931f9dc5bbf49d14df8f568a5b6ce81db49bb86fbb627da12a7638af386d`.
- Report SHA-256:
  `187937ea10611f5a64b00844480010a0c7f8eb4264681b5a8425b98bdf33d1e9`.
- Detailed implementation, test record, cohort correction audit, and CI are
  preserved in [PR #48](https://github.com/Unjuno/tiny-omni-decision/pull/48)
  and branch
  [`codex/r2-speechcmd-fresh-speaker-eval-20261110`](https://github.com/Unjuno/tiny-omni-decision/tree/codex/r2-speechcmd-fresh-speaker-eval-20261110).
  PR #48's two CI jobs passed at recorded HEAD `34114ade7e6a73711e0896a68dccbbb452d83c0e`.

This narrow audio result does not close M-0/M-1, M-2, or the overall R2 exit.
Image, video, screen, sensor, joint-input necessity, the four-modality locked
task battery, and product quality remain separate open gates.
