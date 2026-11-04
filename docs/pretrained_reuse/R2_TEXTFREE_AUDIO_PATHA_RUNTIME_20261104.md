# Path A text-free audio runtime integration — 2026-11-04

## Scope and result

This change connects the existing frozen Whisper Tiny encoder and text-free Speech Commands readout to the Path A typed Decision dispatcher as an optional `audio_native` route. A persistent JSONL worker accepts raw PCM16 WAV files and typed class options, then returns option probabilities in the normal Decision JSON shape. It is an engineering integration smoke, not a quality evaluation or a general audio capability claim.

The existing audio route is unchanged. `audio_native` is opt-in and requires both external artifact paths (encoder and readout); the v4 package and all pre-existing artifacts remain untouched. The route is limited to the readout's fixed ten English Speech Commands classes. It does not support arbitrary audio questions or arbitrary candidate text.

## Pinned inputs and integrity

| Input | Revision / identity | SHA-256 |
|---|---|---|
| Whisper Tiny encoder | `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af` | `d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8` |
| Encoder config | pinned export | `d8c8d403350cf4e21b4351d3812fc61defc6c5784d2d8f8a622d4b8ca23b98e9` |
| Audio preprocessor config | pinned export | `3e4dc53c73e7e3954fc4435034f5027733c61d7c123b38bbd1461598fd8503ef` |
| Frozen text-free readout | `best-readout.pt` | `46808a882ab75f8f19a138e4a91d13ceb42cfdc82534c4608c42ee615774d3cb` |
| TRAIN manifest | Speech Commands frozen sample | `02a43981e153b0f63bd55192f614fe504114f05f212ca0f4da37e2d91480a9da` |

The runtime verifies encoder/readout hashes, rejects unsupported WAV format and rechecks file content after encoding. Audio features are content-hash keyed in a bounded in-memory cache. The route loads the 8,208,384-parameter encoder plus 3,850-parameter readout (8,212,234 total); it does not load the Whisper decoder or a text encoder. No quantization was performed.

## CPU TRAIN-only smoke

- Source code commit: `fa155496ffb0bd0c2f1194816663b3866c7ac204`.
- Environment: Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2.
- Device: CPU; CUDA was not initialized and no GPU was used.
- Data access: `train.jsonl` only. Four raw assets from four distinct speaker groups; no validation manifest, labels for inference/metrics, or sealed audit was opened.
- Requests: five, including a repeated first asset with reversed option order.
- Encoder calls: four expected and observed; the repeated request was a cache hit.
- Class-bound option-permutation probability maximum absolute delta: `2.9802322387695312e-08` (acceptance tolerance `1e-6`).
- Wall time: 18.28 seconds for the smoke, including worker startup and bundle integrity verification. BF16 CPU autocast is slower than the earlier FP32 attempt.
- Result: `PASS_RAW_AUDIO_PATH_A_TRAIN_ONLY_SMOKE`.
- Full run record: `C:\CodexArtifacts\pretrained-reuse-r2\textfree-audio-patha-runtime-smoke-20261104-attempt-06\run-report.json`.
- Run record SHA-256: `2181c96d5fe25b9d242795f15241f19f741076cd6e68abcefae3b0db2e44360e`.

The smoke intentionally did not read targets to calculate accuracy, NLL, Brier, or ECE. It therefore makes no quality statement. Earlier separate development evidence for this narrow readout includes poor calibration; this integration does not resolve that issue. The output probabilities are not evidence of 95–98% accuracy or general multimodal performance.

## Training-feature precision follow-up

Inspection of `scripts/train_frozen_audio_probe.py` showed that the existing frozen training feature cache was created with CUDA BF16 autocast. The first new CPU runtime version used FP32 and was not numerically aligned with that training path. The runtime now applies BF16 autocast around the encoder forward pass on both CPU and CUDA; waveform/Whisper frontend preparation remains outside autocast. A focused unit test asserts this precision mode.

A read-only comparison against the four same TRAIN rows in the existing feature cache (`ace323458e7f443daf63382d2d41d99151d2a5801c671d30b0df68d52338ae64`) found that CPU BF16 autocast did not achieve the `1e-5` feature parity threshold against features produced on CUDA: per-row maximum absolute feature difference was 0.00998–0.02127, with mean absolute difference 0.00231–0.00325. For these four rows the readout's top class was unchanged; maximum absolute probability difference was 0.000967. These are only four TRAIN examples and are not quality evidence. CPU and CUDA BF16 kernels can differ numerically, so exact CUDA runtime parity remains unverified.

At this audit, the RTX 3080 Laptop GPU had 15,832/16,384 MiB allocated and 344 MiB free while another user's GPU workload was active. No GPU test or process intervention was attempted. Revalidate feature parity on CUDA after that GPU is available before claiming runtime parity to the training cache.

## Code and verification

Implementation adds `textfree_audio_runtime.py`, an isolated audio worker, and a TRAIN-only integration smoke script. The Path A dispatcher recognizes the opt-in `audio_native` task and returns the standard typed JSON response. The readout class-index tensor is placed on the logits' device for correct CPU/CUDA use.

Focused tests cover pinned hash checks, WAV contract rejection, feature-cache reuse, option-order invariance and typed dispatch without starting unrelated workers. Ruff and 30 focused CPU tests pass; `git diff --check` passes. No model fitting occurred.

The first repeat attempt used a stale auxiliary VideoPrism path and stopped before loading a model or reading audio; the corrected attempts used the existing pinned VideoPrism runtime paths. The original smoke record (attempt 05) remains preserved and corresponds to the pre-BF16 code commit; attempt 06 is the current runtime implementation. No existing artifact was changed by these attempts.

## Limits and next gate

This proves that one existing frozen audio classifier can be reached through the Path A runtime and emits option-bound typed probabilities with reusable feature caching. It does not establish audio accuracy, calibration, robustness, modality fusion, task generality, or product readiness. The route is not assembled into a new package version, and external encoder/readout paths remain explicit runtime dependencies. R2 remains open; this report does not qualify as a fresh source/task admission or close the multimodal necessity gate.
