# Path A encoder-only Whisper runtime — 2026-11-22

## Outcome

Built a separate, hash-verified assembly that removes the Whisper decoder from
the Path A Audio component. Both Path A Audio and the spoken-palette route now
share one loaded `WhisperEncoder`; the latter's existing task package already
used encoder-only weights. The legacy full-checkpoint assembly was read-only.

The same previously used TRAIN raw-audio/screen request was replayed through
the new assembly's real dispatcher and returned the same option order and
selected option as the previous assembly. Maximum probability difference was
`6.1391e-12`. This is a save/load/runtime-compatibility smoke, not an accuracy
evaluation. The target label was not used, and no validation or sealed-audit
records were opened.

| Measurement | Full Path A assembly | Encoder-only assembly |
|---|---:|---:|
| Package payload bytes | 712,388,681 | 594,168,210 |
| Unique logical runtime parameters | 192,654,853 | 163,102,597 |
| Package target ≤200,000,000 bytes | No | No |
| Whisper Audio artifact | full checkpoint, 37,760,640 params | encoder only, 8,208,384 params |
| Decoder parameters removed | — | 29,552,256 |

The measured package saving is `118,220,471` bytes. The separate audio/text
worker reported 27,314,564 resident unique parameters after loading the
encoder-only route. Its RSS after this request was 892,862,464 bytes. The
previous one-request report had a different RSS value, so this work does not
establish a RAM improvement. One cold dispatcher request took 7.756 seconds
including worker startup and 0.434 seconds in the worker; this is not a latency
benchmark and does not establish a speed improvement.

The remaining payload exceeds the provisional 200 MB target by `394,168,210`
bytes. VideoPrism is still the largest component. INT8 or lower precision for
the other components has not been implemented here.

## Provenance and reproduction

- Base source assembly: `path-a-modular-runtime-assembly-v4`, manifest SHA-256
  `4db84161ec2132528af8d4887e2c0d22fae407edc80682cf802a0667fa3e7de0`.
- New assembly manifest SHA-256:
  `616b9b61e983d875a6a29e71d568f1ba0a6aeba6705da61ee66a88b4be43b3fd`.
- Assembly source commit: `31650bcdb64d8e13a1789e63d7ed51c5588d2c1b`; the builder and runtime source hashes below pin the working-tree implementation used for this attempt.
- New assembly:
  `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4-whisper-encoder-only-20261122-attempt-01`.
- Encoder state SHA-256:
  `d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8`.
- Source Whisper Tiny revision:
  `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`.
- Source checkpoint SHA-256:
  `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`.
- Encoder-only extraction receipt SHA-256:
  `23f648fa7c3554fe63f9f7ab3a75f8ff1e4b04ac11a27a56a54f3d319ec549b6`.
- Builder SHA-256: `795da9fdcddd10371b115131f55b8de5081ef2e034d6e9140f10caa7995951b1`.
- `audio_runtime.py` SHA-256:
  `e10851d235299a0707a3626dd63753a56c5a55c62dc341602377ce88946d1cd4`.
- `path_a_torch_worker.py` SHA-256:
  `139cf2b1994c977bdaf4853094f14ea067b3ebd4785f5c0538c9ace8c4efb5a7`.
- Successful smoke script: `scripts/pretrained_reuse/smoke_path_a_encoder_only.py`.
- Smoke script SHA-256:
  `39a663580673471d29e4993f810515041934d8d4dbc0a9b694593d68b8437ce0`.
- Successful smoke result:
  `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-whisper-encoder-only-runtime-smoke-20261122-attempt-03\result.json`.
- Smoke result SHA-256:
  `86bceba9abafd5ec3615d1b5e6729546006d09226f68cb5e3e7e44b219fc5fe8`.
- Baseline report SHA-256: `7e01bb5815530738602a1296de174df022694c8bf1897365695c6708ee169051`.
- Environment: Windows, Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2,
  CPU; no GPU or cloud compute.
- `verify_modular_bundle` passed for the new assembly, including all payload
  hashes, nested manifests, parameter accounting and byte accounting. The
  original 57-file assembly was verified before copying and left unchanged.
- Smoke attempts 01 and 02 are retained separately; their worker calls
  completed, but the harness initially expected the wrong dispatcher response
  shape. Attempt 03 is the successful replay.

Rebuild the separate assembly and replay the fixed TRAIN request from the
repository root:

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
$env:PYTHONPATH = "src"
python scripts/pretrained_reuse/build_path_a_encoder_only_variant.py `
  --base-assembly "D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4-spoken-palette-20261122-attempt-02" `
  --encoder-state "C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-json-runtime-attempt-04\whisper\encoder.safetensors" `
  --audit-receipt "D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-whisper-encoder-only-audit-20261122-attempt-04\audit-result.json" `
  --output-dir "D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4-whisper-encoder-only-20261122-attempt-01"
python scripts/pretrained_reuse/smoke_path_a_encoder_only.py `
  --assembly "D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4-whisper-encoder-only-20261122-attempt-01" `
  --baseline-report "D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-shared-whisper-audio-image-smoke-20261122-attempt-03\shared-path-a-runtime-smoke.json" `
  --output-dir "D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-whisper-encoder-only-runtime-smoke-20261122-attempt-03"
```

Both tools refuse to overwrite existing output paths.

## Limits and next step

This change establishes a real runtime use of the encoder-only checkpoint and
preserves the fixed paired-audio/image task output on one existing TRAIN
request. It does not change any learned weights or improve task quality. It
does not exercise all four task routes after the storage change, qualify
natural multimodal decision quality, meet the 200 MB package target, or
establish lower RAM or latency. Keep this as an engineering variant; continue
with full Path A route parity and then quantify the remaining VideoPrism / text
storage reductions without promoting this package.
