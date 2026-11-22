# Path A Whisper encoder-only storage audit — 2026-11-22

## Result

The Path A Audio and spoken-palette inference paths call Whisper's encoder and
do not use its decoder. A pinned CPU-only export extracted the encoder tensors
from the existing Whisper Tiny checkpoint, strict-loaded them into a standalone
`WhisperEncoder`, and compared both weights and a deterministic encoder forward
against the original full model. Tensor and output maximum absolute differences
were both exactly zero.

| Measurement | Result |
|---|---:|
| Source full checkpoint | 151,061,672 bytes; 37,760,640 parameters |
| Encoder-only state | 32,840,016 bytes; 8,208,384 parameters |
| Decoder parameters not used by encoder inference | 29,552,256 |
| Weight-file reduction | 118,221,656 bytes (78.26%) |
| Seeded CPU forward max / mean absolute difference | 0 / 0 |
| Forward input | `[1, 80, 3000]` seeded synthetic log-mel tensor |
| Export elapsed time | 2.285 s |

If the full checkpoint were replaced by this encoder-only state in the current
four-route assembly, the payload would be approximately 594,167,025 bytes,
still well above the provisional 200,000,000-byte target. This is an arithmetic
projection from the measured file reduction, not a rebuilt-package measurement.
The current unique parameter count of 192,654,853 includes the full Whisper
checkpoint; an encoder-only runtime would reduce that count by 29,552,256 to
163,102,597, assuming the same encoder is shared and no decoder task is added.
That revised count is also projected until the runtime and manifest are changed.

## Reproduction and provenance

- Source commit: `fb161c846ddc9dd70ec56d8feaccfc88b2141cd8`
- Model: `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`
- Source checkpoint SHA-256: `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`
- Runner: `scripts/audit_whisper_encoder_only_export.py`
- Runner SHA-256: `78dba0515bff28ced0eeea4dac19f75bfa433c6f379079dc2193248a7a85484d`
- Machine-readable receipt: `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-whisper-encoder-only-audit-20261122-attempt-04\audit-result.json`
- Receipt SHA-256: `23f648fa7c3554fe63f9f7ab3a75f8ff1e4b04ac11a27a56a54f3d319ec549b6`
- Extracted state SHA-256: `e57b5c0d63bb0a411db81980982ece7907cc61cbab1c344d6510ee9ac93db380`
- Extracted state bytes: `32,840,016`
- Environment: Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, safetensors 0.8.0; CPU only.
- No dataset media, labels, train/evaluation split, sealed audit, GPU, or cloud resource was used.
- The source package and all source checkpoint files remained unchanged. All exports went to a new attempt directory.

Reproduction command from the repository root:

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
$env:PYTHONPATH = "src"
python scripts/audit_whisper_encoder_only_export.py `
  --source-model-dir "D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4-spoken-palette-20261122-attempt-02\components\audio\models\audio-whisper-tiny" `
  --output-dir "D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-whisper-encoder-only-audit-20261122-attempt-04"
```

The runner refuses to overwrite an existing output directory.

## Limits and next gate

This proves exact extraction and encoder execution parity on a synthetic
feature tensor. It does not yet show that the production Path A runtime can
load an encoder-only package, preserve raw-audio preprocessing and task outputs,
or pass Save/Reload and all existing route parity checks. It also does not
reduce the package below 200 MB. Keep the original assembly intact; the next
implementation step is a separate compact assembly variant that references
the encoder-only state, then replay the already frozen development checks
without opening sealed audit data. Quantization of the remaining VideoPrism,
MiniLM, Whisper encoder and readouts still requires a measured full-package
quality/calibration and runtime study.
