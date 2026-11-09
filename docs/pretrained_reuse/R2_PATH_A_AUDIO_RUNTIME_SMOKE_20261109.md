# R2 Path A audio runtime smoke — 2026-11-09

## Result

The preserved Speech Commands Path A runtime loaded its pinned Whisper Tiny
encoder, MiniLM text encoder, and saved readout from local files, then returned
finite normalized option probabilities on one generated 1 kHz / 16 kHz PCM WAV
input. The CPU output contained two probabilities summing to 1.00000003 and the
runtime reported 55,494,785 parameters. This verifies a narrow local inference
path only. The generated tone has no semantic label, so the predicted option is
not a quality result. No corpus, validation IDs, sealed audit, training, or GPU
were used.

## Reproducibility and integrity

- Repository: `Unjuno/tiny-omni-decision`; roadmap base commit:
  `e07e049ff4387484fb754e0fff963631f3cc7c85`.
- Runtime package: `path-a-modular-runtime-assembly-v4`, source commit
  `b32d2b0c3d3fccfeb7772ca61ec153ce99aa1a6a`.
- Package manifest SHA-256:
  `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
- A read-only check matched all 42 listed package files by size and SHA-256.
- Audio runtime: `components/audio/runtime/tiny_omni_decision/audio_runtime.py`;
  its constructor independently checked the pinned Whisper, MiniLM, and readout
  hashes before loading.
- Input WAV SHA-256:
  `ef8544b046a025554e80d9d1f3a6fbf9685070eaed0dc6b47afee5604a418c35`.
- Result JSON SHA-256:
  `b4c06ac7d0697eb2ca6f0a9cc1b5f7f4c8f10120d49b54760a811234dadcac0c`.
- Output: `C:\CodexArtifacts\pretrained-reuse-r2\path-a-audio-runtime-smoke-20261109-attempt-01\`.
- Elapsed time reported by the smoke process: 0.614 s on CPU; no GPU or paid
  compute cost.

The process used Python 3.11.9 and PyTorch 2.6.0+cu124. The installed
Transformers version was 5.19.0, whereas the assembly manifest records 5.6.2 at
build time. The ordinary venv startup also fails while decoding an editable
`.pth` path containing Japanese characters, so this smoke used `python -S` and
explicitly added the existing site-packages directory. The task runtime imported
and ran without importing PEFT. A separate PEFT import in that environment was
blocked when the optional torchaudio native DLL was loaded by Windows
application control. No packages, environment files, checkpoints, or runtime
source were changed. This is not a compatibility certification for the
manifested build environment.

## Interpretation and next gate

This confirms that one existing Path A audio component can produce the typed
option-probability object from a raw local WAV when its local checkpoint files
are intact. It does not validate the Speech Commands task accuracy, show
cross-modal complementarity, or provide an integrated four-modality runtime.
The package still contains separate task-specific profiles; its previously
measured content size is 687,254,772 bytes for 170,499,716 unique runtime
parameters. The 182,018,983-byte INT8 figure in the assembly audit is a storage
estimate, not a quantized package or runtime measurement.

The next quality step remains gated on a rights-cleared task with independent
groups and controls establishing that each claimed input matters. Do not tune or
re-evaluate exposed validation sets to extend this smoke result.
