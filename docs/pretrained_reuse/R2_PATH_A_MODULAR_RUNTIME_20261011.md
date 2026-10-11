# Path A modular runtime assembly — 2026-10-11

## Scope

This is a local packaging and relocation check for the existing frozen Path A
Audio and CLEVR-4 Image runtimes. It does not train, select, recalibrate, or
change either model. The output supports the two component tasks only. It is
not a four-modality runtime, a temporal video decision runtime, or joint
audio-video inference.

## Assembly

- Builder source commit: `277c128` (`feat: assemble verified modular Path A runtime bundles`).
- Builder: `scripts/assemble_path_a_modular_runtime.py`.
- Assembler and verifier: `src/tiny_omni_decision/modular_bundle.py`.
- Assembly: `C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v1`.
- Assembly manifest SHA-256: `95fd0ce47be48a10c50a237ce0e79f806ee3464d507b493f3ff8554d3505da4d`.
- Source Audio manifest SHA-256: `574640d88d6ddbbdb2d12f2466a248ce13c8d9eb3ac542d099b2f08981b690f4`.
- Source Image manifest SHA-256: `958e1f8735c33237342d0533f7529a0514fc4f002ca812057b2eff28ac51012e`.
- Source components were verified before copying and again after copying;
  the assembled package passed a complete file-set, size, hash, and accounting
  verification. The source directories were not modified.
- Measured model parameters: `170,007,682` (Audio component `55,494,785`,
  Image component `114,512,897`).
- Assembly files excluding the root package manifest: `685,255,687` bytes.
  The full directory including its package manifest is `685,265,908` bytes.
  No component file hashes overlap, so the component files cannot be reduced
  by deduplication. The provisional 200,000,000-byte target is **not met**.
- The Audio component includes Whisper Tiny, MiniLM, and its fixed-task head.
  The Image component includes VideoPrism-B and its CLEVR-4 head plus fixed
  option embeddings; it does not load MiniLM at inference. MiniLM is therefore
  stored once in the assembly.

## Runtime relocation checks

The copied component manifests match the source manifests byte-for-byte. The
following end-to-end replays used the existing development validation records
as packaging checks, not as new model selection or quality evidence. Both used
the components from assembly `v0`; assembly `v1` contains those same component
manifest hashes and passed full file verification.

| Component | Validation slice | Backend / time | Resource | Parity |
| --- | ---: | --- | --- | --- |
| Audio | 8 audio observations, 32 decisions | RTX 3080 Laptop; load 0.544 s, inference 0.730 s total (0.0913 s/audio) | Peak CUDA allocation 380,225,024 bytes | 0 prediction mismatches; max logit delta `3.34e-6`, max probability delta `1.19e-7` |
| Image | 8 images, 32 decisions | JAX CPU; load 3.424 s, inference 13.903 s total (1.738 s/image) | Peak working set 4,256,534,528 bytes | 0 prediction mismatches; max probability delta `2.33e-6` |

Audio replay report SHA-256:
`53f496e1dd6012ab96b26bf93eb9db7565e91cebd54c505b58ec2265346ae13d`.
Image replay report SHA-256:
`f1583f55764541683c1527dbee96f2d2120a422d2be5de56070babd163dbaa7d`.

## Runtime and quality limits

- Audio used Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, and
  CUDA 12.4 on the local NVIDIA GeForce RTX 3080 Laptop GPU (16 GiB).
- Image used the existing separate Python 3.11.16 environment with JAX 0.6.2,
  TensorFlow 2.18.1 and NumPy 1.26.4 on CPU. That environment does not contain
  PyTorch; the Audio environment does not contain JAX or TensorFlow. The
  assembly records these separate profiles and does not vendor environments.
- The two components are independently callable bundles. A single process
  dispatcher and standalone text and temporal-video runtimes have not been
  integrated. The MiniLM tower is only part of the Audio task's fixed query
  and candidate encoding here; this is not standalone text reasoning.
- Audio remains a closed-set Speech Commands task with four fixed query
  templates. Image remains a synthetic CLEVR-4 taxonomy task. The 8-record
  slices above establish relocation parity only; they do not update quality
  estimates or satisfy an independent quality gate.
- Joint audio-video data/task support, text task packaging, temporal video
  inference, a common runtime dependency profile, and package-size reduction
  remain open. No training, quantization, sealed-audit loading, cloud use, or
  model-weight update occurred.

## Verification

- Unit tests: `tests/test_modular_bundle.py` — 3 passed.
- Full suite before the final documentation update: 213 passed, 1 skipped.
- Ruff: passed.
- `verify_modular_bundle` passed for assembly `v1`, including all component
  manifests and files.
- Assembly contents live under `C:\CodexArtifacts`; no model weights or media
  were added to Git.
