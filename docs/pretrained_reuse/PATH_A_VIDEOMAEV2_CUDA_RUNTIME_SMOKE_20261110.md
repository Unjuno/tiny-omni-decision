# Path A VideoMAEv2 CUDA runtime smoke — 2026-11-10

## Result

The pinned Apache-2.0 VideoMAEv2 Small checkpoint loaded on the local RTX 3080
Laptop GPU and ran finite, bitwise-repeatable inference on deterministic
synthetic inputs. This is R1 hardware/runtime evidence only; it is not a
Decision task result and does not establish quality, calibration, or Path A
integration.

| Route | Input | Median forward (3 repeats) | Peak allocation |
|---|---|---:|---:|
| Image-shaped | One synthetic RGB image repeated to 16 frames | 65.44 ms | 229.5 MiB |
| Video-shaped | 16 deterministic synthetic RGB frames | 69.31 ms | 229.5 MiB |

Each forward reported logits `[1,710]` and features `[1,384]`. Outputs were
finite and bitwise identical across repeated calls within the process. The
measurement ran in FP32. The 22,153,286-parameter model used the existing
44,334,609-byte checkpoint; no conversion was performed.

## Reproduction and environment

- Source commit: `573807248ac0cf7483a80b75ffb9b9c401ad90b9`.
- Command:
  `python -m scripts.audit_path_a_videomaev2_cuda_runtime --output C:\\CodexArtifacts\\pretrained-reuse-r2\\path-a-videomaev2-cuda-smoke-20261110-attempt-01 --repeats 3`
- Device: NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB.
- PyTorch `2.6.0+cu124`; CUDA runtime `12.4`; checkpoint/source hashes match
  the already audited R1 files.
- Peak allocated: `240,694,272` bytes; peak reserved: `253,755,392` bytes.
- Script synchronizes around each timed forward, includes one untimed warmup
  per route, and refuses to overwrite an existing output directory. It fails
  closed if CUDA is unavailable; it has no CPU fallback.
- Artifact:
  `C:\\CodexArtifacts\\pretrained-reuse-r2\\path-a-videomaev2-cuda-smoke-20261110-attempt-01\\result.json`
  SHA-256 `c09ecf3396e3f32568f488fa2a8dda551a53c32cd822434dfdf01cf43fe6077e`.

No dataset, training, validation, checkpoint selection, or sealed audit was
used. Synthetic video was supplied as an in-memory tensor, so this does not
verify file decoding, caller-option mapping, calibration, or end-to-end JSONL
routing. The existing VideoPrism worker remains blocked by host Code Integrity;
no policy bypass or environment modification was attempted.

## Verification

- Ruff: pass for the new script and tests.
- CPU tests: 2 passed. They cover overwrite refusal and minimum repeated-run
  enforcement; they do not instantiate a model or require CUDA.

Next gate: integrate this as a separate task-scoped Path A video backend only
after a clean, admitted Decision task and label-to-option contract are fixed.
Do not reuse previously observed CLEVRER, Perception Test, or other exposed
validation results as new quality evidence.
