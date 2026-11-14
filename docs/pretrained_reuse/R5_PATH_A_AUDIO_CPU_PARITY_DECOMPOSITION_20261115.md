# Path A Audio CPU/FP32 parity decomposition — 2026-11-15

## Decision

The earlier CPU-versus-CUDA mismatch was mostly a precision-path comparison: the saved reference uses CUDA BF16, while the CPU runtime uses FP32. A saved same-request CUDA FP32 diagnostic shows much smaller CPU-versus-CUDA differences. CPU execution is repeatable on this request, but strict cross-device parity at the existing 1e-5 threshold is still **not established**: one ten-option decision differs by about 2.11e-5 between CPU FP32 and CUDA FP32. Keep the production path and threshold unchanged.

This is a fixed-development-request runtime diagnostic only. It is not a quality evaluation, checkpoint selection, calibration, training, or a sealed-audit access.

## Fixed inputs and provenance

- Runtime source: codex/path-a-canonical-json-runtime-20261114, commit 6d722c258f188a0a09c624bf7b39e3f352d55319.
- Request JSONL SHA-256: ae200aa3c099a2d8ead6d1dd1890122cf8b093ac1989fe0876f69735b267123e.
- Saved CUDA output JSONL SHA-256: 0cd3437ecd32310a724f2d982c641c7497ff88bab3205c01f07c79e7aebf675a.
- Audio SHA-256: 77ad54371520bd5381b7758694bc6d98193e8f8864750ea56ea7f8085f2daaa1.
- Audio readout SHA-256: 4efa7939a686ad97b38f9eec0b8171e6e62d125ba4eb373e5d83e33ac4df42c7.
- The saved attempt-12 CUDA/CPU decomposition result SHA-256 is d32f02a5aea997f9b48a397e8276c40360db0462aa8cdea25d95d8872e96da4d.
- Four fixed queries were replayed: one 10-option query and three 2-option queries. Their labels were not opened.

## Results

| Comparison | Maximum absolute probability difference | Argmax changes |
|---|---:|---:|
| Same-process CPU FP32 repeat 1 vs repeat 2 | 0 (probabilities and logits were byte-identical) | 0/4 |
| New CPU FP32 vs prior attempt-12 CPU FP32 | 1.19e-7 | 0/4 |
| CPU FP32 vs saved CUDA BF16 reference | 6.18e-4 | 0/4 |
| CUDA FP32 vs saved CUDA BF16 reference | 5.97e-4 | 0/4 |
| CPU FP32 vs CUDA FP32 | 2.11e-5 | 0/4 |

The same-process CPU run used the same 4-thread CPU configuration on both repeats. Both runs produced audio pooled-feature SHA-256 630229ba230b770867490726e5f7a316c2ad578128abfbb3e4a0eddbc10e68bb, matching the earlier attempt-09 CPU result. For the 10-option query, the CUDA FP32-to-BF16 delta was 5.97e-4, while CPU FP32-to-CUDA FP32 was 2.11e-5. This identifies dtype/backend path as the dominant part of the larger reported difference on this request, while leaving the smaller strict-parity miss unresolved.

This does not prove numerical parity on other examples or devices. Equal argmax on four already observed queries is not a task-quality result.

## Execution and retained artifacts

- Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, Windows 10 build 26300.
- CPU-only process used CUDA_VISIBLE_DEVICES=-1; torch.cuda.is_available() was false and torch.cuda.is_initialized() remained false. The RTX 3080 was not used.
- The two measured inference calls took 0.3124 s and 0.1510 s, excluding model load.
- Successful attempt directory: C:\CodexArtifacts\pretrained-reuse-r2\path-a-audio-cpu-parity-20261115-attempt-16.
- Result JSON SHA-256: addcf1460ff3f193ddcc1a9e5112811d5145c5eb1cc2807fa80dc47d97f5ac67.
- Audit script SHA-256: 5d919cda53a7aabb0339d801704786d32a7fbb4ca8d92d1d9aa00fcb50aae0e4.
- Attempts 13–15 are retained separately. Attempt 13 stopped before inference because an empty CUDA_VISIBLE_DEVICES did not hide CUDA on this Windows/PyTorch setup. Attempts 14–15 stopped before model loading because of JSONL array-shape and request-hash-field assumptions. Attempt 16 used -1, parsed the existing JSONL shape, and verified the fixed media hash from the pinned parity receipt.

No existing request, output, checkpoint, runtime, tolerance, or user worktree was modified. No new task-quality claim is made.

## Next diagnostic

If strict 1e-5 cross-device parity is required, retain and compare the preprocessed input, pooled audio feature, question/candidate text vectors, option logits, and probabilities in a single frozen request protocol on both backends. The old CUDA run did not persist all intermediate tensors, so this report cannot localize the remaining 2.11e-5 difference to Whisper, MiniLM, or the readout. Do not widen tolerance from this sample.

