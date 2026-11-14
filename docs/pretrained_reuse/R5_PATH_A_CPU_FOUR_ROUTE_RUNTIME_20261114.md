# Path A four-route CPU runtime qualification (2026-11-14)

## Purpose and scope

This is an operational qualification of the existing canonical Path A JSONL
runtime. It checks that one TRAIN-only request per text, image, audio, and video
route can be processed on CPU and serialized as typed decision JSON. It is not a
quality evaluation: labels, validation data, sealed audit data, and training
were not used.

## Inputs and immutable assembly

- Source worktree: `codex/path-a-audio-cpu-parity-20261114` at
  `fe74c420b9765c1389d4923484ceda50bfca8db9`.
- Read-only assembly manifest SHA-256:
  `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
- Request JSONL SHA-256:
  `cff3b662142c876e78d0ebe65f183a1ae4aa72aa6d897b7a23c6aec37e0f10ef`.
- Inputs were drawn only from the frozen TRAIN partitions: one text record,
  one CLEVR-4 image, one Speech Commands audio file, and one CLEVRER video.
- No labels or target answers were included in the requests.

## Execution

Two fresh Python processes ran the same request file with `--device cpu`,
below-normal priority, and two CPU threads. CUDA was hidden and JAX was forced
to CPU. No repository runtime code, model weights, datasets, training process,
or GPU allocation was changed by this check.

| Run | Exit | Wall time | Sampled process-tree peak RSS | Output SHA-256 |
|---|---:|---:|---:|---|
| 1 | 0 | 36.53 s | 4,618,838,016 bytes | `2277fc9e22f51034eccf9ad9e826d0b7c202d6063ec564b831edbb03ca254542` |
| 2 | 0 | 29.20 s | 4,372,709,376 bytes | `2277fc9e22f51034eccf9ad9e826d0b7c202d6063ec564b831edbb03ca254542` |

Peak RSS is sampled at 250 ms intervals and may miss short-lived peaks. It is
the process tree's resident memory, not a model-only measurement. Both runs
produced byte-identical output; stderr was empty.

## Validation result

The output validator passed. It found seven typed decisions across the four
requests (the audio route emits four decisions), preserved request/option IDs
and text, and verified finite normalized probabilities. The repeat outputs
were byte-identical.

**Result: PASS for CPU route execution and deterministic serialization on
these four TRAIN examples.** This does not demonstrate task accuracy,
calibration, multimodal fusion, robustness, production latency, or deployment
readiness. In particular, separate modality routes do not establish that a
joint decision benefits from more than one input.

The output directory is
`C:\\CodexArtifacts\\pretrained-reuse-r2\\path-a-cpu-four-route-runtime-20261114-attempt-01`.
It contains requests, preflight, both process outputs and runtime metadata,
telemetry, the wrapper and validator, and `validation-summary.json`. These
artifacts are outside Git; no model weights or media were added to this
repository.
