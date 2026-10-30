# PE-Core T16 save/reload parity attempt — 2026-10-22

Status: **incomplete; no parity result**. This was an isolated CPU-only
runtime check of an already cached checkpoint. It was not a dataset evaluation
or training run, and it did not use the GPU.

## Attempt

- Existing runtime artifact read-only: `C:\CodexArtifacts\pretrained-reuse-r1\pe-core-t16-r1-attempt-02-runtime`.
- New attempt directory: `C:\CodexArtifacts\pretrained-reuse-r1\pe-core-t16-r1-attempt-03-save-reload`.
- Intended operation: construct `timm/PE-Core-T-16-384@7fe539ed578ac49a1c2b4f946e4b0747704c825a` through the isolated OpenCLIP 3.3.0 profile, encode one deterministic synthetic image and two fixed text strings, serialize the in-memory state dict, recreate the model, strict-load the state, and compare embeddings.
- The separate Python process (PID 11660, started 2026-10-22 07:01:36 local time) remained live without producing a result file. Its observed CPU time stayed at 6.06 seconds and working set near 715 MB during follow-up checks. The process was stopped because C: free space had fallen from approximately 1.23 GB before the attempt to 268 MB. The attempt output directory is empty.
- The cause of the disk-space decrease and the point where the runtime stopped are **UNKNOWN**. No claim is made about output equality, CUDA initialization, or model failure. No checkpoint, dataset, previous attempt, or user process was deleted or changed.
- At the stop, the RTX 3080 Laptop reported 15,821/16,384 MiB used; compute PID 30196 was still present. It was left untouched. The extra recursive temporary-file inventory process started for diagnosis was also stopped.

## Interpretation and next gate

This is an execution-blocked serialization/parity check, not a PE-Core quality
failure. Do not retry until the local disk constraint is resolved and the
runtime is instrumented with progress checkpoints before loading, forward,
serialization, and reload. Keep the original checkpoint and attempt-02
runtime immutable. The earlier report on remote branch
`codex/r1-pe-core-audit-20261021` records a CPU image/text smoke, but does not
provide a successful serialized-state reload parity result.
