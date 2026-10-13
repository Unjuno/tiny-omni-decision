# SlotFormer object-centric checkpoint audit

Motivation: completed point-trajectory and pooled joint-feature experiments did
not establish condition-sensitive object identity. Inspect an existing pretrained
object-slot representation, rather than repeat the rejected readout schedules.
This is candidate admission, not a switch of product architecture.

## Immutable source and actual archive

Official [repository](https://github.com/pairlab/SlotFormer), pinned revision
`10a6d9b7bc05ef2397ce2586da3fc0212f8c47aa`. Source snapshot includes LICENSE,
benchmark/CLEVRER/install instructions and the StoSAVi configuration, each hashed.
The [official benchmark instructions](https://github.com/pairlab/SlotFormer/blob/10a6d9b7bc05ef2397ce2586da3fc0212f8c47aa/docs/benchmark.md)
link the public pretrained archive. Downloaded369,579,924bytes; SHA256
`1c57bcc33851fa89bc4fbfae9dcf7e5c787dec8fb7d681c8d7a37c9745ba3824`.
Drive file ID:`1v7jbd471qA7efOVJOCMZhO6h2s-BOtCd`.

Archive index saved. Only three named model members were copied for tensor
inspection; no mass extraction, cached slots, dataset JSON or evaluation answers
were loaded. All three load using torch.load(weights_only=True), with no unsafe
globals found and no allowlist additions.

| Checkpoint | File bytes | State tensors | Tensor elements |
|---|---:|---:|---:|
| stosavi_clevrer_params/model_12.pth |21,958,309|62|1,215,620|
| slotformer_clevrer_params/model_80.pth |40,816,173|66|3,739,908|
| aloe_clevrer_params-rollout/model_400.pth |34,714,045|164|2,879,437|

These are state-dict element counts, not loaded parameter counts: buffers and
reconstruction decoders are included. Files also contain optimizer/training state.
Do not sum these numbers as a deployment model size; identify shared/unused parts
after exact architecture loading. No inference has run yet.

Checkpoint SHA256 values:
- StoSAVi:647ed1c177b66cb5507949637c4d61cc0be2f08a015ad82d5f2b456490e89a71
- SlotFormer:b5d9d20fdd9dfbc958ab819a0527d66a182e8506747ef96dee7d1d32aec2fef0
- ALOE:75f142a572138609a56cdb119b75d113e43c470e1a2322d53b13d83da0c67af9

## Scope and limitations

The pinned StoSAVi configuration uses64x64 input, seven128-dimensional slots,
two slot-attention iterations and a CNN encoder. Its training clips have six
frames. This is dataset-specific object decomposition, not established natural
video understanding or Omni fusion. No claim that this solves the real occlusion
task or achieves95% on any project task.

Repository LICENSE is MIT. The downloaded archive contains no separate weight
license file; product weight-licensing admission remains REVIEW pending scope
confirmation. Dataset provenance and overlap must be checked before any quality
claim using a pretrained CLEVRER model. Existing observed benchmarks cannot
become blind audits by changing the model.

Next: inspect dependencies and preprocessing, strictly load the actual encoder,
count named parameters, and attempt a small frozen runtime probe. Do not launch
new world-model pretraining or use the provided dataset-wide slot/answer files.

Artifacts:
`C:/CodexArtifacts/pretrained-reuse-r1/slotformer-source-audit-01/`
and `C:/CodexArtifacts/pretrained-reuse-r1/slotformer-weight-audit-01/`.
Includes immutable source hashes, download hash, archive index and tensor audit.
No legacy artifacts changed, GPU training or cloud compute used.
