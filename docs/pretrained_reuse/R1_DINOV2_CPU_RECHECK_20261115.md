# R1 DINOv2-small local checkpoint runtime recheck (2026-11-15)

## Scope

A CPU-only, offline recheck of the already pinned `facebook/dinov2-small` artifact used by the MiniWoB frozen-feature Path A experiment. This verifies that the real local checkpoint and processor still load and produce finite image features; it is not a new quality result and reads no task or evaluation images.

## Pinned artifact and rights evidence

- Model revision: `facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056`.
- Local model card says `apache-2.0`; the checkpoint is used under that published weight license. This is checkpoint-specific evidence and does not grant rights to any unrelated source dataset.
- Local model card SHA-256: `4c20dca454a8e5c670e8de5c7e6040f512aeca5438516f7623eedc4e3b00599c`.
- Config SHA-256: `1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d`.
- Processor config SHA-256: `14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828`.
- Safetensors SHA-256: `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`; 88,249,960 bytes.

## Runtime result

Loaded `AutoImageProcessor` and `AutoModel` from the local directory with `local_files_only=True` and `trust_remote_code=False`. A constant 518x518 RGB image fixture was processed on CPU; the processor emitted `[1,3,224,224]`, and the model returned finite `[1,257,384]` last-hidden-state features. Parameter count was 22,056,576; all parameters were frozen. Load time was 0.169 s and forward time 0.118 s in this single CPU sample. Environment: Python 3.11, PyTorch 2.6.0+cu124, Transformers 5.6.2. `CUDA_VISIBLE_DEVICES` was empty; no CUDA/GPU work or cloud compute occurred.

## Interpretation

The local image checkpoint and processor remain executable under the pinned revision and published checkpoint license. This does not measure downstream Decision quality, video behavior, cross-modal fusion, or target-device latency. The DINOv2 image path remains a task-scoped Path A candidate; the separate VideoPrism video path and R2 task-admission gates remain in force.
