# Compact visual checkpoint feasibility audit

Pinned facebook/dinov2-small revision
ed25f3a31f01632728cabb09d1542f84ab7b0056 was downloaded and verified against
Hub weight SHA-256 ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1.
The [publisher card](https://huggingface.co/facebook/dinov2-small/blob/ed25f3a31f01632728cabb09d1542f84ab7b0056/README.md)
declares Apache-2.0. This audit concerns DINOv2-small, not later differently
licensed Cell-DINO or XRay checkpoints.

Exact loaded count: **22,056,576 parameters**, FP32 weight file 88,249,960 bytes.
CPU only, two threads, Python 3.11.9 / torch 2.6.0+cu124 / Transformers 5.6.2.
AutoModel and AutoImageProcessor loaded local files with trust_remote_code=False.
No missing/unexpected/mismatched keys. Two generated RGB 256x256 solid-color
images (128,64,32) and (32,64,128), processed by the unchanged pinned processor,
produced [2,257,384] last_hidden_state from [2,3,224,224] pixel_values. Repeated
inference in eval/inference_mode was exactly equal and finite. Timed load plus
processing and two forwards: 0.465 seconds; not an end-to-end deployment benchmark.
First attempt passed forwards but failed report serialization on a set in loader
metadata; unchanged CPU execution was repeated after fixing serialization.

This establishes loading and parameter feasibility only. It is an image encoder,
not a native temporal encoder. No classification, video reasoning, alignment,
fusion, quantization or quality threshold was evaluated. Keep spatial tokens for
future evidence-based comparisons; pooling them now would lose information.
No existing visual tower has been replaced. The current language candidates
also fail the conditional reasoning diagnostic, so a small parameter sum alone
is insufficient to select any combined architecture.

Exact artifact and file hashes are in the adjacent JSON. Download and CPU audit
ran concurrently with the fixed GPU training run; its code, inputs and settings
were unchanged. No cloud cost, sealed audit access or legacy artifact writes.
