# DINOv2 GPU smoke and Clevr-4 100k (2026-10-19)

## Scope and safety

This work verified local DINOv2 execution, acquired and integrity-checked the
official Oxford Clevr-4 100k archive, prepared a fresh internal split, extracted
frozen image features, and trained two small readout runs. GPU feasibility
smokes are not model-quality results. Readout outcomes are reported separately
in `R2_CLEVR4_100K_READOUT_RESULTS_20261019.md`.

Both GPU smokes used `scripts/run_with_hotgpu_guard.py` with a 3,200 MiB
reservation, a 14,000 MiB supervisor block threshold, and an 8,000 MiB minimum
child free-memory floor. The cooperating Orbitune supervisor yielded before
each child. No live process was stopped, signaled, or modified. The archive
download and media hashing were CPU/network-only.

## Pinned model and environment

- Checkpoint: local `facebook/dinov2-small`
- Hub revision: `ed25f3a31f01632728cabb09d1542f84ab7b0056`
- `model.safetensors` SHA-256: `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`
- Parameters: 22,056,576
- License: Apache-2.0 (pinned model-card audit in R1)
- Device: local RTX 3080 Laptop, 16,384 MiB reported VRAM
- Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4,
  Transformers 5.6.2, NumPy 2.4.6

### Synthetic-input smoke

Seed-17 synthetic RGB images were processed in one batch of 64. Output was
finite with shape `[64,257,384]`; output SHA-256 was
`3a8d1c4b7804c5eae80641bfe0441ed3960a5d0fab0fea0179522f293b239157`.
Forward time was 0.4490 s (142.54 images/s); peak allocated/reserved CUDA
memory was 490.2/658.0 MiB. This validates the model/tensor path, not image
task performance.

Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\dinov2-small-gpu-smoke-01\result.json`
(SHA-256 `833b92b8127d719ce4b9f68de2846c51c17be992d206441198cb0144bcd0472c`).

### Existing-image input smoke

Eight images from an already-used Clevr-4 train split were verified against
the input label/media manifests and each image's length/SHA-256. This was
inference only; labels were not scored. Output shape was `[8,257,384]`,
SHA-256 `44d2dc475c82c59e068c0bc9654681a73dce7589e305368a6e679b854961f456`.
Forward time was 0.1882 s (42.50 images/s); peak allocated/reserved CUDA
memory was 162.9/196.0 MiB. This rate excludes archive reading and image
decode/processing and is not a throughput projection.

The input train-label manifest SHA-256 was
`00f03f27982e2785636e85a50f4ecf258c0ac3ee483d7242e297cee14b50ec2e`; its
train-image manifest SHA-256 was
`2c56815beb9d2e9951cd38686eafdbe2ecdee05e69e74b5acfad24fbefbb9010`.
Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\dinov2-clevr4-realimage-smoke-01\result.json`
(SHA-256 `ce0d0ad08927743bd03abd52d872713cb30ff167f7e7c869c5dd397b884c6f7c`).

## Clevr-4 100k source and split

The Oxford VGG archive is pinned at
`https://thor.robots.ox.ac.uk/clevr4/clevr_4_100k_v1.zip`. The source code
repository is pinned at
`sgvaze/clevr4@cddc78fb2a8359dc958987b2c750bfdd4bfd2c73` and states CC BY 4.0.
The archive has 36,057,238,315 bytes and matches the official SHA-512:
`7e8454f9f44d7b527eb20252a687b15b4cd28f3e9c79cf6147317971034c367b64bf6bba98d2a66f64b26f1e7eea2105a8acf221198342e89a844fd0007cc94c`.
It was transferred as 68 verified HTTP ranges and assembled. The original
116,371,456-byte single-stream partial and all range parts remain preserved.

Archive inventory is 100,002 entries: one annotation file, scene metadata,
and 100,000 images. Actual annotation split counts are 80,012 `train` and
19,988 `val`; **there is no official test partition**. For non-train records,
the parser read only each top-level `split` string and skipped label values as
opaque bytes. Official validation labels were not decoded and official
validation images were not opened. Since the public validation partition is
publicly available, it is called untouched, not sealed.

The 80,012 official-train records were deterministically split by image asset
identity with seed 17 into 72,010 train and 8,002 internal validation images.
Train/validation ID, asset and media overlaps are all zero. SHA-256 was
computed for every selected image by streaming it from the archive, without
extracting media. The selected media also have zero SHA-256 overlap with the
previous 8,424-image Clevr-4 train manifest. This rules out byte-identical
reuse against that manifest; it does not prove semantic independence from
other possible prior exposures.

External archive/preflight files are under
`C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-videoprism-v0\`.
Train/validation labels and ID index are under
`C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-index-seed17-v0\`.
Hashed media manifests and their overlap report are under
`C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-index-seed17-media-hashed-v1\`.
No image media have been extracted.

## Decision and next gate

The pinned model and real-image input path work locally under cooperative
HotUGPU gating. Archive, asset and selected media-byte integrity gates pass.
The narrow Clevr-4 taxonomy readouts improve on the prior head on the same
internal split, but count remains weak. This synthetic benchmark does not
establish natural-image reasoning, video quality or general multimodal
decision quality.

No quantization, cloud compute, product promotion, or quality claim was made.
Existing Clevr-4 artifacts and checkpoints remain unchanged.
