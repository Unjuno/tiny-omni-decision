# DINOv2 GPU smoke and Clevr-4 100k acquisition (2026-10-19)

## Scope and safety

This work checked whether the pinned DINOv2-small checkpoint can execute on
the local RTX 3080 Laptop, and resumed acquisition of the official Oxford
Clevr-4 100k archive as a possible higher-coverage image task. These are
feasibility checks, not model-quality results. No readout was fit, no
checkpoint was selected, and no validation or audit metric was computed.

Both GPU smokes used `scripts/run_with_hotgpu_guard.py` with a 3,200 MiB
reservation, a 14,000 MiB supervisor block threshold, and an 8,000 MiB minimum
child free-memory floor. The cooperating Orbitune supervisor yielded before
each child. The child reported a `hotgpu_reserved` gate with 13,015 MiB free;
after each wrapper exited, Orbitune was allowed to resume. No live process was
stopped, signaled, or modified. The later archive transfer is CPU/network-only.

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

Seed-17 synthetic RGB images were processed in one batch of 64. The output was
finite with shape `[64,257,384]`; output SHA-256 was
`3a8d1c4b7804c5eae80641bfe0441ed3960a5d0fab0fea0179522f293b239157`.
Forward time was 0.4490 s (142.54 images/s); peak allocated/reserved CUDA
memory was 490.2/658.0 MiB. This only validates model loading and the tensor
path, not image-task performance.

External receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\dinov2-small-gpu-smoke-01\result.json`
(SHA-256 `833b92b8127d719ce4b9f68de2846c51c17be992d206441198cb0144bcd0472c`).

### Existing-train-image input smoke

Eight images were read from the already-used Clevr-4 train split. Before
decoding, the train label manifest and media manifest were checked against
their recorded SHA-256 values; each of the eight image members was checked
against its manifest byte length and SHA-256. The run performed inference
only and did not inspect labels for scoring.

The output was finite with shape `[8,257,384]`; output SHA-256 was
`44d2dc475c82c59e068c0bc9654681a73dce7589e305368a6e679b854961f456`.
Forward time was 0.1882 s (42.50 images/s); peak allocated/reserved CUDA
memory was 162.9/196.0 MiB. The observed rate covers model forward only, not
download, image decode, or processor time; it is not a throughput projection.

Inputs: the pinned train-label manifest SHA-256 is
`00f03f27982e2785636e85a50f4ecf258c0ac3ee483d7242e297cee14b50ec2e`; the
train-image manifest SHA-256 is
`2c56815beb9d2e9951cd38686eafdbe2ecdee05e69e74b5acfad24fbefbb9010`.
The corresponding external receipt is
`C:\CodexArtifacts\pretrained-reuse-r2\dinov2-clevr4-realimage-smoke-01\result.json`
(SHA-256 `ce0d0ad08927743bd03abd52d872713cb30ff167f7e7c869c5dd397b884c6f7c`).

## Clevr-4 100k source and acquisition status

The source is the Oxford VGG Clevr-4 100k archive, whose official page
describes 100,000 roughly balanced synthetic images and a train/test split.
The source code repository is pinned at
`sgvaze/clevr4@cddc78fb2a8359dc958987b2c750bfdd4bfd2c73` and states CC BY 4.0.
The official archive is `https://thor.robots.ox.ac.uk/clevr4/clevr_4_100k_v1.zip`;
the official expected size is 36,057,238,315 bytes and SHA-512 is
`7e8454f9f44d7b527eb20252a687b15b4cd28f3e9c79cf6147317971034c367b64bf6bba98d2a66f64b26f1e7eea2105a8acf221198342e89a844fd0007cc94c`.

The full archive has **not** been acquired or verified yet. The initial
single-stream attempt left a 116,371,456-byte partial file; it is retained but
not used. The separate eight-worker Range transfer stores disjoint 512 MiB
parts and validates every `Content-Range`. At the initial resume snapshot the
eight parts held 2,214,592,512 bytes combined (about 2.06 GiB), none complete.
After resume, five chunks had completed and 3,355,443,200 bytes (3.125 GiB)
were present across 13 part files as of 01:27 JST. This is an intermediate
size snapshot while the transfer continues; part hashes are deferred while
the downloader holds files open. No files have been extracted, no annotation
has been parsed, and neither the official test split nor any sealed audit data
has been opened. Do not use these partial bytes as a dataset.

All acquisition receipts and partial data are outside Git under
`C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-videoprism-v0\`. The
preflight JSON records official source/license/checksum metadata, local free
disk at initial preflight, and download state. Once complete, verify the full
official SHA-512 before listing members or extracting any data. Split policy,
unique-asset accounting, media hashes and a separate development split must
be fixed before training. Keep the official test split unopened until a
configuration is frozen.

## Decision

The fixed checkpoint and real-image input route work on this local GPU under
the cooperative HotUGPU handoff. This removes a runtime-feasibility blocker for
frozen-feature experiments; it does not improve any task metric or show that
DINOv2 is an adequate video or general-image representation. The next gate is
successful, checksum-verified source acquisition and a training-only split
audit. If the archive cannot be fetched reliably, preserve the partial
receipts and report the actual transfer constraint rather than silently
switching sources.

No model training, extra seeds, quantization, product promotion, cloud compute,
or quality claim was made. Existing Clevr-4 artifacts and all earlier
checkpoint/results remain unchanged.
