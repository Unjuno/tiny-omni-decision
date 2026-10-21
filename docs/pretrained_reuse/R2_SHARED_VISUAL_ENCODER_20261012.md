# Shared image/video encoder: actual alternating execution

## Design and preserved contracts

One VideoPrism model object and one device-resident parameter tree serve two
explicit profiles. Jitted functions take parameters as arguments, rather than
capturing the weight tree as a constant. The image profile returns17x768 cells
from one frame per image; the video profile returns8x768 time-bin features from
eight frames. This is shared visual encoding, not an audio/sensor Lens result.

No fitting, checkpoint selection, quantization or legacy modification. The
worker checks pinned checkpoint/source/media/cache hashes before execution.
Input validation rejects wrong modality, frame count, size, dtype, nonfinite
pixels and values outside[0,1]. Output shape/finiteness are checked. A later
CPU regression also adds explicit rejection of a wrong image-output batch size.

## Measured execution

Source commit `92c09fa`; exact runner/helper hashes and input pins in companion
JSON. CPU JAX0.6.2, same existing VideoPrism environment. Local artifacts:
`C:/CodexArtifacts/pretrained-reuse-r2/shared-visual-encoder-probe-02`.

Inputs: first eight existing CLEVR-4 selector images and the already-used
VIDIMU training video S40A01T01, with the fixed past window[0,4000]ms. The
eight selected timestamps end at4000ms; no later video frames are decoded.
The video reference is the first previously saved past-window feature tensor,
not a full-episode classifier input. No audit payload or new quality set.

| Request | Forward/profile seconds | Max cache difference | Working set bytes |
|---|---:|---:|---:|
| Image batch8, first call | 1.333483 | 0 | 1,108,168,704 |
| Video batch1, first call | 1.553220 | 0 | 1,142,870,016 |
| Image batch8, repeat | 0.922695 | 0 | 1,122,820,096 |
| Video batch1, repeat | 1.124698 | 0 | 1,143,517,184 |

All comparisons pass the predeclared atol1e-4/rtol1e-5 gate with actual zero
difference. Repeat outputs are exactly equal despite intervening other-modality
calls. The full parameter-tree hash is unchanged before/after all four calls.

Unique model parameters: **114,365,184**, shared by both profiles. Peak process
working set: **1,195,708,416bytes**. This is a real shared-tree execution, not
two model counts added together or a file-size-only estimate. Internal backend
buffer deduplication is not proven, and compiled executables/workspaces also
consume memory. The process count does not include other Omni encoders/heads.

First-call timings include any compilation performed in that environment;
persistent compilation-cache state was not independently cleared. They are not
guaranteed cold-start timings. Forward measurements exclude source hashing,
model loading, media decoding and preprocessing. Two calls per profile are
insufficient for tail-latency or throughput claims. Prior single-profile
processes used different execution paths, so no causal memory-speedup factor
is claimed against them.

## Reproduction and checks

Run `python -m scripts.probe_shared_visual_encoder` using the existing JAX
environment with repository `src` on PYTHONPATH. The fixed output directory
refuses overwrite. Keep the original artifact for reproduction comparisons.

Unit tests check a shared explicit parameter object for both functions, fixed
shapes and fail-closed invalid inputs. Full CPU suite:574 passed,1 skipped;
Ruff and workflow manifest/catalog commands pass. No new GPU training or paid
compute. Final pushed commit CI is checked separately.

## Preserved first attempt

Attempt01 under `a738061` also had exact cache/repeat parity, but its hash
closure used the initially captured leaf list. Attempt02 hashes the current
encoder tree before and after execution and includes the added batch-output
check. Both artifacts are preserved; their feature-file hashes are identical.
This was a stronger integrity recheck, not a changed model or quality search.

## Remaining gate

This makes one shared feature service implementable while preserving measured
image and video representations. It does not establish a combined model's
quality, arbitrary options, synchronized sensor/audio fusion, package size or
95–98% accuracy. The video window only becomes available after its last evidence
frame and actual computation; individual token positions are not online causal
availability timestamps. Preserve these timing/profile distinctions when
connecting the shared worker to the typed observation and decision layers.
