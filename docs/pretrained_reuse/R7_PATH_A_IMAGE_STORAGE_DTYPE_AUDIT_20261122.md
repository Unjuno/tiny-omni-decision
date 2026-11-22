# R7 Path A image storage dtype audit — 2026-11-22

## Result

The latest encoder-only four-route package is 594,168,210 payload bytes
(package manifest excluded), with 163,102,597 unique logical parameters. A
metadata-only inspection of its largest weight archive found that
`components/image/videoprism-base-f16r288.npz` contains 114,365,184 parameters
stored as **float32**, despite `f16` in its filename. Its SHA-256 is
`61fa62eedbe4f8d4ba76a99b8b79abd0137426a4931c48204ab70446b6411f29`.

This is a storage-format finding, not evidence of a runtime or quality bug.
The existing package and archive were not modified. The archive's 40 NPY
headers were read without materializing tensor values. The package manifest
SHA-256 is `616b9b61e983d875a6a29e71d568f1ba0a6aeba6705da61ee66a88b4be43b3fd`.

## Payload profile

| Component | Bytes |
|---|---:|
| Image (including 457,475,296-byte VideoPrism archive) | 458,147,497 |
| Audio | 108,886,352 |
| Spoken-palette AV | 25,134,558 |
| Video readout/runtime | 1,198,068 |
| Text readout/runtime | 801,298 |
| Root README | 437 |
| **Total** | **594,168,210** |

The VideoPrism archive alone is 76.99% of total payload. Float32 values
account for exactly 457,460,736 raw bytes; the remaining archive bytes are
container/header overhead.

## Arithmetic-only size envelope

Holding every other byte fixed and changing only the VideoPrism numeric
payload gives these optimistic lower-bound estimates:

| Image weight storage | Estimated package payload | Under 200,000,000 bytes? |
|---|---:|---|
| Current float32 | 594,168,210 | No |
| float16 | 365,423,282 | No |
| int8 | 251,058,098 | No |
| packed 4-bit | 193,875,506 | Barely, before scales/metadata/runtime overhead |

These are arithmetic estimates, not exported artifacts or verified inference
results. They show that simply storing VideoPrism in float16 would leave the
package about 165 MB over target. Even int8 image weights alone would remain
about 51 MB over target. A packed 4-bit export is the first image-only
precision level that could fit on paper, but would need a real implementation,
runtime support, measured package accounting, and task-matched quality checks.

## Scope and next engineering gate

- Read only the pinned `path-a-modular-runtime-assembly-v4-whisper-encoder-only-20261122-attempt-01` package manifest and NPY headers.
- No tensor values were loaded, no package was written, and no model was run.
- No validation, calibration, evaluation, or sealed-audit records were opened.
- No training, GPU, cloud compute, or quantization was used.
- Existing components, checkpoints, corpora, and external artifacts remain unchanged.

The next useful size experiment is a separate, non-overwriting VideoPrism
precision export with deterministic runtime parity and a measured full-package
byte count. Any quality comparison must use the already declared development
validation protocol and must not be presented as independent evidence; the
currently exposed evaluation cannot become blind again.
