# Shared VideoPrism spatial path: measured load and parity limits

The earlier shared image/video baseline already uses VideoPrism, but its image
readout consumes globally pooled final features. DINO's later spatial readout
improved image quality while adding22,056,576resident parameters. Before adding
another encoder, test whether the existing VideoPrism spatial intermediate path
is usable directly. This is not a repeat of the pooled-image experiment.

## Fixed executed probe

One first-listed training image, CLEVR_new_000000, media SHA256
`35124371d35f595370acb4845341cf8481dccba94b406a828a43ea7615ef3b3f`,
training manifest`2c56815beb9d2e9951cd38686eafdbe2ecdee05e69e74b5acfad24fbefbb9010`.
Same pinned VideoPrism checkpoint/source as the existing runtime. Hashes checked.
Resize/crop unchanged. Request `return_intermediate={"spatial_features"}` and
return only those features from the JIT. This path exposes the16x16x768 grid
before temporal attention. It is not the previous768-dimensional global feature.

Compare a single frame with eight exact copies. Run each shape twice, no fit or
data/seed search. One-frame output is1x256x768; eight-frame output8x256x768.
All features are finite. Same-shape repeats are bitwise identical.

| Shape | First compiled call | Repeated call |
|---|---:|---:|
| 1frame | 3.794709s | 0.193640s |
| 8repeated frames | 4.694779s | 0.891266s |

Timing includes JAX call/materialization, not image decoding/preprocessing or
decision readout. CPU only; peak process working set4,024,647,680bytes. This
single input is not a representative latency distribution.

## Parity failed at the declared tolerance

The original rtol/atol1e-5 comparison fails for85/1,572,864 elements, max absolute
delta3.8504601e-5. Attempt01 stopped at that assertion. Attempt02 repeated the
same inference and tolerance, changing only diagnostic recording so failed
parity, features, timings and parameter groups are saved. It again reports
`declared_tolerance_passed=false`; no threshold was relaxed to claim success.

Do not silently substitute this output for a historical cache or claim exact
single/eight-frame equivalence. A future single-frame image experiment can
define it as its own frozen representation with separate extraction/training
and quality comparison. Nor is this proof that numerical differences cause
meaningful prediction differences: no readout was evaluated here.

## Actual resident parameter accounting

| Weight group | Count |
|---|---:|
| patch projection | 747264 |
| spatial encoder | 85054464 |
| spatial layer norm | 1536 |
| spatial position embeddings | 196608 |
| temporal encoder | 28351488 |
| temporal layer norm | 1536 |
| temporal position embeddings | 12288 |
| **Full loaded shared encoder** | **114365184** |

Although only spatial output is requested, all loaded encoder parameters count
toward residency. The image path introduces no second backbone or newly learned
weights. Removing DINO is a quality-dependent hypothesis, not an adopted size
reduction. The full Omni resident count and joint quality remain unverified.

## Reproduction and retained outputs

From repo root with its`src` on PYTHONPATH:

```text
C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Scripts/python.exe -m scripts.probe_videoprism_shared_spatial
```

`C:/CodexArtifacts/pretrained-reuse-r2/videoprism-shared-spatial-smoke-02/`:
config with pins/tolerance/executed script hash; all four spatial feature arrays;
result with measurements. Feature SHA256:
`d5bf6f67e5648eea0cc1c36a0b8329c4bf4d37e0d16a0a726ecfd758d562ccb1`.
Attempt01 config and subsequently recorded failure.json remain separate.
No validation/audit image, GPU/cloud, extra training, legacy checkpoint/cache
mutation or image-quality claim. Next gate is a separately frozen image-quality
comparison using this spatial representation, not promotion based on speed.
