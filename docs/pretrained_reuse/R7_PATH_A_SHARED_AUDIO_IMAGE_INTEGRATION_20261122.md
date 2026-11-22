# Path A shared audio–image task integration — 2026-11-22

## Outcome

Added the narrow spoken-color/generated-screen Decision task to a new copy of
the four-route Path A runtime assembly. The shared task uses the resident
Whisper-tiny encoder already loaded for Audio; it adds the DINOv2 dynamic-INT8
state and its existing frozen scorer without loading a second Whisper encoder.
The integrated route returned the same typed JSON as the separately packaged
runtime on one previously used TRAIN request, with maximum probability delta
`6.14e-12` on both cold and warm requests.

This is an implementation, parameter-accounting and fixed-input replay result.
It is **not** a new quality evaluation or product-quality claim. No training,
checkpoint selection, calibration, label access, evaluation split, or sealed
audit access occurred.

## Reuse check and processor handling

The standalone R7 Whisper encoder file has 67 tensors. Each tensor was compared
by name, dtype, shape and exact value against `model.encoder.*` in the pinned
Path A full Whisper-tiny checkpoint:

- R7 encoder export SHA-256: `d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8`.
- Path A full checkpoint SHA-256: `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`.
- Compared tensors: 67; exact mismatches: 0.
- Revision: `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`.

The two `WhisperFeatureExtractor` configurations differ: the Path A Speech
Commands processor embeds `mel_filters`, while the R7 processor relies on the
Transformers-generated default. The integration therefore shares only the
verified encoder module and retains the R7 processor configuration with the
spoken-palette task. It does not silently substitute preprocessing from the
unimodal Audio route.

## Integrated package and resource measurements

- Base assembly: manifest-only verified
  `path-a-modular-runtime-assembly-v4-clean-copy-20261026-01`.
- Base manifest SHA-256:
  `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
- Reproducible final output (outside Git):
  `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4-spoken-palette-20261122-attempt-02`.
- Integrated manifest SHA-256:
  `4db84161ec2132528af8d4887e2c0d22fae407edc80682cf802a0667fa3e7de0`.
- Unique logical pre-quantization parameters: **192,654,853**. This counts the
  shared full Whisper checkpoint once and includes DINOv2's 22,056,576 logical
  parameters despite dynamic-INT8 storage.
- Package payload: **712,388,681 bytes** (about 712.4 MB), excluding the root
  package manifest. The provisional 200 MB package goal remains unmet.
- Integrated Audio/Text Torch worker after load: RSS `803,512,320` bytes in the
  four-route replay. VideoPrism worker RSS: `816,234,496` bytes; combined
  worker RSS: `1,619,746,816` bytes. The spoken-palette one-task worker's RSS
  after warm inference was `856,133,632` bytes.
- One fixed TRAIN request: worker model load `1.516 s`; first dispatcher call
  including process startup `8.229 s`; warm request `0.340 s`. These are single
  request timings, not latency benchmarks.
- No GPU or cloud compute was used.

The worker's reported `56,866,820` is its Torch `nn.Parameter` count and omits
packed dynamic-qint8 buffers and the separate VideoPrism process. It is not a
replacement for the full package's logical count above.

## Runtime evidence

The raw-input smoke used the same frozen first TRAIN source as the prior
standalone R7 runtime smoke: sample
`bc0d8bde512c9f14a4f98f883a47fbef006c2d48c32f2d9057cae7a7acff96ec`, audio
SHA-256 `8408220551860f06b8ea61187802066a0b784578f88e130bfe9fdbec97cedd87`,
and screen SHA-256
`de0cee89cdf50832df82e16cf8bf83afafd3293d3e7265920deaf75bb1dc2f53`. The
worker confirmed the exact same Whisper encoder object was used by both task
paths. The result remained normalized, kept the supplied option order, and
matched the prior selected option and probabilities.

- Final smoke attempt 03 report:
  `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-shared-whisper-audio-image-smoke-20261122-attempt-03\shared-path-a-runtime-smoke.json`.
- Report SHA-256:
  `7e01bb5815530738602a1296de174df022694c8bf1897365695c6708ee169051`.
- Execution source HEAD was `e48ba9713c200cfec5bc307aad390b807cdcd7c4`; the
  working tree contained implementation changes at execution. The smoke
  receipt pins their exact content: package builder
  `ebf4031d38465521f22a77368f43e8410adf8c3de462505de8fa4a47090303da`, shared
  runtime `31e4d9e062a1a3f8d68c76baa46571ad5da3751cec99a2339b1f0de3c7a3ee53`,
  dispatcher `158c4dc6b0f9250c538e642e47a1581b79d5b3a051c9e663859e77bf9cdf8341`,
  Torch worker `61d0acccbed8c3fdfc80d90b15ecd772fccce0fda00d522eb06431087ef9c85a`,
  and smoke script
  `805bada7c7fef47460aed2aaed7cef3346d98cb476147b5fdffe89f6215ff9c8`.
  Those exact implementation files are committed at
  `4a4e312d593542b6226d7f71ce437872a475fb97`.
- The base R7 output was a previously retained TRAIN-only smoke receipt; no
  Evaluation or sealed-audit rows were used for this integration.

The existing four-route CPU replay still returned typed JSON for every route.
Video, Image and Text met their previous probability-parity checks. Audio
retained the known CPU-FP32-versus-saved-CUDA-BF16 discrepancy (`6.1774e-4`
maximum probability delta on that fixed request), so the aggregate strict
parity result remains **failed**. The tolerance and default precision were
not changed.

- Four-route replay report:
  `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-shared-four-route-cpu-replay-20261122-attempt-01\dispatcher-benchmark.json`.
- Report SHA-256:
  `1e4786d0eaead2944f91e83cff06c10f1059bf98a23f110a430e45347dd051c7`.

## Verification and limits

- Focused tests: 27 passed across `test_spoken_palette_runtime.py`,
  `test_runtime_dispatch.py` and `test_modular_bundle.py`.
- Ruff passed for the changed runtime, worker, packaging/smoke scripts and
  tests.
- The original standalone R7 bundle, base Path A assembly, checkpoint files,
  corpus and outputs were read-only. All new package and smoke outputs use
  distinct external attempt directories.
- The spoken task still uses generated still screens and an already exposed
  development Evaluation cohort in its historical quality report. Its
  99.99%/98.70% results are not broad image/audio quality evidence and were
  not recomputed here.
- The assembly is still task-scoped; this integration does not add a natural
  audiovisual/video task to the shared spoken-color model. Text, Image, Audio,
  Video and sensor quality remain separate task-specific evidence, and the
  four-modality 95–98% objective remains unproven.
- The first package build correctly refused the original assembly because it
  contained files absent from its manifest. The source was not changed; the
  build was rerun from the existing manifest-only verified copy.

Keep this as a shared-runtime engineering step, not a product promotion.
