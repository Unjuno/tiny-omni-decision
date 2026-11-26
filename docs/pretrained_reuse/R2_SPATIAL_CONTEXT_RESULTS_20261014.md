# Complete frozen spatial context corpus

Execution passed; this is feature/cache integrity evidence, not QA quality.
All99 original training and257 observed-development videos retain their source
cuts and32 uniform frame IDs. No source annotation/answer is read by extraction.

## Measured execution

- Source commit `eb70e282a6e488eb13c55447ccd16cef48630327`.
- Command: `python -m scripts.extract_spatial_context_corpus --output C:/CodexArtifacts/pretrained-reuse-r2/spatial-context-corpus-01`.
- Start/end UTC: `2026-10-14T01:11:36.841117+00:00` / `2026-10-14T01:26:57.703064+00:00`.
- Body920.854241seconds; summed synchronized encoder windows84.454354seconds.
  Body includes decode, hashing, FP64 NumPy pooling, file writes and CPU tests
  run concurrently during part of extraction; it is not uncontended GPU throughput.
- RTX3080Laptop16GB; Torch2.6.0+cu124, FP32, deterministic algorithms, TF32off.
- Peak allocated677,198,848bytes; reserved834,666,496bytes. Observed process
  working-set peak during execution1,419,526,144bytes is a partial observation,
  not a measured final process peak. Local compute only; cloud cost zero.
- Saved1,068 NPY arrays /4,514,740,736file bytes; appearance per video32×256×384,
  positions32×256×3 and original frame IDs32. All source cells contribute to
  fixed16×16 mean pooling; small-object semantic identity preservation is unproven.

## Integrity and replay

Pinned TAPIR source `730cda1c730877cfedbe01bf87fb1cadb78a565d`, released checkpoint
revision `5d3fb48e76c5422841e38501514121e251beabb7`, exact31,072,327parameters.
Source material is the previously reviewed Perception Test train/validation
source revision `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`; no sealed/test data.

All frozen state bytes remained identical:
`3c92ad70854f549bc46fde282f4248e949d733fadffcfa2b060afada2ea6dd4a`.
Every clip left CPU/CUDA RNG unchanged. Eight earlier full-grid references and
the first-clip repeat match bitwise. The independent completed-cache loader
checked1,070files including result/config, all IDs/media/time/positions and zero
cross-split selected-RGB overlap. A separate fresh process re-extracted the first
training clip and exactly matched appearance/positions/frame files.

Result SHA-256: `a306930eb673a00c246b7b2e9caad3044315cb9f19b0f5ee2c43987d2f04f0da`.
Effective config SHA-256: `d17c1f1f9e55f17d58983603c1c8bf27ac171bed9d2ee04108535c84b15e35ff`.
Full per-file/media/source hashes remain in the local result/config. Audit:
`C:/CodexArtifacts/pretrained-reuse-r2/spatial-context-independent-audit-01.json`.
Fresh replay: `C:/CodexArtifacts/pretrained-reuse-r2/spatial-context-fresh-reload-01/`.
Existing corpus/weights/artifacts are unchanged. Partial progress is rejected
as training authority. No new runtime parameters are added by this cache.

## Next experiment

[Fixed paired spatial readout protocol](R2_SPATIAL_READOUT_PROTOCOL_20261014.md)
compares actual time-varying appearance with broadcast initial appearance under
identical architecture/init/sampling. Feature variation alone is not proof of
useful temporal reasoning or the complete four-modality product.
