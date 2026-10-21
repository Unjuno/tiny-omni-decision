# Shared global readout: algebraic export and raw image replay

## Fixed question and scope

The previous global arm repeats the same 768-dimensional cell mean sixteen
times, dividing each copy by four. Sum the corresponding linear weights and
divide by four instead. Retain the separate 768-dimensional spatial summary:
different floating point averaging paths need not produce identical means.
No fitting, checkpoint selection, new data, quantization or encoder changes.
The original global and spatial artifacts remain intact.

Predeclared gates: logits atol1e-4/rtol1e-5, probabilities absolute1e-5,
metric absolute1e-5, zero changed answers, exact folded checkpoint reload.
This transformation is explicitly rejected for the spatial arm. Unit tests
check float64 function equivalence, original weights and caller RNG preservation.

## Measured export

| Item | Original | Folded |
|---|---:|---:|
| Observation dimensions | 13,056 | 1,536 |
| Readout parameters | 1,720,576 | 246,016 |
| Readout checkpoint bytes | 6,884,152 | 986,000 |
| Median batch128 CPU scorer time | 20.078ms | 3.075ms |

CPU timing uses two Torch threads, float32, three warmups and twenty interleaved
trials per arm with inputs already resident. The 6.53x ratio is scorer-only,
excluding media decode, encoder, feature assembly and text embedding; it is
not an end-to-end speedup or service latency guarantee.

All 512 existing selector images / 2,048 questions replayed. Zero answers
changed. Maximum logit difference3.815e-5 and probability difference3.636e-6
pass the original tolerances. Folded save/reload outputs match exactly.

| Metric | Folded |
|---|---:|
| Accuracy | 0.907714844 |
| NLL | 0.224677270 |
| Brier | 0.124737206 |
| ECE15 | 0.012183266 |

All metric differences from the saved original GPU predictions are <=1e-5.
This is output preservation on reused development data, not new quality evidence.
Counting remains66.41%; no95% claim.

The conceptual image stack counts the full frozen VideoPrism114,365,184,
MiniLM17,389,824 and new readout246,016 = **132,001,024 parameters before
quantization**. This does not count as a complete deployed Omni model. The
large encoder checkpoint remains referenced, not duplicated or quantized;
the total package-size gate remains open.

## Raw images to typed JSON

Eight fixed selector images, first batch in the original manifest order, were
decoded again using the original one-frame, batch8, 288 center-crop CPU JAX
profile and pinned VideoPrism checkpoint. Spatial cells are exactly equal to
the original cache (maximum difference0). No cache-only claim is substituted
for this actual encoder run.

- Model load:0.313s; decode/preprocess:0.129s.
- First batch including JIT:4.755s; warm batch:0.876s.
- Peak encoder-process working set:3,315,384,320bytes.
- Folded CPU scorer produced32 decisions; all match saved answer identities.
- Deterministic `classification_json` emits custom option IDs with probabilities.
  Reversing supplied option order preserves the selected ID in all32 cases.

Encoder and scorer run in separate Python environments/processes. These numbers
do not measure a combined resident service or single-image latency. The JSON
interface accepts the complete fixed10-class taxonomy for each of four image
tasks; arbitrary semantic options and joint multimodal decisions are unverified.
MiniLM option embeddings are hash-verified cached outputs of the pinned model.

## Reproduction and provenance

Use the normal Torch environment for `python -m scripts.export_shared_global_readout`
and `python -m scripts.verify_shared_global_json`. Between them, run
`python -m scripts.replay_shared_global_raw_images` with the existing VideoPrism
JAX environment and repository `src` on PYTHONPATH. These scripts use fresh
fixed output directories and refuse overwrite; preserve existing outputs.

Local artifacts under `C:/CodexArtifacts/pretrained-reuse-r2/`:

- `shared-global-folded-readout-01`: new weights, all2,048 predictions,
  config, telemetry and exact executed-source copies.
- `shared-global-raw-image-replay-01`: raw-image feature replay and timing.
- `shared-global-json-replay-01`:32 typed JSON decisions and verification.

The export ran from base commit `ae6dc8e` plus then-uncommitted export/helper
code. Its exact source is retained under `executed-source` and its hash is
verified against the saved config. The implementation was committed in
`383a62a`; subsequent script edits only wrap overlong lines. Raw encoder replay
ran from that commit. Machine-readable result and executed-source hashes are
in the companion JSON. Folded weight SHA256:
`e78dbaa31a251f2668d411329ea05d9b0998d66aca6de0d0911991a1bc2bb6f9`.

571 CPU tests pass,1 skipped; Ruff and manifest/catalog validations pass.
No GPU training, cloud expense, sealed audit or old artifact replacement.

## Remaining work

The readout redundancy is removed without a measured decision change. Preserve
this as a task-scoped runtime candidate. Native multi-input fusion quality,
arbitrary conditions, complete resident parameter accounting, package size,
and general per-modality95–98% quality still require work. This export does not
justify compression or product promotion. The next integration must preserve
these input/feature contracts while handling actual multi-input observations.
