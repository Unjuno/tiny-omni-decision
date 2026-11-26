# Fixed input-only appearance association diagnosis — before execution

The full <=200M pre-quantization Omni objective remains unchanged. Numerical
track-aligned sampling passed, but descriptors changing does not prove object
association. This bounded train-only diagnosis measures that missing connection;
no learning or new QA/checkpoint selection follows it automatically.

## Input-only prediction

Use the same eight known training clips, all256 tracks, all32 retained source
frame IDs, and byte-pinned track-appearance-primitive-01 arrays. The query bank
is all256 actual-track descriptors at the first sampled time. At every time,
retrieve the largest cosine-similarity descriptor among all256 candidates:

1. actual-track per-time descriptors, with their actual estimated coordinates;
2. first-position per-time descriptors, with first coordinates broadcast in time;
3. actual descriptors rolled64 candidate indices, while coordinates stay actual.

Normalize only for cosine retrieval, in FP64. Deterministic first-index tie
handling; save maximum scores, margins, tie counts and zero-norm flags. No point
is removed for support, uncertainty, similarity, GT role or labels. Save every
prediction before opening annotations. A fourth fixed-index control follows
the original model trajectory p at all times, without appearance retrieval.

## Posthoc geometric comparison

Only after prediction, open pinned all_train.json. Use the exact grounded
"Track these similar or identical objects." group to define the physical game
objects for *reporting only*, retaining variable group counts and every group ID.
No box, group ID, answer, mask flag or label enters prediction or chooses queries.
Do not substitute is_occluder/is_masked or map answer IDs to physical ranks.

Source boxes are sparse: linearly interpolate each normalized bbox component
within its annotated frame range only, never extrapolate. Record this proxy-label
limitation. This does not establish visible pixels, point tracking GT, semantic
recognition, target-hidden-item association, actor-relative semantics or QA.
Out-of-raster/none/unavailable and multiple-group bbox membership remain explicit.
When any group's annotated range is unavailable at a time, every point's group
identity at that time remains unknown; retain the per-group availability mask.
Partial coverage cannot establish a unique initial label or a resolved future hit.

Assign each initial query its unique group bbox ID if available. Predict for all
queries, then report lower-bound future same-group bbox hits over all initially
unique-group queries ×31 later times. Future unknown/overlap counts stay in that
denominator and are not counted as hits. Report conditional resolved-hit rate
separately with its denominator, initial coverage of all256 queries, per-object
counts, unknown/ambiguous predictions and all-query point-times. Time0 self-match
is a separate sanity result, never part of the future association statistic.

Compare actual/static/fixed-index/misaligned-feature arms without tuning any
threshold, prompt, seed, selector or label definition. Results are training-source
diagnosis on eight reused scenes, not new validation or an Accuracy target.

## Implementation and resource gates

Pure NumPy helpers keep retrieval separate from posthoc box labels. CPU tests
prove frame routing, candidate permutation, tie/zero behavior, no mutation,
within-range interpolation, overlap/unknown retention, and malformed-input
rejection. Use CPU four BLAS threads, existing dependencies, new output paths,
300s admission checks between clips/at completion and8GiB disk reserve.
No external watchdog is claimed. Save config/source/file hashes and raw matches,
then independently recompute reported counts and exact fresh-process predictions.
No encoder/model load, optimizer, GPU learning, cloud, dev/audit data, old artifact
overwrite, quantization, main merge or product promotion. Numerical/association
evidence must change the next action; do not use this diagnostic to justify more
epochs on the failed99-scene readout by itself.
