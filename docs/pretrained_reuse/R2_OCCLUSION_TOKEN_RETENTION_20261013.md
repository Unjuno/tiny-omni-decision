# Occlusion token retention diagnostic — 2026-10-13

## Fixed scope

Inference only: first two train videos in the frozen manifest, first and last windows of each. No validation scoring, training, checkpoint selection or semantic localization claim. Same pinned VideoPrism weights, source revision, decoded input hashes and CPU JAX backend as the completed dense extraction. Fresh output directory; existing artifacts unchanged.

## Results

All four windows completed with exit code0. Full output grid is8 x16 x16 x768 FP32. Averaging these tokens exactly reproduces the corresponding stored768-vector (maximum absolute difference0). Reindexing horizontal token positions changes the grid but leaves its mean equal within floating-point summation error. This is a token permutation, not physically mirrored input.

| Train video | Window | Token relative L2 after reindex | Mean max absolute change |
|---|---:|---:|---:|
| video_10103 | 0 | 0.639465 | 9.53674316e-07 |
| video_10103 | 7 | 0.677050 | 9.53674316e-07 |
| video_10186 | 0 | 0.656744 | 9.53674316e-07 |
| video_10186 | 10 | 0.680720 | 1.49011612e-07 |

## Decision

The pooled interface discards explicit token arrangement; full grids can be retained reproducibly. Neither nonzero variation nor permutation invariance proves that target object identities are represented, that they can be read out reliably, or that pooling caused the failed quality comparison. Positional embeddings may also leave implicit location information in pooled values. Do not treat this algebraic control as a new accuracy result.

Next gate: determine whether a compact spatial readout can recover known train-only visual attributes under content-preserving controls before paying for full token extraction or another validation-selected fit. A token-grid storage feasibility result alone does not justify production adoption or greater model capacity.

## Reproduction

`python -m scripts.audit_occlusion_token_retention` in the existing CPU JAX environment. The script deliberately requires a fresh output path. Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-token-retention-01/`; config pins parent report and script hashes, result pins each token file. No weights were changed.

## Boundary-frame visual check

Contact sheets of all16 selected frames per video were inspected. The initial
windows show the tabletop setup before the hiding action; the ending windows show
covered objects and hands. These boundary windows alone do not visibly establish
the hidden-object location. Do not attach whole-video target labels to these
individual windows as if independently observable. Both preselected videos have
initial and final target index1, so this tiny diagnostic also cannot establish
condition switching. A future semantic probe must account for presentation and
movement intervals and actor-relative coordinates. No data or labels were changed.
Local contact sheets: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-token-contact-01/`.
