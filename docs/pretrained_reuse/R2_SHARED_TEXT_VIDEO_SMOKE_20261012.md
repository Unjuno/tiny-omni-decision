# Shared-text video smoke

Implementation check only, not quality evidence. The initial attempt01 failed
before any token extraction because the separate JAX Python could not import the
repository src package. Preserve its config/request/worker.log. The new runner
now adds repository src/root only to the child's environment and working directory;
it does not install packages or change the parent's environment. Regression test
covers path ordering and parent-environment preservation.

Attempt02 passed: six selected scenes were re-extracted with the existing CPU
VideoPrism worker. Every token SHA-256 matched the old full-run extraction record,
with unchanged pooled-feature tolerance. Text token ID/order digests match between
MiniLM and the frozen Cross-Encoder. Both produce the unchanged1536-wide interaction
features; shared model count22,713,601 includes its unused classifier for honest
resident accounting.

One epoch over12 train and12 validation choice rows per condition exercised
MiniLM/shared text crossed with attention/no-video readouts. All four start from
the exact same head tensor hash and use the same sample order; finite-gradient,
source-hash and exact best-reload gates passed. No checkpoint is adopted from this
smoke. Full24-epoch comparison remains separate, using a new output directory.

Full result and token/feature hashes are in the adjacent JSON. The script was
first exercised uncommitted; exact script hash is in each local config. Both
attempts remain at `C:/CodexArtifacts/pretrained-reuse-r2/shared-text-video-smoke-01`
and `shared-text-video-smoke-02`. No sealed audit was opened or old run modified.
