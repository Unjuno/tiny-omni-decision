# GRID one-slot correspondence construction: infeasible

The [predeclared one-slot procedure](R2_GRID_HARD_CORRESPONDENCE_PROTOCOL_20261028.md)
failed closed with `ValueError: empty quartet set` before writing manifests.
Structural counts on the pinned train and development alignment metadata found
**zero same-speaker one-slot neighbors for every anchor**: 23,992/23,992 train
and 7,999/7,999 development anchors. No media, predictions, model, or sealed
audit was read. This is a property of this corpus/selection, not model quality.

A separate retrospective enumeration found at least three same-speaker
two-slot neighbors for every anchor in both development partitions. A new
two-slot protocol may therefore be frozen separately. Do not describe it as
the original one-slot task, compare its scores to one-slot performance, or
change it after observing model results.

The actual executed script is preserved at
`C:/CodexArtifacts/pretrained-reuse-r2/grid-hard-correspondence-failed-20261028-attempt-01/executed-script.py`,
SHA-256 `6440b82aa4ff46ddf4012414bb4d7980f42ae4213b2d463ed65731225335cb00`.
The repository copy subsequently received a formatting-only correction for
Ruff; the failed attempt and its output directory remain untouched.
