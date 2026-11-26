# R0 artifact-capacity recheck — 2026-11-03 22:27 JST

## Scope and safety

Read-only inventory and exact-prefix checks were performed before any cleanup.
No media, corpus, checkpoint, model, feature, source archive, or user manifest
was deleted or moved. The two untracked PhysicalAI manifests in the worktree
were left untouched. No sealed-audit data was opened, and no GPU work was
started.

The checked-out roadmap worktree is
`C:\CodexWorktrees\r2-perception-control-freezer-20261031`, branch
`docs/pretrained-reuse-roadmap-20261010`, at
`7011e85254985f88416df643ae765d8f56dc59a5`. Fetching the branch confirmed the
remote points to the same commit.

## Reusable RoboMNIST feature cache

The fixed cache remains at
`C:\CodexArtifacts\pretrained-reuse-r2\robomnist-av-match-feature-cache-20261103-attempt-01`.
It has **1,778 verified group receipts out of 1,780**. The two absent groups
are `Ar1_Vhigh_Ac1_Rp10_UNC085` and `Ar1_Vhigh_Ac1_Rp22_UNC304`; both belong to
the frozen validation split. Reducing the split or evaluating without them
would change the fixed protocol, so the readout was not fit.

The cache's effective contract still pins manifest SHA-256
`3f4575ba41d944310e1de83cbdc68385390581723262245b97dc0d2c71db660f` and
extractor SHA-256
`daf83f8e504b63173e6bd08a6a5253badbd6deff055e0597b5facbcf0424304d`. Resume
used the recorded CPU-only runtime: Python 3.11.16, NumPy 1.26.4, JAX 0.6.2,
TensorFlow 2.18.1, Torch/Torchaudio 2.6.0+cu124; CUDA was not initialized and
TensorFlow saw no GPU.

The first valid resume attempt rechecked existing receipts, then stopped at
2,145,927,168 free bytes, 1,556,480 bytes below the 2 GiB guard. After three
stable capacity readings averaging 2,165,060,949 bytes, a second attempt
rechecked through manifest position 506 and stopped at 2,147,328,000 bytes,
153,648 bytes below the guard. Neither attempt created a feature for either
missing validation group. At 22:27:17 JST the drive had 2,152,390,656 free
bytes, only 4,907,008 bytes above the guard; no process remained active.

The run metadata's `completed_groups` field now says 506 because the current
implementation counts groups visited during the latest resume invocation,
including already-complete receipts. It is not the total number of unique
completed groups. The receipt inventory (1,778) is authoritative. The frozen
script was not edited and `run.json` was not manually edited; the resume itself
updated its progress and error fields. Changing the script's implementation
hash would invalidate the resume contract.

## Cleanup candidates

The remaining 19 `.part`/`.tmp` files were not removed. Two same-named full
files were checked byte-for-byte against partials: the HoloAssist
`R115-10Aug-Navvis-Part2.mp4.part` and GRID `s4.zip.part` are **not** exact
prefixes of those files. GRID audio partials have no same-named complete file
in the checked store. Earlier cleanup records also retain HoloAssist partials
that do not match their proposed canonical copies.

The worktree contains 15,251,027 bytes of regenerable Python, pytest, and Ruff
caches. The attempted cache removal was rejected by the automatic action
review before execution; none were deleted, and no alternate deletion route
was attempted. These caches are too small to provide a robust reserve for the
feature-extraction resume in any case.

## Disposition

No reviewed item met the evidence bar for safe removal. R2 cannot resume until
stable disk headroom above the 2 GiB guard is available. Preserve the 1,778
completed features and resume the same cache only when capacity is sufficiently stable;
do not change the split, lower the guard, or rerun the readout on a reduced
validation set. No quality metric has been produced by this attempt.
