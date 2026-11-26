# R2 open source-screen PR reconciliation (2026-10-28)

This is a read-only reconciliation of open Draft PRs #35 and #36 against the
pretrained-reuse roadmap. Their branches and external artifacts remain separate
and untouched. No source media, labels, or model outputs were opened for this
reconciliation.

| PR | Branch / head | New evidence | Current disposition |
|---|---|---|---|
| [#35](https://github.com/Unjuno/tiny-omni-decision/pull/35) | `codex/r2-zod-cam-lidar-screen-20261028` / `3160975fac62e1c1cb79ad91dc94b0f3b7bf2bcb` | Pinned SDK schema has a shared annotation UUID linking 2D and optional 3D boxes. An archive/torrent descriptor and size metadata are recorded. | `REVIEW`; do not fetch payload. Access method/terms, drive-level split integrity, pretraining overlap, and a fixed task showing both-input benefit remain unresolved. The PR reports its test check passed; PR remains open and draft. |
| [#36](https://github.com/Unjuno/tiny-omni-decision/pull/36) | `codex/r2-fzi-aura-source-screen-20261028` / `57aa474d9f15b2f70d9f9dc6fb441e69d377ba99` | FZI-AURA SDK requires accepting the dataset terms and signing in. One sample was previously retrieved over an anonymous path without that acceptance; it is quarantined outside Git and must not be read, evaluated, or used. DAVE has an explicitly both-required AV-alignment diagnostic, but combines non-commercial and agreement-bound source corpora. | FZI-AURA `HOLD`; no further sample reads or use absent provider-authorized access resolution. DAVE `REVIEW`; do not use without resolving each source's rights and agreement. PR reports lint/tests/manifest CI passed; it remains open and draft, with GitHub currently reporting `mergeable=false`. |

## Effect on the active plan

Neither PR admits a new training/evaluation task under the current policy. ZOD
is potentially useful for image/LiDAR object association but lacks established
rights/access and group integrity. DAVE is a strong diagnostic design for
testing audio-video complementarity but cannot be admitted from the package's
MIT label because upstream source terms differ. FZI-AURA's sample is not an
available project asset. No repeat fit, media download, GPU evaluation, PR
merge, or artifact change is justified by this evidence.

The R2-M gate remains open for a naturally captured, rights-cleared,
group-disjoint task where both proposed inputs are necessary and can be tested
against matched unimodal, metadata-only, shuffled, stale, missing, and timing
controls. Keep existing branches, CI records, and the quarantined FZI-AURA
artifact unchanged.

## Verification scope

PR metadata, changed-file names and reviewable documentation patches were
retrieved read-only from GitHub on 2026-10-28. The source-screen reports are
[`R2_ZOD_CROSS_BRANCH_RECONCILIATION_20261028.md`](R2_ZOD_CROSS_BRANCH_RECONCILIATION_20261028.md),
[`R2_FZI_AURA_SOURCE_SCREEN_20261028.md`](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-fzi-aura-source-screen-20261028/docs/pretrained_reuse/R2_FZI_AURA_SOURCE_SCREEN_20261028.md),
and
[`R2_DAVE_SOURCE_TASK_SCREEN_20261028.md`](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-fzi-aura-source-screen-20261028/docs/pretrained_reuse/R2_DAVE_SOURCE_TASK_SCREEN_20261028.md).
