# R0/R1/R2 cross-branch recheck — 2026-10-31

## Snapshot and preservation

- Repository: `Unjuno/tiny-omni-decision`.
- PR #15 remains **open / draft**, from
  `docs/pretrained-reuse-roadmap-20261010` to `main`; the observed head is
  `66810de2534fc0f8186f5e8ecc7968b6f984fc6d` and the observed base is
  `09246d7b1821662a6b1eba47fc08c43f9dc3863c` (GitHub REST API,
  2026-10-31 02:51 JST).
- The active user worktree remains on
  `codex/teacher-quality-clean-dev-v1` at `041c5954`; it contains pre-existing
  uncommitted changes. No existing user file, branch, worktree, run, dataset,
  checkpoint or pull request was reset, deleted, stopped or overwritten.
- At 02:51 JST, the local RTX 3080 Laptop was idle at 0% / 0 MiB of 16 GiB.
  No Python model worker was present. C: had 31,842,226,176 bytes free.
- A separate Orbitune full-validation job had completed by 02:51. Two Python
  processes (PIDs `20488` and `18284`) had run the same command against the
  same checkpoint and output path concurrently, so writer provenance is
  ambiguous. The resulting JSON parses and reports checkpoint step `220813`,
  42,323 windows / 21,669,376 events, 744.592 seconds, and event-weighted
  trainer loss `-1.430447`. Its SHA-256 is
  `917773009828DB39C3979A75AC70005B5534122C880739DA488C7DCD5C3EC56C`.
  This is an Orbitune training-task validation result, not Tiny Omni quality
  evidence; no Tiny Omni training or inference was launched for this snapshot.

## R1 checkpoint decision

The exact V-JEPA 2.1 ViT-B distribution checkpoint remains **REVIEW** for
product use. The 2026-10-30 official-source and local-hash recheck found no
checkpoint-specific license grant; the official object is pinned by S3
version ID, and its local hash still matches the prior audit. The already
audited VideoPrism-B checkpoint remains the admitted Path A video feature
source. See [the rights recheck](R1_VJEPA21_LICENSE_RECHECK_20261030.md).

## R2 image evidence and data gate

The independent `codex/r2-inat-expansion-roadmap-20261029` evidence at
`e3909424` adds a fresh-observer transfer readout and a later metadata window:

- The fixed VideoPrism-B readout scored `74.88%` Accuracy, `1.1258` NLL,
  `0.4004` Brier and `0.1562` ECE on 430 new observer-disjoint validation
  images. Its selected epoch-10 checkpoint improved over the prior readout on
  those same IDs, but train/validation gaps were large and validation NLL
  worsened after epoch 10. This is an eight-species, image-only development
  task, not general image Decision performance or a sealed audit. See [the
  fresh validation report](R2_INATURALIST_FRESH_VALIDATION_READOUT_20261029.md).
- The subsequent pages 41–50 expansion yielded only 27 train and 14 validation
  records across two of eight species after identity exclusions. It fails the
  frozen minimum of 700 train / 200 validation with all classes represented;
  no images, features, metrics, or checkpoint were produced. Do not train on
  these 41 rows or relax the gate. See [the coverage gate](R2_INATURALIST_PAGES_41_50_COVERAGE_20261030.md).

## R2 multimodal gate

The one-time Perception Test source-candidate validation transfer remains
diagnostic-only: joint audio/video scored `52.17%` set Accuracy versus
`55.07%` video-only, while mismatched audio reduced Accuracy to `41.30%`.
This shows sensitivity to audio alignment but does not establish reliable
joint improvement or a task that requires both modalities. R2 M-2 remains
open; do not train or tune another fusion head on the exposed cohort. No new
admitted synchronized AV/sensor task was found in this recheck.

The target-branch CPU tests for the label-blind Perception control freezer pass
(5 tests), and Ruff check/format pass. The frozen-control utility was not run
against an exposed cohort to generate another evaluation; the already saved
input-control results remain the evidence source.

## Decision and next gate

No new GPU training is justified by these results: the latest iNaturalist
expansion fails its predeclared sample gate, and the Perception validation is
already exposed and did not pass joint-quality criteria. The existing
clean-dev-v2 Teacher continuation is recorded separately as complete at 2,048
steps; it is not a substitute for Path A or M-2 evidence. Continue source/task
admission for a rights-cleared, independently grouped task requiring multiple
inputs, and close the R0 deployment-device and task-specific release criteria
before any promotion. No sealed audit was accessed.
