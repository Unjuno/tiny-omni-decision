# R2 CATER temporal-video source screen — 2026-11-13

## Disposition

**Status: `REVIEW_NOT_ADMITTED` pending data-scope and accessibility checks.**
CATER is a promising synthetic video-only temporal Decision candidate, but this
metadata screen does not authorize payload acquisition, model fitting, or
quality claims. No CATER videos, scene JSONs, train/validation labels, or model
features were downloaded or opened. No sealed or official validation/test labels
were accessed.

## Pinned source and product fit

- Official repository: <https://github.com/rohitgirdhar/CATER>.
- GitHub API pinned `master` to commit
  `13a19643f1a2fb24e931df25abd74353e4f2fdcb` (2020-12-18).
- The repository has an Apache-2.0 `LICENSE` and says “CATER is Apache 2.0
  licensed.” The generator's `generate/data/` tree contains Blender scene,
  material, and shape assets. The README says the project is built on the CLEVR
  codebase. The source screen did not independently verify the grant for every
  pre-generated video or the provenance/rights of every generated asset; treat
  Apache as promising source evidence, not a completed dataset-rights audit.
- Official [project description](https://rohitgirdhar.github.io/CATER/) says
  CATER uses controllable synthetic 3D videos for compositional action and
  temporal reasoning, including long occlusions and recursive container
  relations. The [generator documentation](https://github.com/rohitgirdhar/CATER/blob/13a19643f1a2fb24e931df25abd74353e4f2fdcb/generate/README.md)
  describes pre-generated videos, scene metadata, task lists, train/validation
  lists, and `train_subsetT` / `train_subsetV` partitions. For `localize`, it
  describes 4x4 and 8x8 variants.
- A suitable Path A task, if admitted, is a fixed video-only supplied-option
  Decision: use the upstream 8x8 snitch localization label and return a
  calibrated distribution over the 64 upstream-defined cells. Keep raw video
  input; do not use scene JSON or labels as model features. This tests a narrow
  temporal synthetic task only and cannot establish natural-video quality,
  cross-modal fusion, or the product target.

## Split, exposure, and access gates

- A repository-wide search of this tracked worktree found no CATER references.
  The known R1/R2 artifact roots
  `C:\CodexArtifacts\pretrained-reuse-r1`,
  `C:\CodexArtifacts\pretrained-reuse-r2`,
  `D:\CodexArtifacts\pretrained-reuse-r1`,
  `D:\CodexArtifacts\pretrained-reuse-r2`, and
  `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2` contained no
  CATER-named run directory. This is a bounded exposure check, not proof that
  no other branch, generic run directory, or external cache ever used CATER.
- Generator code maps each scene JSON to a corresponding video filename and
  shuffles video records for its train/validation split. This screen did not
  prove that scene configurations are unique or that the published subset
  files are disjoint by normalized scene content. Before any fit, hash the
  scene JSON and decoded video content, verify exact and normalized overlap,
  and group by underlying scene identity. Do not assume a video ID alone is an
  independent scene.
- The official README points to a CMU Box share and provides direct ZIP links
  for video, scene, and list files. A metadata-only HTTP `HEAD` check to the
  documented Box share and the `all_actions` videos/scenes/lists URLs returned
  HTTP 404 from this workstation. No `GET` or alternate download route was
  attempted. Availability and exact byte sizes therefore remain **UNKNOWN**;
  the error does not prove the files are unavailable to all clients.

## Next gate

Keep this candidate unadmitted until a bounded metadata-only follow-up confirms
the exact payload license scope, an accessible source with stable size/hash
receipts, and scene-grouped train/selector isolation. If those pass, freeze the
upstream `train_subsetT` and `train_subsetV` cohorts, keep official `val.txt`
and any test set unopened, and preregister temporal-order, frame-shuffle,
position-prior, and option-order controls before one frozen-feature fit. Use
the verified VideoMAE V2 CUDA runtime as the candidate encoder; its existing
runtime smoke is not a CATER quality result. If rights, source access, or group
integrity fails, stop this candidate and retain the negative gate without
installing a new generator environment or fetching media.
