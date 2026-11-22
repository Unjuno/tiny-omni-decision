# R2 CATER temporal-video source screen — 2026-11-13

## Disposition

**Status: `REVIEW_NOT_ADMITTED` pending data-scope and cross-source identity checks.**
CATER is a promising synthetic video-only temporal Decision candidate, but this
initial source screen did not authorize model fitting or quality claims. Its
original no-payload statement was superseded by the later, bounded
[access/integrity recheck](R2_CATER_ACCESS_RECHECK_20261113.md): only the small
lists and scenes ZIPs and the `train_subsetT` / `train_subsetV` entries were
downloaded/read. The corresponding 3,849 training/selector scene JSONs were
audited. No CATER videos, official validation/test list contents, model
features, or sealed audit data were accessed.

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
- The initial metadata-only check used the `cmu.box.com` host and received
  HTTP 404. The later recheck followed the official redirect to
  `cmu.app.box.com`; the lists/scenes archives were downloaded and hashed, and
  `HEAD` reported the video archive size. No video archive was downloaded.
  See the later recheck for exact receipts and the remaining archive-rights
  limitation.

## Next gate

Keep this candidate unadmitted until the hosted-payload rights scope and
cross-source scene identity with prior Clevr-4 exposure are resolved. A later
bounded central-directory check found the video members use `CATER_new_*`,
while an inspected scene JSON embeds a mismatching `CLEVR_new_*` filename.
The train/selector subsets have zero overlap by archive video ID and exact
normalized CATER scene metadata, but this does not prove independence from
Clevr-4 or detect all visual near-duplicates. Do not download video payloads or
fit a model before those gates pass. If admitted later, freeze only the already
audited train/selector cohorts, keep official `val.txt` and any test set
unopened, and preregister temporal-order, frame-shuffle, position-prior, and
option-order controls before one frozen-feature fit. Use the verified
VideoMAE V2 CUDA runtime as the candidate encoder; its existing runtime smoke
is not a CATER quality result. If rights or identity cannot be resolved, stop
this candidate and retain the negative gate.
