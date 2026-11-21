# HapTile multimodal Decision source screen — 2026-11-22

## Disposition

`REVIEW_NOT_ADMITTED` pending task and grouping preflight. This is the strongest
current video+tactile development candidate found: the primary release is
public and CC BY 4.0, with 38 manipulation tasks, language instructions,
synchronized vision/tactile/state/action streams, and a published tactile
ablation. The paper reports 1,726 demonstrations from nine operators. The
pinned primary release has 1,700 trajectory entries; pinned Voxel51 conversion
revision `c8593da98d3facb7a9293842514f0ee4ad3f2318` has 1,699 samples. The
per-task counts differ only at `move_plush_toy` (71 source entries vs. 70
converted samples). The conversion card says one empty trajectory was omitted,
but the omitted episode itself was not opened to verify its task or contents.
The paper's 26-count difference from the pinned source remains unexplained.

The release has no documented train/validation/test split or per-episode
operator identity mapping in the metadata screened so far. The paper says all
data came from one laboratory scene. A task-held-out development split can
test transfer across tasks, but it cannot establish unseen-operator or
unseen-environment performance. Do not call task-held-out scores a blind
population estimate.

## Source and evidence

- Primary release: <https://huggingface.co/datasets/HapTile2026/HapTile>
- Pinned source revision: `54887384f1fe9e93b799e104194a582f9d8840b8`
- Source card license: CC BY 4.0; public and ungated.
- Source repository at the pinned revision: 38 task ZIPs totaling
  16,001,478,139 bytes; all repository files including `.gitattributes` and
  `README.md` total 16,001,480,670 bytes. No archive content was fetched.
- The 38 task ZIPs contain 1,700 `trajectory.h5` entries (27–71 per task).
  Central-directory filenames show the only source/conversion count difference
  is one `move_plush_toy` entry. The conversion card says one empty trajectory
  was omitted, but we did not inspect that file, so its identity is not proven.
  The paper's 1,726 count remains 26 above the pinned release.
- Paper: <https://arxiv.org/abs/2606.04825>, v2.
- Paper statistics: 1,726 demonstrations, 38 tasks, nine human operators,
  synchronized observations/actions at 15 Hz, and one laboratory scene.
- Voxel51 conversion: <https://huggingface.co/datasets/Voxel51/HapTile> at
  revision `c8593da98d3facb7a9293842514f0ee4ad3f2318`; its card reports 1,699
  samples and 6.31 GB. It documents a single `train`
  split and unique per-episode session timestamps, but no operator field.
- The pinned conversion README is 15,985 bytes (SHA-256
  `f0462ecc55d172e8a3c51ac5c874969e6f0482f093812482d28f730ba0558fbb`). A
  machine-readable comparison of all 38 per-task counts is stored outside Git
  at `C:\CodexLocalArtifacts\pretrained-reuse-r2\haptile-source-metadata-screen-01\conversion-count-reconciliation.json`,
  SHA-256 `6bdbe176adf3b76abfbebcce25a7c28c7a18a0131bcdbdf699a94c56893326c1`.
- Pinned upstream Hugging Face API metadata is stored outside Git at
  `C:\CodexLocalArtifacts\pretrained-reuse-r2\haptile-source-metadata-screen-01\`.
  The info response is 1,961 bytes with SHA-256
  `55d5763e7f9d7ca725f7a0d97b7157a8c592d73dfab43444115aa311a95e6e50`.
  `pagination-receipt.json` records the original 41-entry API tree response
  (37,644 bytes; SHA-256
  `d8537e2e725f4715f62e48f821ff612669faf8c604f05c18d1815dbddf4d557f`). The
  parsed/reformatted `file-tree-complete.json` used by the index scan is 50,506
  bytes (SHA-256
  `57af93cac7a41c56d11968c191de668ac084390232d412d728b378d8eaddec48`).
- Read-only ZIP index scan receipt (outside Git):
  `C:\CodexLocalArtifacts\pretrained-reuse-r2\haptile-source-metadata-screen-01\zip-index-range-scan.json`,
  SHA-256 `24bfc28a4f477c7818c4199a90bd70ac891ba9237fd0c62e2914469b9428d281`.
  Its companion `count_zip_central_directory.ps1` reproduces the count from
  the pinned revision and requests only the last 65,536 bytes of each ZIP.

## Modeling cautions

The paper evaluates vision-only, vision+tactile, and vision+tactile+marker
motion on four tasks. Tactile gains vary by task and model; some combinations
decrease success. This supports testing adaptive fusion, not assuming tactile
always helps.

The haptic feedback channel is computed from tactile marker motion and then
mapped to vibration levels. Do not treat same-time haptic levels as independent
ground-truth labels when tactile input is present. A more defensible Decision
target is a causal future action choice derived from the demonstrated action
trajectory, with the observation window ending before the target action.

## Next admission gate

Before downloading media or fitting a model:

1. Use only the pinned 1,700-entry source revision or pinned 1,699-sample
   conversion. Source/conversion per-task counts match except for one
   `move_plush_toy` entry, consistent with—but not proving—the conversion
   card's one-empty-trajectory omission. Keep the paper's 1,726 count separate
   and unresolved; do not use it as the corpus count.
2. Verify whether the source provides stable task, asset, and operator/session
   identifiers. If operator IDs cannot be mapped to episodes, explicitly limit
   claims to task-held-out development evidence.
3. Freeze a task-held-out split by task identity and a causal action-choice
   construction before reading held-out action labels.
4. Check action-code support using train groups only; preserve all four
   observation arms (vision+instruction, tactile+instruction,
   vision+tactile+instruction, and state-only control) under one shared split.
5. Keep sealed/reserved task groups unopened until architecture and checkpoint
   selection are frozen. No model, features, data payload, or audit labels have
   been loaded for this screen.

No training, feature extraction, model loading, or sealed-audit access occurred.

