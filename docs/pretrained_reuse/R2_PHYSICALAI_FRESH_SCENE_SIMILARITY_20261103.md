# PhysicalAI fresh cohort scene/layout similarity audit — 2026-11-03

## Result

The new 4,000-train / 1,000-validation cohort does not provide evidence for a
scene/layout-disjoint evaluation. A label-blind visual review of a fixed random
sample (30 videos per split) shows the same robot, camera, work surface, and
background geometry throughout; object positions and colors vary. The 12
highest DINOv2 cross-split neighbors show the same visual environment pattern.
Together with the earlier eight-pair review, this supports treating the source
as a shared-environment cohort. It does not identify a publisher scene key or
prove every one of the 5,000 episodes belongs to a single scene.

**Disposition remains `REVIEW_NOT_ADMITTED`.** Do not derive a scene split from
these embeddings and do not fit/evaluate a readout on this cohort. The source
metadata lacks scene/layout identity, so an episode split cannot establish the
scene-independent R2 gate. No action labels, Parquet payload, reserved cohort,
model quality, or predictions were inspected. No training or feature readout
was fitted.

## Frozen inputs and integrity

- Dataset: NVIDIA PhysicalAI Robotics Manipulation SingleArm,
  `panda-stack-platforms`, revision
  `73c00c18e7ca108e91846294b20a1d7cd5de9340`.
- Publisher license in the pinned source screen: CC BY 4.0.
- Split manifest:
  `manifests/pretrained_reuse/physicalai-stack-platforms-fresh-action-split-20261103.json`,
  SHA-256 `fb51eb0028c8f55372930548d2e480bbf616debc1a59779f29098d86ce1bbe7f`.
- Acquisition receipt SHA-256:
  `b7a40364f1f931e897430e3057379cbd52ebaa4e1aae823c266d33dcb7e4f634`.
  Its independent verifier passed all 10,000 publisher file hashes, with no
  reserved payload downloaded. The previously completed exact normalized
  32x32 RGB frame check found no train/validation frame hashes in common.
- The audit script checked each selected video against the receipt's exact
  byte length and publisher SHA-256 before decoding. It used the receipt only
  for media path, episode identity, and train/validation assignment; it did not
  open the paired Parquet files.
- The 1,000 reserved-development episode records remained unopened. This audit
  ran on the selected 4,000 train and 1,000 validation videos only.

## Frozen visual protocol

- Encoder: locally loaded `facebook/dinov2-base`, revision
  `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, FP32. Checkpoint SHA-256
  `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`
  (346,345,912 bytes).
- For each decoded video, sample the frame indices nearest 20%, 50%, and 80%
  of the declared frame count. Extract the DINO CLS vector at each frame and
  L2-normalize the mean of the three vectors per episode.
- Compare all 1,000 validation episode vectors with all 4,000 train vectors
  (4,000,000 cross-split pairs). Similarity thresholds are descriptive only;
  they are not an identity classifier or split rule.
- Human-blind visual checks used the 12 highest-scoring cross-split pairs and
  SHA-256 ordered episode sampling (seed string `physicalai-scene-audit-seed17`),
  first 30 episodes per split. The sample rule and IDs are in the machine result.
- No scene identity was inferred from action type, object/action labels, or
  source task outcomes.

## Measurements

| Measure | Result |
| --- | ---: |
| Videos | 5,000 (4,000 train / 1,000 validation) |
| DINO frames | 15,000 (3 per video) |
| Cross-split episode pairs | 4,000,000 |
| Cosine similarity median | 0.824653 |
| Cosine similarity p95 | 0.898082 |
| Cosine similarity p99 | 0.922491 |
| Maximum cosine | 0.980785 (1 pair >= 0.98; 0 >= 0.99) |
| Video decode time | 229.37 s |
| DINO inference time | 169.31 s |
| Total extraction + comparison time | 418.52 s |
| Peak CUDA allocation | 391,544,832 bytes |
| Feature cache | 60.36 MiB |
| Cloud cost | $0 |

All 60 sampled videos visually show a common camera/robot/workspace/background
configuration. The single >=0.98 pair is not treated as a duplicate or a
separate-scene finding. Similarity alone cannot validate a semantic scene
grouping, and the images do not establish the scene distribution of every
unreviewed video.

## Runtime and saved artifacts

Local GPU was an idle RTX 3080 Laptop 16 GiB. The model used a pinned local
checkpoint; no network model load occurred. The command was:

```powershell
python -m scripts.audit_physicalai_fresh_scene_similarity `
  --receipt C:/CodexArtifacts/pretrained-reuse-r2/physicalai-stack-platforms-fresh-action-20261103-attempt-01/download-receipt.json `
  --manifest manifests/pretrained_reuse/physicalai-stack-platforms-fresh-action-split-20261103.json `
  --model-dir C:/CodexArtifacts/pretrained-reuse-r2/dinov2-base-runtime-smoke-01/model `
  --output C:/CodexArtifacts/pretrained-reuse-r2/physicalai-fresh-scene-similarity-20261103-attempt-01 `
  --device cuda
```

The full output is outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/physicalai-fresh-scene-similarity-20261103-attempt-01/`:

- `scene-similarity-features.npz` — 5,000 episode IDs, splits, three-frame
  DINO vectors, and 32x32 RGB thumbnails; SHA-256
  `1fd83fafff30b98d04bc0deca76ed3f19264511d28a7b95a9e7cb3f668397cc9`.
- `scene-similarity.json` — full pair summary and top 100 neighbors; SHA-256
  `4720b7d09bcc57ed4a7d829087fc4273f1bd003efa0b941a3d2413729fd0681b`.
- `top-neighbor-contact-sheet.png` — the 12 nearest cross-split pairs; SHA-256
  `5db7b6efe53584a89b146cb2a0d18c63881a8d7f83533cf3d0407503d59e6cb9`.
- `representative-contact-sheet.png` — deterministic 30+30 video sample; SHA-256
  `4095d76f2ae3ee6cefad611e24f66239a8e367d894c97cc6c8938f2a6995f27c`.
- `progress.json` — final progress counters.

The representative sheet was rendered from the saved thumbnail cache after the
encoder run; it did not decode or evaluate more videos. Its post-run rendering
helper is included in the committed script. No legacy model, data, corpus,
checkpoint, run, or other artifact was deleted or modified. The previous
partial/temp files remain retained because their resume/recovery value has not
been disproved.

Environment: Windows 10 build 26300, Python 3.11.9, PyTorch 2.6.0+cu124,
CUDA 12.4, Transformers 5.6.2, PyAV 18.1.0, NVIDIA driver 616.92, RTX 3080
Laptop 16 GiB. No package upgrades or cloud services were used. The committed
script's final form has SHA-256
`149641bd7558048b8c9cbb36bda88a606bf594eb8761fd39497e9b34428c1b1b`. The
feature extraction process ran before its script hash was captured; later
changes added representative contact-sheet rendering, literal input-hash
guards, and small-smoke count handling. The exact runtime script hash is
UNKNOWN. A later 1+1-video smoke passed with the literal receipt, manifest, and
checkpoint pins enabled. An earlier 1+1 smoke stopped only in sample-sheet
postprocessing because it assumed 30 examples; that handling was corrected.
The saved full-run receipt, manifest, and checkpoint hashes are recorded in the
machine result.

The audit plus cache used about 60 MiB; C: retained approximately 6.24 GiB
free afterward. No old data was deleted to create space. The 26 older partial
or temporary artifacts (~1.765 GiB) remain pending per-source recovery review.

## Checks and next gate

The new CPU tests cover deterministic temporal frame selection, fixed split
counts, exact manifest/receipt identity binding, reserved-holdout rejection,
exact file pins, and split-correct contact-sheet construction. Focused result:
10 tests passed. Full result: 2,826 passed, 11 skipped. Ruff and all four
existing dataset-catalog validations passed.

This closes the scene-screen question for the fresh cohort enough to keep it
out of R2 fitting. Next work must use a source with an auditable independent
scene/layout grouping key, or identify another admissible task/source with a
true both-input requirement and enough locally practical grouped examples.
Do not substitute RoboMNIST's long full-media transfer or Lombard's already
closed exposed validation as a shortcut.
