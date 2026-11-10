# VideoMAE global-feature count separability probe — 2026-11-10

## Frozen question and limits

This one-shot diagnostic asks whether the existing 384-dimensional, globally
pooled VideoMAE V2 representation carries linearly accessible object-count
information on the frozen Clevr-4 count task. It follows the count-error
diagnosis in `R2_VIDEOMAE_CLEVR4_100K_PATH_A_RESULTS_20261110.md`.

The predeclared candidate is a single closed-form multi-output ridge readout
with regularization `lambda=1.0`, fit on all 72,010 train images and evaluated
once on the same 1,000-image selector pool used by the earlier fixed Path A
experiment. Input coordinates are standardized with train-only means and
standard deviations (zero-variance coordinates use scale 1). Targets are
centered one-hot count classes 1–10; prediction is the highest ridge score.
No hyperparameter or seed search is performed. The primary metric is selector
count Accuracy, with macro recall, per-class recall, and confusion matrix as
diagnostics. Ridge scores are not treated as calibrated probabilities.

This selector was already used to select the earlier MLP checkpoint, so this
is an exposed development diagnostic, not a blind evaluation. It must not be
used to make a product-quality claim or to reopen the 5,296-image one-time
reporting pool. No reporting-pool features or labels, official validation, or
sealed-audit data are inputs. The existing checkpoint, corpus, and result files
remain read-only.

## Inputs and isolation

- Base branch/commit: `codex/path-a-videomae-clevr4-100k-20261110` at
  `7d65f11e`.
- Source archive: existing Clevr-4 100k v1 archive, 36,057,238,315 bytes;
  source receipt pins SHA-512
  `7e8454f9f44d7b527eb20252a687b15b4cd28f3e9c79cf6147317971034c367b64bf6bba98d2a66f64b26f1e7eea2105a8acf221198342e89a844fd0007cc94c`.
  Each consumed image is checked against its frozen individual SHA-256.
- Train media/label hashes: `e48687af785a74864d2d4eb6e632a50d381d821a3ade8c2227eda7624647f410` /
  `45de9d36d557619789dd6160e808f7ff0c531f06a298f0b2af80c9720103b695`.
- Selector media/label hashes: `4c551e91d4e02bd896380476d0e59f1348a01ed3f95a5eed7e0d39cb9a311dee` /
  `c08af3839089638d081dd31c1118be7256418e6c2440a70a55f79e50456fb920`.
- Encoder: OpenGVLab VideoMAE V2 ViT-Small K710, frozen, 22,153,286
  parameters; source SHA-256
  `d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb`,
  checkpoint SHA-256
  `24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172`.
- The new extraction cache contains only train and selector IDs. It is written
  to a unique external attempt directory on D:; no existing artifact is
  overwritten.

## Interpretation rule

If the fixed ridge readout is materially better than the selected MLP on this
same selector, the global feature may contain more accessible count signal
than the MLP used; this does not demonstrate fresh generalization. If it is
similar or worse, that is consistent with a limitation of the pooled feature
or the linear probe, but does not prove VideoMAE or its spatial tokens lack
count information. Either result is diagnostic only; no model is promoted.

## Execution record

- Code commit: `6c664ff9313351fb4a84f63ee9ffce90d5518001`.
- Exact commands:

  ```powershell
  python -m scripts.videomae_count_feature_probe prepare `
    --inputs C:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-inputs-20261110-attempt-03 `
    --output D:\CodexArtifacts\pretrained-reuse-r2\videomae-count-separability-20261110-attempt-01\prepared

  python -m scripts.extract_videomae_clevr4_frozen_features `
    --archive C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-videoprism-v0\clevr_4_100k_v1.full.zip `
    --manifest D:\CodexArtifacts\pretrained-reuse-r2\videomae-count-separability-20261110-attempt-01\prepared\media-manifest.jsonl `
    --output D:\CodexArtifacts\pretrained-reuse-r2\videomae-count-separability-20261110-attempt-01\features-train-selector `
    --batch-size 8

  python -m scripts.videomae_count_feature_probe analyze `
    --prepared D:\CodexArtifacts\pretrained-reuse-r2\videomae-count-separability-20261110-attempt-01\prepared `
    --features D:\CodexArtifacts\pretrained-reuse-r2\videomae-count-separability-20261110-attempt-01\features-train-selector `
    --output D:\CodexArtifacts\pretrained-reuse-r2\videomae-count-separability-20261110-attempt-01\ridge-v1
  ```

- Feature run: 2026-11-10 05:12:21.836–06:12:33.809 UTC; 3,612.172 seconds
  extraction wall time, 3.063 seconds model load, 1.235 GB peak CUDA
  allocation. Local hardware/software: RTX 3080 Laptop 16 GiB, Python 3.11.9,
  PyTorch 2.6.0+cu124, CUDA 12.4, NumPy 2.4.6, FP32. No cloud or paid compute.
- The archive SHA-512 was not recomputed; the frozen archive byte length
  matched, and all 73,010 consumed images passed their per-image SHA-256
  checks. The VideoMAE source/checkpoint hashes matched the pinned values.
- Extraction result SHA-256:
  `74700a81de136d5bda3100d1ab80b887bc8b946cf950bb7aa694c859ef61ee39`.
  The `[73010,384]` FP32 feature archive is 107,728,297 bytes with SHA-256
  `a23b09988149cc0b60fb1ad89f752b4a796b6a046a68c1d8d7c9b295dd217611`.
  Extracted media rows match the input manifest exactly
  (`9a62c867aff48392407280469109eea9d2fa56ee24f104a8fb70e65d7d12e329`).
- Prepared split receipt SHA-256:
  `67dfeb5c5107d849c59f94089697231789160c1d10cad11034f15862e7b24bb1`.
  The prepared media manifest contains only train+selector records and hashes
  to `9a62c867aff48392407280469109eea9d2fa56ee24f104a8fb70e65d7d12e329`;
  the count-label file SHA-256 is
  `e94f2cb2997f0ece24db7be7ba5ddce8a526d2350b53efe4c0807eb67e446754`.
- Analysis result SHA-256:
  `c1b625403af78e5ba1ce0a317b3fa2efca29603e8a6ecd5d0fbf7ed6434681a7`.
  Selector prediction JSONL SHA-256:
  `f4d83f23b853a8f2dab9549562dbf8d01bdc9a7553b2f3a1d990ccdbb044abcb`.
  Fit+prediction took 0.553 seconds on CPU. GPU was idle for this step.
- All external outputs are under
  `D:\CodexArtifacts\pretrained-reuse-r2\videomae-count-separability-20261110-attempt-01\`.
  Existing VideoMAE run, weights, inputs, and checkpoints were not changed.

## Result

The fixed ridge obtained 24.40% selector Accuracy and 27.60% macro recall.
The prior selected 52,682-parameter MLP scored 42.60% count Accuracy on these
same 1,000 selector images. Ridge is lower by 18.20 percentage points; this is
not a blind comparison because the selector already selected the MLP and was
used for this follow-up diagnostic.

| Count | Support | Ridge recall |
|---:|---:|---:|
| 1 | 39 | 100.00% |
| 2 | 90 | 7.78% |
| 3 | 107 | 9.35% |
| 4 | 101 | 11.88% |
| 5 | 106 | 9.43% |
| 6 | 123 | 30.08% |
| 7 | 123 | 13.82% |
| 8 | 95 | 6.32% |
| 9 | 94 | 2.13% |
| 10 | 122 | 85.25% |

This says count is poorly accessible to this particular linear readout over the
global 384D feature. The MLP's higher score shows that the result does **not**
mean the pooled feature has no count information; a fixed nonlinear head can
decode more of it. It remains plausible that discarded spatial/patch structure
is an additional bottleneck, but this experiment does not test that claim.
Ridge scores were not treated as probabilities, so NLL/Brier/ECE are not
reported for this diagnostic. The result is development-only and cannot be
used to promote the image path or infer broader visual ability.

## Checks and next action

- `ruff check .`: pass.
- Full CPU suite: `2,889 passed, 11 skipped`.
- New focused probe tests: `7 passed`.
- The 32-image real-train CUDA smoke passed before full extraction; its separate
  attempt was not reused for the fit.
- The next justified experiment is a single frozen spatial/token readout with
  a newly frozen **train-only grouped holdout** and one fixed selector-free
  comparison. Do not reuse the 5,296-image reporting subset or reinterpret the
  prior selector as blind. First confirm the VideoMAE token path and memory
  cost, then fit only if the held-out split is disjoint and reproducible.
