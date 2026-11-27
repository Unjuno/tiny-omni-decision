# IntPhys training-data coverage preflight — 2026-11-28

## Decision and scope

The prior IntPhys Path A runs used only `train.1.tar.gz` (3,750 of the
publisher's 15,000 possible-only training videos). This branch prepares a
single data-coverage experiment using all four official train shards. It does
not modify the existing 8-frame or 16-frame runs, checkpoints, caches, dev
predictions, or hidden test. The already-observed 90-scene development set
will not be evaluated again for this data-scaling experiment.

The question is limited to whether broader unique training coverage improves
the frozen-feature next-frame prediction objective on a common, source-grouped
internal validation split. It cannot establish a calibrated possible/impossible
Decision result: IntPhys training labels are positive-only, and the old dev
evaluation is already exposed.

## Source and availability audit

The official [download page](https://intphys.cognitive-ml.fr/download.html)
declares CC BY 4.0, 15,000 train scenes, four compressed train shards, and the
following published size/MD5 pairs:

| Shard | Bytes | Published MD5 | Direct URL |
| --- | ---: | --- | --- |
| train.1 | 37,016,506,106 | `3428bcf398441a489111528634530212` | `https://download-intphys.cognitive-ml.fr/train.1.tar.gz` |
| train.2 | 37,111,037,048 | `47dc1c3578e079584f4e3b5b8ab8304b` | `https://download-intphys.cognitive-ml.fr/train.2.tar.gz` |
| train.3 | 37,146,175,637 | `389782958ad058f7a7e1db2b07748259` | `https://download-intphys.cognitive-ml.fr/train.3.tar.gz` |
| train.4 | 37,379,611,745 | `7d345eaa3a33d2abc4a75442eefb277a` | `https://download-intphys.cognitive-ml.fr/train.4.tar.gz` |

A read-only HTTP `HEAD` check against each official URL on 2026-11-28 returned
200, the exact published Content-Length, `Accept-Ranges: bytes`, and these
ETags:

| Shard | Observed ETag | Last-Modified |
| --- | --- | --- |
| train.1 | `"5d9bc2aa-89e5b0efa"` | 2019-10-07 22:56:42 GMT |
| train.2 | `"5d9bc78d-8a3fd7c78"` | 2019-10-07 23:17:33 GMT |
| train.3 | `"5d9bc99a-8a615a895"` | 2019-10-07 23:26:18 GMT |
| train.4 | `"5d9bcb97-8b3ff9c61"` | 2019-10-07 23:34:47 GMT |

The first shard is already locally pinned and hashed (SHA-256
`5f4ce9b2240bdbd57e32ed2d8315300314d30572f39adb0d9e58afa3b27da83a`). Shards
2–4 have not been downloaded or opened. Their combined compressed size is
111,636,824,430 bytes; C: currently has about 61.0 GiB free, so all raw shards
cannot coexist locally. Process each new archive one at a time, verify its
published MD5 and computed SHA-256, preserve the complete derived feature cache
and receipt, then remove only that newly downloaded temporary archive before
fetching the next. The pre-existing train.1 archive and all earlier experiment
artifacts remain untouched. Feature caches and run receipts stay outside Git.

## Data-scaling comparison frozen before acquisition

Use all four shards and the same VideoPrism-B revision, checkpoint, 8-frame
sampling policy, 768-wide per-frame pooled features, 64-wide causal LSTM, and
seed 17. The 8-frame path is the reference because the previously completed
16-frame comparison did not reliably improve the task diagnostic; frame count
will remain fixed here.

1. Extract every official train video, preserving the 100-frame status and
   channel completeness checks plus status, full-RGB, and normalized-input
   identity hashes.
2. Globally group identities across all four shards before making one
   deterministic 80/20 split. Never trust shard-local splits to detect a
   cross-shard duplicate. Require zero cross-split overlap for sample IDs,
   status hashes, full ordered RGB hashes, sampled-input hashes, and groups.
3. Use the same global validation set for both comparisons. Candidate `D-small`
   uses exactly 3,004 training groups selected by the predeclared SHA-256 rank
   from the global training split; `D-full` uses every remaining training
   group. This compares a one-shard-sized unique sample against the full
   training coverage without selecting examples by label or model score.
4. Hold the predictor, seed, AdamW settings (`lr=1e-3`, `weight_decay=1e-4`),
   batch size 64, and optimizer budget at 3,901 updates for both candidates.
   This matches the update count of the existing corrected 8-frame fit while
   varying unique data coverage. Record every validation checkpoint and retain
   both the minimum-validation-loss and terminal checkpoints.
5. Compare only on the common internal validation set. Do not reopen or rescore
   the already-observed IntPhys dev set, the Physical Concepts probe, or any
   hidden/sealed test. Treat next-feature cosine loss as a representation
   objective, not decision accuracy or a calibrated probability.

## Resource gate

The previous 16-frame extraction needed 10,563.6 seconds for one 3,750-video
shard on the local JAX CPU path (peak process working set 4.16 GB). The 8-frame
dev run measured 1.164-second median and 1.230-second p95 forward time per
video; this is not a full-shard throughput guarantee. A linear projection for
15,000 videos is about 4.9 hours of model forward time before archive hashing,
PNG parsing, and I/O. The active RTX 3080 process belongs to the separate
Orbitune job; this pipeline remains CPU-only and will not stop or alter it.
Feature extraction starts only with at least 6.5 GiB of available host RAM
(the prior 4.16 GB peak plus a reserve). No cloud or paid compute is planned.

## Existing evidence retained

- The corrected 8-frame run used 3,004 train / 746 validation videos and
  selected epoch 81. Its fixed dev diagnostic was near chance (Accuracy 0.500,
  AUC 0.492) on 360 videos / 90 scene groups.
- The 16-frame run held those records and settings fixed, but had worse
  validation cosine loss and no reliable dev improvement. Its input-collision
  count decreased from 9/90 scenes to 0/90; the quality change was uncertain.
- Both dev results remain exposed development evidence. Neither can serve as a
  blind estimate for this expanded-data experiment.

No new shard has been acquired, no new model has been trained, and no existing
artifact has been changed by this preflight.
