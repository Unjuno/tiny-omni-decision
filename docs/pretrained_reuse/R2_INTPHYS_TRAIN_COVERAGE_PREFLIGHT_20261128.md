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

Before full transfer, three bounded 64 MiB HTTP Range requests were measured
against shards 2, 3, and 4. All returned HTTP 206 with the expected byte range,
file length, and ETag; the temporary fragments were SHA-256 hashed and then
removed. Measured rates were 3.246, 3.260, and 1.222 MB/s (mean 2.576 MB/s,
range 1.222–3.260 MB/s). This projects 111.6 GB at the observed mean to about
12.04 hours of transfer. The receipt is outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2-local/intphys-full-train-coverage-20261128-attempt-01/bandwidth-samples/speed-test-receipt.json`.
These short samples do not establish sustained throughput or reliability.

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
3. From each global train/validation identity group, choose one representative
   video by lexicographically smallest qualified sample ID. Report how many
   duplicate rows were excluded. Use that same unique-group validation set for
   both comparisons. Candidate `D-small` uses exactly 3,004 unique training
   groups selected by the predeclared SHA-256 rank; `D-full` uses every unique
   training group. This compares a one-shard-sized unique sample against the
   full training coverage without selecting examples by label or model score.
4. Hold the predictor, seed, AdamW settings (`lr=1e-3`, `weight_decay=1e-4`),
   batch size 64, and optimizer budget at 3,901 updates for both candidates.
   This matches the update count of the existing corrected 8-frame fit while
   varying unique data coverage. Record every validation checkpoint and retain
   both the minimum-validation-loss and terminal checkpoints.
5. Compare only on the common internal validation set. Do not reopen or rescore
   the already-observed IntPhys dev set, the Physical Concepts probe, or any
   hidden/sealed test. Treat next-feature cosine loss as a representation
   objective, not decision accuracy or a calibrated probability.

The fixed-update comparison runner is
`scripts/run_intphys_data_coverage_comparison.py`; it saves best and terminal
checkpoints, all fixed validation points, train-stream and fixed-train-monitor
losses, validation per-example errors, unique-example consumption, and hashes.
It checks that repeated media identities stay in one global group, then uses
one deterministic representative per group for training and validation. Its
CPU smoke tests do not load the large VideoPrism checkpoint.

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

## 2026-11-28 shard-2 transfer verification follow-up

The pre-existing `train.2.tar.gz.partial` was found at the exact pinned size
(37,111,037,048 bytes) after the recorded curl process had exited. The
monitor receipt marked the transfer complete but explicitly had
`hash_verified=false`; no extraction or fitting was started. Rather than
discarding or redownloading it, the existing fail-closed downloader was run
against the pinned publisher ETag (`"5d9bc78d-8a3fd7c78"`). Because the file
already matched the pinned size, it made zero HTTP requests and verified the
entire local archive:

- Publisher-pinned MD5: `47dc1c3578e079584f4e3b5b8ab8304b` (pass).
- Computed SHA-256: `0085060c9e2cb8143c7ee5f5c7866d8d2e5b70a0cb208e265ae1f1812210a257`.
- Result: 37,111,037,048 bytes, 0 network responses; local full-file hashing
  completed successfully.

The verified archive remains at the same `.partial` path to preserve the
existing artifact layout; the suffix is historical and does not indicate
unverified bytes after this receipt. No tar members have been inspected or
extracted, no features/model were produced, and no GPU was used. This closes
only the transfer-integrity gate for shard 2. It does not make the planned
four-shard comparison complete: shards 3 and 4 remain absent, and the global
four-shard identity split and coverage fit have not run. The RTX 3080 remains
owned by the unrelated Orbitune process; C: has approximately 24.6 GiB free
after hashing.

The broad-coverage plan also needs its resource decision updated before model
work: the initial preflight assumed all four compressed shards could be
processed one at a time with the 20 GiB disk reserve, but the current free
space cannot hold even one additional ~37 GiB shard under that reserve. Do not
start shards 3 or 4 until an authorized storage plan or a lower-footprint
streaming design is reviewed. Existing source archives, features, checkpoints,
predictions, validation/dev artifacts, and other branches remain unchanged.
