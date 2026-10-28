# DROID unseen-site shard sample (2026-10-28)

## Purpose and outcome

The previous DROID next-action development study used official-train shards
00000–00019 and already observed a four-site validation. Before acquiring more
data for another site-held-out run, this audit tested whether a small,
label-blind sample from later shard indices exposed any collection site absent
from that earlier cohort.

Three additional fixed indices (00466, 00759, 01632) were selected by the
lowest salted SHA-256 ranks among indices 00023–02047, with at least 20 shard
indices between selections. The selected shards contain 138 episodes, 137 of
which meet the existing 8-observation / 25%-of-episode eligibility rule. All
three shards contain only sites from the same 14-site set seen in the first 20
shards; the sample contained **zero new sites**. Across the six added shards
(00020–00022 plus the three later shards), the audit found 276 unique source
IDs and recording groups, no overlap with the previous 20-shard cohort, no
repeated IDs/groups, and no exact serialized-record duplicates.

This is evidence about six sampled shard files only. It does **not** prove
that the complete 2,048-shard release has only 14 sites. It provides no reason
to keep downloading similarly small samples to seek a fresh site, and it does
not admit a new model fit or validation claim. No labels, action values, media
frames or audio were decoded; no checkpoint was loaded; no training, sealed
audit or GPU work occurred.

## Download timing and storage

The first measured batch was shards 00020–00022. The second was the
preselected 00466/00759/01632 sample. All six transfers matched the GCS
generation and byte size plus the publisher's GCS MD5; SHA-256 receipts are
listed below. Throughput uses MiB/s (bytes divided by 1,048,576).

| Batch / shard | Bytes | Transfer seconds | MiB/s |
|---|---:|---:|---:|
| 00020 | 967,984,899 | 38.412* | 24.033 |
| 00021 | 1,193,949,950 | 59.405 | 19.168 |
| 00022 | 917,042,094 | 40.763 | 21.455 |
| **First-batch mean** | **3,078,976,943** | **138.580** | **21.552 mean / 21.189 aggregate** |
| 00466 | 1,034,736,346 | 48.359 | 20.406 |
| 00759 | 1,362,668,663 | 52.579 | 24.716 |
| 01632 | 769,377,906 | 33.336 | 22.010 |
| **Second-batch mean** | **3,166,782,915** | **134.274** | **22.377 mean / 22.492 aggregate** |
| **All six** | **6,245,759,858** | **272.854** | **21.965 mean / 21.830 aggregate** |

The first-shard rate projected about 84 seconds for the remaining two shards;
they took 100.168 seconds. The initial live estimate used an incorrect
remaining-byte total; the table gives the corrected projection. After the
first two shards of batch two averaged 22.561 MiB/s, the last 769 MB was
projected at about 33 seconds and completed in 33.336 seconds. Use the latter
observed rate, around 22 MiB/s, only as a rough estimate for similarly sized
GCS objects; the first batch shows meaningful per-shard variation.

The six files total about 6.25 GB decimal (5.82 GiB) and remain outside Git.
They were saved under separate, immutable attempt directories. At the initial
check C: had 35.78 GB free; the download script enforced a 2 GiB reserve.

## Reproducibility and integrity

- Source: Google Research DROID 1.0.0 official train shards, pinned dataset
  metadata SHA-256 `8cbd8f70428943515a67e2a571267c4de2eb71f33957c0269d8fcd5608951e4f`.
- Earlier cohort records SHA-256:
  `b810e8a8141dce6f877f1ff5584221785cbdb2b079a5b7a55ee6e5392cd0ef04`.
- Sample salt: `droid-r2-unseen-sites-v1`; candidates 23–2047; 3 smallest
  SHA-256 ranks with minimum index spacing 20. The pure selector is covered by
  `tests/test_droid_site_holdout.py`.
- Frozen sample manifest SHA-256:
  `bfd50abbe9a87f3bdcbe44c3ebdff2144ea596ae558d699b350cdd177b32f43e`.
- Metadata-only audit result SHA-256:
  `738231fb8bb100ae0d000a663ae6dee824d42bf3dd5dfe8e1c4e3f75ea1dad4d`.
- Shard SHA-256: 00020 `17ddcfa82b3612cfc00272b4d88a2d785a5377fa3f9725352ff9772eef758d73`;
  00021 `11d3f12950beb1cf7754b40cf486419cff1b987f602118d81223285afecba2b1`;
  00022 `e0c5564ea4b06b6c1552e9c53895c450903b4bcd2de0017f117cb92cb94303a1`;
  00466 `6d8316537912dc65cd4924f2f4f9df4950269d564a2e88da53eeca5dc1632591`;
  00759 `d827d53c865854eeea8194d4a889be1f0e96e3a51916d7d7e62006efeb6e0203`;
  01632 `6de00c548fcf7c25e54ba183701f451d56e0d502e77fae86e55597c8f4d5225a`.

The audit runner verifies each input size/SHA, pinned `dataset_info.json` shard
record counts, stable episode and recording-group IDs, duplicate serialized
records, and site overlap with the previous cohort. It reads action-array
lengths only to count eligible rows; it never reads target action values.
Reproduction requires the six external shard files, their receipts, the
previous external `cohort-records.jsonl`, and the pinned dataset metadata:

~~~powershell
py -3.11 -X utf8 scripts\audit_droid_unseen_site_sample.py `
  --inputs "20=C:\CodexArtifacts\pretrained-reuse-r2\droid-next-sites-acquisition-20261028-attempt-02\shard-00020.tfrecord" `
  --inputs "21=C:\CodexArtifacts\pretrained-reuse-r2\droid-next-sites-acquisition-20261028-attempt-03\shard-00021.tfrecord" `
  --inputs "22=C:\CodexArtifacts\pretrained-reuse-r2\droid-next-sites-acquisition-20261028-attempt-03\shard-00022.tfrecord" `
  --inputs "466=C:\CodexArtifacts\pretrained-reuse-r2\droid-unseen-site-sample-20261028-attempt-02\shard-00466.tfrecord" `
  --inputs "759=C:\CodexArtifacts\pretrained-reuse-r2\droid-unseen-site-sample-20261028-attempt-02\shard-00759.tfrecord" `
  --inputs "1632=C:\CodexArtifacts\pretrained-reuse-r2\droid-unseen-site-sample-20261028-attempt-02\shard-01632.tfrecord" `
  --receipts C:\CodexArtifacts\pretrained-reuse-r2\droid-next-sites-acquisition-20261028-attempt-02\verified-recovery.json `
  --receipts C:\CodexArtifacts\pretrained-reuse-r2\droid-next-sites-acquisition-20261028-attempt-03\download-receipt.json `
  --receipts C:\CodexArtifacts\pretrained-reuse-r2\droid-unseen-site-sample-20261028-attempt-02\download-receipt.json `
  --selection C:\CodexArtifacts\pretrained-reuse-r2\droid-unseen-site-sample-20261028-attempt-02\selection.json `
  --previous-cohort C:\CodexArtifacts\pretrained-reuse-r2\droid-expansion-sample-20261028-attempt-01\cohort-records.jsonl `
  --dataset-info C:\CodexArtifacts\pretrained-reuse-r2\droid-full-metadata-screen-20261026-attempt-01\dataset_info.json `
  --output C:\CodexArtifacts\pretrained-reuse-r2\droid-unseen-site-sample-20261028-attempt-02\site-metadata-audit-new.json
~~~

The `new` output filename is required because the audit refuses to overwrite a
prior receipt. The first acquisition attempt stopped before transfer because
a manually transcribed MD5 value was truncated. A second transfer completed
but its old-runtime MD5 conversion helper failed; the preserved `.part` was
then verified with a PowerShell-compatible hex conversion and promoted without
redownloading. One sample-runner attempt also stopped before transfer due to
PowerShell boolean syntax. These failures did not alter the input shards; the
successful integrity receipts above are authoritative.

## Decision

Do not start another DROID site-held-out fit from these files. The next useful
DROID step would require a different, preregistered group-holdout task with
fresh observations and matched video/sensor controls; it must be described as
same-site development evidence unless an independent site is actually found.
Otherwise, screen a different product-usable source. The prior DROID video +
sensor result remains unchanged: sensor-only beat joint VideoPrism + sensor on
the observed four-site validation, so this acquisition does not reverse that
result or close R2 M-2.
