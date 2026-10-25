# MultiEventVideo MEV source screen (2026-10-26)

## Decision

Keep MEV in **review** status. It is a plausible natural-video event corpus, but
the pinned release does not provide the per-video rights data needed to admit
any assets under the repository's approved-source gate. It also does not
establish a both-audio-and-video-required task for R2 M-2. No video tar shard,
training run, or model inference was started.

## Pinned metadata findings

- Source: `MultiEventVideo/MEV`, repository revision
  `1e9460d5909116807dd45345b174fb91e9244553`.
- The official card reports 25,262 event rows and 8,016 video records. A local
  read of the pinned `events.parquet`, `videos.parquet`, and canonical
  `annotations/mev.json` confirms 25,262 events grouped into 8,016 UUIDs, with
  every UUID's event rows together in the video index.
- The repository has 14 video tar shards totalling about 76.3 GB. Only metadata
  was fetched for this screen. The table and annotation files list UUIDs,
  captions, event segments, and tar members; neither table exposes source
  license, commercial-use, redistribution, source-platform, or source-ID fields.
  The repository file tree contains no separate source manifest.
- The dataset license marks the release `mixed-third-party-licenses`. It says
  MEV-created annotation metadata is CC BY 4.0, but individual video rights
  remain with original rights holders and are governed by source/platform terms.
  The license describes per-video provenance fields that should appear in a
  source manifest when available; those fields are absent from this pinned
  release's distributed metadata.
- The published dataset splits are `events`, `videos`, `sample_events`, and
  `sample_videos`, not independent train/validation/test partitions. The UUID
  grouping is visible, but no separate underlying-source identity is available
  to verify cross-video leakage.

## Reproducibility record

Metadata-only files are retained at
`C:\CodexArtifacts\pretrained-reuse-r2\mev-metadata-preflight-20261026-attempt-01`.
SHA-256 values:

| File | SHA-256 | Bytes |
|---|---|---:|
| `videos.parquet` | `099327781db64fcbf26d205bee689abcff1620faa4cc7d243ba94b07c7c70391` | 33,070,965 |
| `events.parquet` | `1ed5bc3b2bec7b741b6944467671a6b2f1be4c206a3319a4e4dcd34df2796d97` | 67,163,607 |
| `mev.json` | `2a6a06c38dda5a40e629ecdc8800e27495c214754e480344f267f939ad07fd19` | 70,239,613 |
| `README.md` | `7547ef8977a4801da50dbcfe4fe316250362e067e474f81275d957f560281895` | 2,721 |
| `LICENSE` | `265513b35c28defed7c7e08236c74e7e329cfd433193e15043743277d3a371fa` | 5,416 |

## Disposition

Do not download or use video assets until the maintainers provide a pinned
source manifest with an asset-level compatible license, commercial-use status,
and redistribution status. If those rights later clear, a prospective video
task could group all event segments by UUID and separately audit whether the
UUID corresponds to a distinct source video. It would remain a video-only task
unless an independent task contract proves that audio is required and provides
audio-specific labels and controls.

## Official sources

- [MEV dataset card](https://huggingface.co/datasets/MultiEventVideo/MEV/tree/1e9460d5909116807dd45345b174fb91e9244553)
- [Pinned README](https://huggingface.co/datasets/MultiEventVideo/MEV/blob/1e9460d5909116807dd45345b174fb91e9244553/README.md)
- [Pinned license](https://huggingface.co/datasets/MultiEventVideo/MEV/blob/1e9460d5909116807dd45345b174fb91e9244553/LICENSE)
- [Pinned metadata/videos.parquet](https://huggingface.co/datasets/MultiEventVideo/MEV/blob/1e9460d5909116807dd45345b174fb91e9244553/metadata/videos.parquet)
