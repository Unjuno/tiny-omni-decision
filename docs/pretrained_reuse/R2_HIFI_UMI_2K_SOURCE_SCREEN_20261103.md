# HiFi-UMI-2K source/task screen — 2026-11-03

## Disposition

HiFi-UMI-2K is a plausible video-plus-state candidate, but remains
`REVIEW_NOT_ADMITTED`. No data or episode Parquet, video media, model, or
quality result was obtained. The publisher README and two small metadata JSON
files were fetched into memory at a pinned revision and hashed; no copy was
written into the workspace. No sealed audit was accessed.

## What the pinned metadata establishes

The publisher's Hugging Face repository is pinned at
`a53b7b5784afdd50b2fda9195c9f724ef75ffdaf`; its API reports 5,176 files and
`cc-by-4.0`. The publisher card explicitly says commercial adaptation is
allowed with attribution. It describes 2,000 hours, 16 TB, synchronized video,
bimanual state/action trajectories, language task annotations, and six camera
views. Audio is not listed as an available modality. The technical report
describes the capture system and manipulation-policy experiments; those
published results are not Decision-model quality results.

The sampled pinned part `chunk-0000/part-0000` has 1,125 episodes, 478,810
frames, 38 tasks, and 25 fps. Its `info.json` declares only `train: 0:1125` and
contains no scene/site/environment field. The card describes episode metadata
fields but does not document a scene identifier; the episode Parquet itself
was not opened. The card's `>480` scene number is explicitly attached to the
larger source corpus, not clearly to the released 2,000-hour subset. A
secondary UMI dataset index says the release spans 110+ scenes, but the
episode-level IDs and split cannot be verified from that statement.

The sampled part is 40,549,578,578 bytes (37.765 GiB); its six MP4s total
40,467,521,713 bytes. The whole release is reported as 16 TB. At the screen's
1.9308 GiB of local free space, no whole sampled part can be acquired. HTTP
range streaming was not tested.

## Open gates

- **Independent grouping and split:** only one train range is documented.
  Scene IDs and a train/validation/audit group split are unverified.
- **Rights:** the publisher's CC BY 4.0 declaration is product-compatible as
  stated. The reviewed card does not describe consent or publicity/privacy
  scope for the human-recorded videos, so that scope remains unknown.
- **Decision task:** the source supplies continuous next-state actions, not a
  fixed supplied-option Decision benchmark. A calibrated candidate-action
  task and controls proving that both video and state are needed remain to be
  designed and frozen.
- **Resource feasibility:** a single sampled part exceeds current local free
  space by a wide margin; no range-streaming feature extraction has been
  demonstrated.

Do not acquire media, extract features, or fit a readout from this candidate
until those gates are resolved. The machine-readable snapshot, including
revision, file hashes/sizes, task and rights dispositions, is
[`hifi-umi-2k-source-screen-20261103.json`](../../manifests/pretrained_reuse/hifi-umi-2k-source-screen-20261103.json).

## Sources

- [Publisher dataset card and license](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K)
- [Pinned publisher repository revision](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K/tree/a53b7b5784afdd50b2fda9195c9f724ef75ffdaf)
- [Technical report](https://arxiv.org/abs/2607.25895)
- [Secondary UMI dataset index](https://umi-data.github.io/)
