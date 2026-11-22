# MEV video source screen — 2026-11-14

## Disposition

`REVIEW_NOT_ADMITTED` for model fitting or quality evaluation. MEV is a
potential temporal-video task source, but the pinned public release does not
provide the source-level rights data required by this project's source policy.
No video archive, sample video, or event-label values were opened. No features,
predictions, splits, or model outputs were created.

## Pinned release and metadata-only inspection

- Dataset: `MultiEventVideo/MEV`.
- HF Git `main` HEAD observed with `git ls-remote`:
  `cae7b543ace360d2ce9cca7a21ff799230abd3b1`.
- The pinned tree reports `metadata/videos.parquet` as 33,483,037 bytes with
  LFS SHA-256
  `831f446696cb075ef21d3d97e0cd226c0df75a21c2edcfccb352c21503a13b31`.
- At the same revision, the downloaded `LICENSE` is 4,545 bytes (SHA-256
  `13e62d30ca900c54f9eb3d42619d65bcd917dcd8971e37ce38ab374502088cbf`) and
  `README.md` is 9,272 bytes (SHA-256
  `de441ee9ed2e508d0befa7db1a9cabba4dfd85500e2a97089a938f4dbe080ba1`).
- The file was downloaded over HTTPS to
  `D:\CodexArtifacts\pretrained-reuse-r2\mev-source-screen-20261114-attempt-01\metadata-videos.parquet`;
  its local size and SHA-256 match the pinned LFS metadata.
- The metadata table has 8,016 unique UUIDs and 7,178 distinct non-null
  `raw_720p_sha256` values. Thus raw-media identity must be grouped across
  UUIDs; UUID alone would leave 838 duplicate-identity rows uncollapsed.
- Its schema contains event/video references, captions and media hashes, but no
  per-video source platform, source license, redistribution, commercial-use, or
  takedown fields. Inspection read only the schema, UUID column and raw-video
  hash column; it did not read caption, prompt, event-label, or media content.
- The pinned repository tree contains no source-manifest file. The official
  README describes `events`, `videos`, `sample_events`, and `sample_videos` as
  table/subset names, not experimental train/validation/test splits.

## Rights and task fit

The pinned MEV license releases author-created annotations and metadata under
CC BY 4.0, but leaves video files subject to each source's license or platform
terms. It does not grant blanket rights to those video files. No per-video
source manifest is present in this pinned Hub tree, and the video metadata
table lacks the rights fields needed for a source-level allow decision. The
dataset page describes 8,016 original
videos and 25,262 temporally grounded event clips. A video-only temporal
Decision task could be designed around state changes or event ordering, but
task wording and options would need to avoid copying the event captions into
the question or leaking the target through text.

The source therefore fails the current rights gate before split construction or
model evaluation. Reconsider only if the maintainers provide a pinned source
manifest with enough item-level rights-cleared videos and permissible model
training use. Any later cohort must group all event clips by both underlying
video identity and exact media hash, then freeze new train/calibration/
validation/audit partitions; the current release provides no standard task
split. This screen is video-only and does not qualify the required joint AV or
video/sensor task.

## Primary sources

- [Official MEV project page](https://mev-dataset.github.io/)
- [Pinned MEV dataset README](https://huggingface.co/datasets/MultiEventVideo/MEV/blob/cae7b543ace360d2ce9cca7a21ff799230abd3b1/README.md)
- [Pinned MEV license](https://huggingface.co/datasets/MultiEventVideo/MEV/blob/cae7b543ace360d2ce9cca7a21ff799230abd3b1/LICENSE)
